"""Build idtracker.ai id-image HDF5 files from a SLEAP predictions file.

This is the hand-off point between the SLEAP half of the port and the
idtracker.ai half: everything upstream of here is keypoints, everything
downstream (fragmentation, contrastive training, assignment) reads these files.

Output layout mirrors idtracker.ai's own (``session.py:716``): one file per
episode, ``id_images/id_images_{episode}.h5``, each holding

    id_images     (n, HEIGHT, WIDTH) uint8   the canonical hull crops
    crossings     (n,)               bool    True = overlaps another instance
    identities    (n,)               int64   all zeros; filled after tracking
    frame_numbers (n,)               int64   absolute frame each image came from
    global_index  (n,)               int64   non-resetting id across episodes
    instance_idx  (n,)               int64   position within its LabeledFrame

The first three are upstream's; the last three are ours.

DELIBERATE DEVIATIONS FROM UPSTREAM
-----------------------------------
1. NON-SQUARE IMAGES. Upstream allocates ``(n, size, size)``
   (``list_of_blobs.py:253``). Flies are ~1.6x longer than wide, so a square
   frame wastes pixels; the plan note fixes non-square id-images as a hard
   constraint. Shape is (HEIGHT, WIDTH) -- numpy row-major, so HEIGHT first.

2. GLOBAL, NON-RESETTING IMAGE INDEX. Upstream's ``id_image_index`` restarts at
   0 in every episode (``list_of_blobs.py:238``) and ``load_id_images`` indexes
   the episode's dataset directly with it (``py_utils.py:468``). Per spec the
   index here does NOT reset, so it can be used to reference frames globally.

   That makes ``global_index`` unusable as a direct h5 row index for any episode
   past the first. The conversion is stored, not left implied: each file carries
   an ``index_offset`` attribute, and

       row_in_this_file = global_index - index_offset

   recovers upstream's local index exactly. ``local_index`` is also written out
   so nothing downstream has to do the arithmetic.

3. CROSSINGS ARE GEOMETRIC, NOT LEARNED. Upstream trains a CNN to classify
   blobs as individual-vs-crossing (``crossing_detector.py:81``) because
   thresholded blobs merge when animals touch. SLEAP already separates
   overlapping animals into distinct instances, so the classifier is not needed
   and is skipped entirely: an instance is a "crossing" iff its convex hull
   intersects another instance's hull in the same frame. Note this is a
   different quantity than upstream's -- upstream's crossing blob is ONE blob
   containing N animals, ours is N instances that happen to overlap. Semantics
   downstream (exclude from individual fragments) are the same.

Run:
    /opt/anaconda3/envs/sleap_id/bin/python src/sleap_idtracker/build_id_images.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import sleap_io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from episodes import episodes_from_slp  # noqa: E402
from rotate_boxes import CENTROID_NODE, TOP_NODE, heading_angle_deg  # noqa: E402

import config  # noqa: E402

REPO = config.REPO
SLP = config.SLP
OUT_DIR = config.OUT_DIR

# All set in config.py. The hull crossing knobs are DIAGNOSTIC: the `crossings`
# dataset written below is read by nothing downstream -- the load-bearing
# determination is `is_overlapping` in build_overlaps.py. See config.py.
BOTTOM_NODE = config.BOTTOM_NODE
CROSSING_NODES = config.CROSSING_NODES
CROSSING_METRIC = config.CROSSING_METRIC
CROSSING_THRESHOLD = config.CROSSING_THRESHOLD
BODY_NODES = config.BODY_NODES
SIZE_STAT = config.SIZE_STAT
MAX_CROP_AREA = config.MAX_CROP_AREA


# ---------------------------------------------------------------------------
# Pass 1: geometry, from keypoints only (the video is never opened)
# ---------------------------------------------------------------------------


def node_index(skeleton, name: str) -> int:
    names = [n.name for n in skeleton.nodes]
    if name not in names:
        raise KeyError(f"node {name!r} not in skeleton; have {names}")
    return names.index(name)


def rotation_matrix(pts: np.ndarray, i_top: int, i_cen: int) -> np.ndarray | None:
    """2x3 affine rotating `pts` so the top node sits above the centroid, or None."""
    top, cen = pts[i_top], pts[i_cen]
    if not (np.isfinite(top).all() and np.isfinite(cen).all()):
        return None
    return cv2.getRotationMatrix2D(
        (float(cen[0]), float(cen[1])), heading_angle_deg(top, cen), 1.0
    )


def apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (M @ np.c_[pts, np.ones(len(pts))].T).T


def collect_geometry(labels) -> dict:
    """Population statistics over every instance in the movie.

    Reads only the .slp. Returns the stats plus the per-frame keypoint arrays,
    so pass 2 does not have to re-parse them.
    """
    skel = labels[0].instances[0].skeleton
    i_top, i_cen = node_index(skel, TOP_NODE), node_index(skel, CENTROID_NODE)
    i_bot = node_index(skel, BOTTOM_NODE)

    x_spread, y_extent, d_bottom = [], [], []
    per_frame = []  # (frame_idx, [(pts_raw, M), ...])
    n_skipped = 0

    for lf in labels.labeled_frames:
        entries = []
        for inst in lf.instances:
            pts = np.asarray(inst.numpy(), dtype=float)
            M = rotation_matrix(pts, i_top, i_cen)
            if M is None:
                n_skipped += 1
                entries.append(None)  # keep positional alignment with lf.instances
                continue
            entries.append((pts, M))

            ok = np.isfinite(pts).all(axis=1)
            rel = apply(M, pts[ok]) - pts[i_cen]
            x_spread.append(rel[:, 0].max() - rel[:, 0].min())
            y_extent.append(rel[:, 1].max() - rel[:, 1].min())
            if np.isfinite(pts[i_bot]).all():
                b = (M @ np.r_[pts[i_bot], 1.0]) - pts[i_cen]
                d_bottom.append(b[1])  # +ve = below the centroid
        per_frame.append((lf.frame_idx, entries))

    x_spread = np.asarray(x_spread)
    y_extent = np.asarray(y_extent)
    d_bottom = np.asarray(d_bottom)

    stat = np.median if SIZE_STAT == "median" else np.mean
    W_raw = int(round(float(stat(x_spread))))
    H_raw = int(round(float(stat(y_extent))))
    up_from_bottom = float(np.median(d_bottom))
    cy_raw = int(round(H_raw - up_from_bottom))  # centroid's distance from the TOP edge

    # Clamp the crop AREA, scaling both dimensions by the same factor so the
    # animals keep their shape. See config.MAX_CROP_AREA.
    #
    # Round first, since it tracks the true aspect ratio more closely than
    # truncation; but rounding UP can push the product back over the bound, so
    # fall back to floor in that case. floor(W*s) * floor(H*s) <= W*H*s*s ==
    # MAX_CROP_AREA exactly, so the second attempt always satisfies it.
    scale = 1.0
    W, H, cy = W_raw, H_raw, cy_raw
    if MAX_CROP_AREA is not None and W_raw * H_raw > MAX_CROP_AREA:
        scale = math.sqrt(MAX_CROP_AREA / float(W_raw * H_raw))
        W, H = int(round(W_raw * scale)), int(round(H_raw * scale))
        if W * H > MAX_CROP_AREA:
            W, H = int(W_raw * scale), int(H_raw * scale)
        W, H = max(1, W), max(1, H)
        cy = int(round(cy_raw * scale))

    return dict(
        W=W, H=H, cy=cy, up_from_bottom=up_from_bottom,
        # Pre-clamp values. W_raw/H_raw are in VIDEO pixels, so H_raw is the
        # animal's real length on screen -- which is what the centroid
        # similarity's scale needs, and is no longer the same as H once the
        # clamp engages. `scale` is what canonical_crop folds into its affine.
        W_raw=W_raw, H_raw=H_raw, cy_raw=cy_raw, scale=scale,
        i_top=i_top, i_cen=i_cen, i_bot=i_bot,
        per_frame=per_frame, n_skipped=n_skipped,
        x_spread=x_spread, y_extent=y_extent, d_bottom=d_bottom,
    )


# ---------------------------------------------------------------------------
# Crossings: geometric hull overlap within a frame
# ---------------------------------------------------------------------------


def frame_crossings(
    entries,
    node_idx=None,
    threshold: float = CROSSING_THRESHOLD,
    metric: str = CROSSING_METRIC,
) -> tuple[np.ndarray, list[float]]:
    """Flag instances whose hull overlaps another's by more than `threshold`.

    `node_idx` selects which keypoints form the hull (None = all of them).
    Returns (flags, overlap_values) where overlap_values holds the metric for
    every candidate pair, for reporting.

    Bounding boxes are tested first because they are vectorised and reject the
    overwhelming majority of pairs; the hull test only runs on survivors. Hulls
    rather than boxes because a diagonal fly's box is mostly empty -- box
    overlap alone would report contact that never happens.
    """
    n = len(entries)
    out = np.zeros(n, dtype=bool)
    values: list[float] = []

    hulls, areas, boxes, valid = [], [], [], []
    for e in entries:
        pts = None if e is None else e[0]
        if pts is not None:
            if node_idx is not None:
                pts = pts[node_idx]
            pts = pts[np.isfinite(pts).all(axis=1)]
        if pts is None or len(pts) < 3:
            hulls.append(None)
            areas.append(0.0)
            boxes.append((np.nan,) * 4)
            valid.append(False)
            continue
        h = cv2.convexHull(pts.astype(np.float32))
        hulls.append(h)
        areas.append(float(cv2.contourArea(h)))
        boxes.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
        valid.append(True)

    b = np.asarray(boxes, dtype=float)
    valid = np.asarray(valid)
    if valid.sum() < 2:
        return out, values

    # Vectorised pairwise box overlap.
    overlap = (
        (b[:, None, 0] < b[None, :, 2])
        & (b[:, None, 2] > b[None, :, 0])
        & (b[:, None, 1] < b[None, :, 3])
        & (b[:, None, 3] > b[None, :, 1])
    )
    np.fill_diagonal(overlap, False)
    overlap &= valid[:, None] & valid[None, :]

    for i, j in zip(*np.triu_indices(n, k=1)):
        if not overlap[i, j]:
            continue
        inter, _ = cv2.intersectConvexConvex(hulls[i], hulls[j])
        if inter <= 0:
            continue
        if metric == "iou":
            denom = areas[i] + areas[j] - inter
        elif metric == "iomin":
            denom = min(areas[i], areas[j])
        else:
            raise ValueError(f"unknown CROSSING_METRIC {metric!r}")
        value = inter / denom if denom > 0 else 0.0
        values.append(value)
        if value > threshold:
            out[i] = out[j] = True
    return out, values


# ---------------------------------------------------------------------------
# Pass 2: render and write
# ---------------------------------------------------------------------------


def canonical_crop(image, pts, M, geo) -> np.ndarray:
    """One W x H hull-masked crop, via a single small warp.

    Rather than rotating the whole 1536x1536 frame and then cutting a window
    (what canonical_hulls.py does, 32000 times over), the rotation and the
    translation-to-window are composed into one matrix and handed to warpAffine
    with dsize=(W, H). Identical output, ~1000x less pixel work.
    """
    W, H = geo["W"], geo["H"]
    cen = pts[geo["i_cen"]]

    # Built in UNSCALED coordinates first, then the whole affine is multiplied
    # by the clamp factor. A maps p -> A @ [p, 1], so s * (A @ [p, 1]) ==
    # (s * A) @ [p, 1]: scaling every entry, translation included, scales the
    # output. The animal therefore lands at (s*W_raw/2, s*cy_raw) == (W/2, cy)
    # up to the rounding in collect_geometry, and warpAffine resamples straight
    # into the reduced frame -- one interpolation, not a warp followed by a
    # resize. When no clamp is needed scale is 1.0 and this is a no-op.
    A = M.copy()
    A[0, 2] += geo["W_raw"] / 2.0 - cen[0]  # centroid -> horizontal centre
    A[1, 2] += geo["cy_raw"] - cen[1]       # centroid -> cy from the top edge
    if geo["scale"] != 1.0:
        A = A * geo["scale"]

    crop = cv2.warpAffine(image, A, (W, H), flags=cv2.INTER_LINEAR)

    finite = pts[np.isfinite(pts).all(axis=1)]
    hull = cv2.convexHull(apply(A, finite).astype(np.int32))
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return cv2.bitwise_and(crop, crop, mask=mask)


def main(slp_path: Path = SLP, out_dir: Path = OUT_DIR) -> None:
    t0 = time.time()
    print(f"loading {slp_path.name}")
    labels = sio.load_slp(str(slp_path))
    episodes, n_frames, video_path, _ = episodes_from_slp(slp_path)

    print(f"  {len(labels.labeled_frames)} frames, "
          f"{sum(len(lf.instances) for lf in labels.labeled_frames)} instances, "
          f"{len(episodes)} episodes")

    print("pass 1: geometry from keypoints (video not opened)")
    geo = collect_geometry(labels)
    W, H, cy = geo["W"], geo["H"], geo["cy"]
    print(f"  x spread : median {np.median(geo['x_spread']):.1f}  "
          f"mean {geo['x_spread'].mean():.1f}")
    print(f"  y extent : median {np.median(geo['y_extent']):.1f}  "
          f"mean {geo['y_extent'].mean():.1f}")
    print(f"  canonical frame: {W} x {H} (W x H), centroid {cy} px from top "
          f"({geo['up_from_bottom']:.1f} px above bottom)")
    if geo["scale"] != 1.0:
        print(f"  AREA CLAMP: {geo['W_raw']}x{geo['H_raw']} = "
              f"{geo['W_raw'] * geo['H_raw']} px > MAX_CROP_AREA={MAX_CROP_AREA}"
              f" -> scaled by {geo['scale']:.4f} to {W}x{H} = {W * H} px")
    else:
        print(f"  area {W * H} px, under MAX_CROP_AREA={MAX_CROP_AREA} "
              "-> no rescaling")
    if geo["n_skipped"]:
        print(f"  WARNING: {geo['n_skipped']} instances lack an alignment keypoint")

    # Map frame -> episode once. Episodes tile [0, n_frames) exactly (asserted
    # in episodes.py), so a boundary search is safe and O(1) per frame.
    starts = np.array([ep.global_start for ep in episodes])
    per_frame = geo["per_frame"]
    frame_to_ep = {fi: int(np.searchsorted(starts, fi, side="right") - 1)
                   for fi, _ in per_frame}

    # `counts` is rows actually written; `n_inst` is every SLEAP instance. They
    # differ by the instances collect_geometry could not align (TOP_NODE or
    # CENTROID_NODE was NaN). Those produce no row at all, so the difference is
    # recorded per episode as an attribute rather than left to a printed
    # warning -- a silently absent row is otherwise indistinguishable from an
    # instance SLEAP never predicted.
    counts = np.zeros(len(episodes), dtype=int)
    n_inst = np.zeros(len(episodes), dtype=int)
    for fi, entries in per_frame:
        ep_i = frame_to_ep[fi]
        counts[ep_i] += sum(x is not None for x in entries)
        n_inst[ep_i] += len(entries)
    skipped = n_inst - counts

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("id_images_*.h5"):
        f.unlink()

    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])

    print(f"pass 2: rendering {counts.sum()} images -> {out_dir}")
    files, dsets, cursors = [], [], np.zeros(len(episodes), dtype=int)
    for e, ep in enumerate(episodes):
        fh = h5py.File(out_dir / f"id_images_{e}.h5", "w")
        n = int(counts[e])
        d = dict(
            id_images=fh.create_dataset("id_images", (n, H, W), np.uint8),
            crossings=fh.create_dataset("crossings", (n,), bool),
            # Zero-filled, not absent: 0 is upstream's "unassigned" sentinel
            # (list_of_fragments.py:372) and require_dataset there needs a
            # matching shape. No identity information is written here.
            identities=fh.create_dataset("identities", (n,), np.int64),
            frame_numbers=fh.create_dataset("frame_numbers", (n,), np.int64),
            global_index=fh.create_dataset("global_index", (n,), np.int64),
            local_index=fh.create_dataset("local_index", (n,), np.int64),
            instance_idx=fh.create_dataset("instance_idx", (n,), np.int64),
        )
        fh.attrs.update(
            episode=e, global_start=ep.global_start, global_end=ep.global_end,
            index_offset=int(offsets[e]), width=W, height=H, centroid_y=cy,
            # width/height are the crop's own dimensions. body_length is the
            # animal's length in VIDEO pixels, which stops being equal to
            # `height` as soon as the area clamp engages -- anything needing a
            # real-world scale (e.g. the centroid similarity) must read this.
            body_length=int(geo["H_raw"]), crop_width_raw=int(geo["W_raw"]),
            crop_scale=float(geo["scale"]), max_crop_area=(
                -1 if MAX_CROP_AREA is None else int(MAX_CROP_AREA)),
            top_node=TOP_NODE, centroid_node=CENTROID_NODE, bottom_node=BOTTOM_NODE,
            crossing_nodes=CROSSING_NODES, crossing_metric=CROSSING_METRIC,
            crossing_threshold=float(CROSSING_THRESHOLD),
            video_path=str(video_path), slp_path=str(slp_path),
            # Instances in this episode that SLEAP predicted but that have no
            # row here, because they had no alignment keypoint. n_instances is
            # the SLEAP total, so n_instances - n = n_skipped by construction.
            n_skipped=int(skipped[e]), n_instances=int(n_inst[e]),
        )
        files.append(fh)
        dsets.append(d)

    if CROSSING_NODES == "body":
        crossing_nodes = [node_index(labels[0].instances[0].skeleton, n) for n in BODY_NODES]
    elif CROSSING_NODES == "all":
        crossing_nodes = None
    else:
        raise ValueError(f"unknown CROSSING_NODES {CROSSING_NODES!r}")
    print(f"crossings: {CROSSING_NODES}-hull {CROSSING_METRIC} > {CROSSING_THRESHOLD}")

    n_crossing = 0
    all_overlaps: list[float] = []
    for k, (frame_idx, entries) in enumerate(per_frame):
        lf = labels.labeled_frames[k]
        assert lf.frame_idx == frame_idx
        image = lf.image
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        crossing, overlaps = frame_crossings(entries, crossing_nodes)
        all_overlaps.extend(overlaps)
        e = frame_to_ep[frame_idx]
        d, off = dsets[e], int(offsets[e])

        for i, entry in enumerate(entries):
            if entry is None:
                # No usable alignment keypoint (see collect_geometry), so this
                # instance gets NO row. `i` still advances -- that is the whole
                # point of the None placeholder. instance_idx below stays the
                # position in lf.instances, so every surviving instance keeps
                # the index SLEAP gave it and the column is simply sparse in i
                # rather than renumbered. Downstream stages index by
                # instance_idx (build_overlaps) or by the dense global_index
                # (build_fragments); neither assumes i is contiguous.
                continue
            pts, M = entry
            row = int(cursors[e])
            d["id_images"][row] = canonical_crop(image, pts, M, geo)
            d["crossings"][row] = bool(crossing[i])
            d["frame_numbers"][row] = frame_idx
            d["global_index"][row] = off + row
            d["local_index"][row] = row
            d["instance_idx"][row] = i
            cursors[e] += 1
            n_crossing += bool(crossing[i])

        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(per_frame)} frames  ({time.time() - t0:.0f}s)")

    for fh in files:
        fh.close()

    assert (cursors == counts).all(), "row counts do not match the pre-allocation"

    total = int(counts.sum())
    print(f"\nwrote {len(episodes)} files, {total} images ({W}x{H}) to {out_dir}")
    print(f"crossings: {n_crossing}/{total} ({100 * n_crossing / total:.1f}%) "
          f"at {CROSSING_NODES}-hull {CROSSING_METRIC} > {CROSSING_THRESHOLD}")
    if all_overlaps:
        ov = np.asarray(all_overlaps)
        print(f"  overlapping pairs: {len(ov)}, max {ov.max():.4f}, "
              f"p99 {np.percentile(ov, 99):.4f}, median {np.median(ov):.4f}")
        print(f"  headroom: threshold is {CROSSING_THRESHOLD / max(ov.max(), 1e-9):.1f}x "
              "the largest overlap seen")
    else:
        print("  no pairs overlap at all under this node set")
    print(f"episode rows: {counts.tolist()}")
    print(f"index offsets: {offsets.tolist()}")
    if skipped.any():
        print(f"skipped per episode: {skipped.tolist()} "
              f"({int(skipped.sum())}/{int(n_inst.sum())} instances have no row; "
              "surviving rows keep their SLEAP instance_idx)")
    print(f"elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
