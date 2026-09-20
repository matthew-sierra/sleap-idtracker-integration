"""Build idtracker.ai id-image HDF5 files from a SLEAP predictions file.

This is the hand-off point between the SLEAP half of the port and the
idtracker.ai half: everything upstream of here is keypoints, everything
downstream (fragmentation, contrastive training, assignment) reads these files.

Output layout mirrors idtracker.ai's own (``session.py:716``): one file per
episode, ``id_images/id_images_{episode}.h5``, each holding

    id_images     (n, HEIGHT, WIDTH) uint8   the canonical hull crops
                  (n, HEIGHT, WIDTH, 3)      when config.COLOR_MODE == "RGB"
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

3. COLOUR IS DECLARED, NOT DETECTED. config.COLOR_MODE says whether the video
   is single-channel or three-channel, and every frame is checked against it
   (rotate_boxes.check_frame). Nothing converts between the two. The crop
   geometry is identical in both modes -- same HEIGHT, same WIDTH, same rows,
   same affine -- so an RGB session differs only in that each pixel occupies
   three uint8 slots instead of one, making its files 3x the bytes at the same
   image dimensions. A mismatch aborts the session and removes whatever was
   written, because a partly-filled file reads as a complete one downstream.

4. NO CROSSING DETECTION HERE. Upstream trains a CNN to classify blobs as
   individual-vs-crossing (``crossing_detector.py:81``) because thresholded
   blobs merge when animals touch. SLEAP already separates overlapping animals
   into distinct instances, so the classifier is not needed and is skipped
   entirely. The load-bearing determination is ``is_overlapping``, computed from
   the assignment cost matrix in build_overlaps.py -- not from geometry, and not
   in this stage. A within-frame hull-overlap flag used to be written here as a
   `crossings` column; it was read by nothing and has been removed.

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
from rotate_boxes import (  # noqa: E402
    CENTROID_NODE, TOP_NODE, TOP_NODE_FALLBACK, ColorModeError, check_frame,
    heading_angle_deg,
)

import config  # noqa: E402

REPO = config.REPO
SLP = config.SLP
OUT_DIR = config.OUT_DIR

# All set in config.py.
BOTTOM_NODE = config.BOTTOM_NODE
SIZE_STAT = config.SIZE_STAT
MAX_CROP_AREA = config.MAX_CROP_AREA
COLOR_MODE = config.COLOR_MODE
N_CHANNELS = config.N_CHANNELS
N_ANIMALS = config.N_ANIMALS
SURROUNDING_KEYPOINTS = config.SURROUNDING_KEYPOINTS
BOUNDED_KEYPOINTS = config.BOUNDED_KEYPOINTS


# ---------------------------------------------------------------------------
# Pass 1: geometry, from keypoints only (the video is never opened)
# ---------------------------------------------------------------------------


class PreflightError(ValueError):
    """An input precondition failed. Raised before any work is done."""


def preflight(labels, n_animals: int = None, min_keypoints: int = None,
              max_listed: int = 20) -> None:
    """Validate the .slp against the config, before a single frame is read.

    Two preconditions, both cheap enough to pay unconditionally and both fatal
    if violated -- the point is to fail in milliseconds with a message naming
    the problem, rather than deep inside pass 2 with a shape error.

    1. NO FRAME MAY HOLD MORE THAN N_ANIMALS INSTANCES. More detections than
       animals means SLEAP over-detected, and the surplus propagates: a phantom
       can win an identity and collide with a real animal, blanking BOTH for
       that frame (build_sleap_tracks leaves collisions untracked).

    2. AT LEAST ONE INSTANCE MUST CARRY >= MIN_HULL_KEYPOINTS FINITE KEYPOINTS.
       If not one instance in the whole video can form a hull, every crop would
       be skipped and the session would produce an empty HDF5. Individual
       thin instances are still skipped one by one (InsufficientKeypointsError);
       this catches only the case where there is nothing to work with at all.

    Frame numbers in messages are 1-BASED -- the first frame of the video is
    frame 1 -- because that is how they are read off a video player. Everything
    internal stays on sleap-io's 0-based ``frame_idx``.
    """
    n_animals = N_ANIMALS if n_animals is None else n_animals
    min_keypoints = MIN_HULL_KEYPOINTS if min_keypoints is None else min_keypoints
    frames = labels.labeled_frames

    # --- 1. instance count per frame ---------------------------------------
    # One len() per frame; no keypoints touched, no video opened. Vectorised
    # over the counts rather than branching per frame so the whole check is a
    # single pass plus one numpy comparison.
    counts = np.fromiter((len(lf.instances) for lf in frames), dtype=np.int64,
                         count=len(frames))
    over = np.flatnonzero(counts > n_animals)
    if over.size:
        # +1: report the first frame of the video as frame 1.
        bad = [int(frames[i].frame_idx) + 1 for i in over]
        shown = ", ".join(map(str, bad[:max_listed]))
        if len(bad) > max_listed:
            shown += f", ... ({len(bad) - max_listed} more)"
        worst = int(counts[over].max())
        raise PreflightError(
            f"{len(bad)} frame(s) hold more instances than N_ANIMALS={n_animals} "
            f"(worst: {worst} instances). Frames: {shown}.\n"
            "  SLEAP detected more animals than the video contains. Raise "
            "N_ANIMALS if the count is wrong, or clean the extra instances out "
            "of the .slp -- a surplus detection can take an identity from a "
            "real animal and leave both untracked."
        )

    # --- 2. is ANY instance usable? ----------------------------------------
    # Early exit on the first instance that qualifies, so the common case costs
    # one instance rather than a sweep. points["xy"]/["visible"] are read
    # directly instead of inst.numpy(), which copies and re-derives the NaNs.
    for lf in frames:
        for inst in lf.instances:
            pts = inst.points
            usable = np.isfinite(pts["xy"]).all(axis=1) & pts["visible"]
            if int(usable.sum()) >= min_keypoints:
                return
    raise PreflightError(
        f"not enough keypoints: no instance in this file has {min_keypoints} or "
        f"more finite keypoints. Each instance needs at least {min_keypoints} to "
        "generate a crop, because a convex hull needs three corners.\n"
        "  Nothing can be built from this .slp -- check that the predictions "
        "carry keypoint coordinates and that the skeleton matches the video."
    )


def node_index(skeleton, name: str) -> int:
    names = [n.name for n in skeleton.nodes]
    if name not in names:
        raise KeyError(f"node {name!r} not in skeleton; have {names}")
    return names.index(name)


def rotation_matrix(pts: np.ndarray, i_top: int, i_cen: int,
                    i_top_fb: int | None = None) -> tuple[np.ndarray | None, bool]:
    """2x3 affine rotating `pts` so the top node sits above the centroid.

    Returns ``(M, used_fallback)``. ``M`` is None when the centroid is NaN, or
    when the top node is NaN and either no fallback is configured
    (``i_top_fb is None``) or the fallback is NaN too -- i.e. the instance
    cannot be aligned and gets no row, which is the behaviour this function had
    before the fallback existed.

    ``used_fallback`` is True only on the rows that would have been dropped
    under the old rule and are now kept, so summing it counts exactly what the
    fallback recovered.
    """
    cen = pts[i_cen]
    if not np.isfinite(cen).all():
        return None, False

    top = pts[i_top]
    used_fallback = False
    if not np.isfinite(top).all():
        if i_top_fb is None:
            return None, False
        top = pts[i_top_fb]
        if not np.isfinite(top).all():
            return None, False
        used_fallback = True

    return cv2.getRotationMatrix2D(
        (float(cen[0]), float(cen[1])), heading_angle_deg(top, cen), 1.0
    ), used_fallback


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
    # None unless config.TOP_NODE_FALLBACK names a node, in which case
    # rotation_matrix reaches for it whenever TOP_NODE is NaN.
    i_top_fb = None if TOP_NODE_FALLBACK is None else node_index(skel, TOP_NODE_FALLBACK)
    if i_top_fb == i_top:
        raise ValueError(
            f"TOP_NODE_FALLBACK ({TOP_NODE_FALLBACK!r}) is the same node as "
            f"TOP_NODE ({TOP_NODE!r}); a fallback to itself can never help")
    # Silhouette roles, resolved here for the same reason as the alignment
    # nodes: once, against one skeleton, so the render loop indexes by integer.
    i_surround = [node_index(skel, n) for n in SURROUNDING_KEYPOINTS]
    i_bounded = [node_index(skel, n) for n in BOUNDED_KEYPOINTS]
    overlap = set(SURROUNDING_KEYPOINTS) & set(BOUNDED_KEYPOINTS)
    if overlap:
        raise ValueError(
            f"a keypoint cannot be both surrounding and bounded: {sorted(overlap)}")

    x_spread, y_extent, d_bottom = [], [], []
    per_frame = []  # (frame_idx, [(pts_raw, M), ...])
    # Fallback uses, counted per LabeledFrame so main() can aggregate them per
    # episode for the HDF5 attribute. Parallel to per_frame by position.
    fb_per_lf = []
    n_top_fallback = 0
    n_skipped = 0
    n_degenerate = 0
    n_insufficient = 0

    for lf in labels.labeled_frames:
        entries = []
        fb_here = 0
        for inst in lf.instances:
            pts = np.asarray(inst.numpy(), dtype=float)
            M, used_fb = rotation_matrix(pts, i_top, i_cen, i_top_fb)
            if M is None:
                n_skipped += 1
                entries.append(None)  # keep positional alignment with lf.instances
                continue

            # Can this instance produce a crop at all? Asked HERE, in pass 1,
            # because pass 2 writes into datasets pre-allocated from these
            # counts -- discovering it later would leave a row's worth of hole.
            # Both checks are affine-invariant, so testing the raw keypoints
            # answers it for the rotated ones too.
            #
            # hull_points raises InsufficientKeypointsError on the count;
            # graham_scan raises DegenerateHullError on the geometry. Counted
            # separately because they mean different things to a user: "SLEAP
            # gave me too little" versus "SLEAP gave me a straight line".
            try:
                graham_scan(hull_points(pts, i_surround, i_bounded)[0])
            except InsufficientKeypointsError:
                n_insufficient += 1
                entries.append(None)  # same treatment as a missing alignment node
                continue
            except DegenerateHullError:
                n_degenerate += 1
                entries.append(None)
                continue

            entries.append((pts, M))
            # Counted HERE, after the hull checks, so the number means "rows in
            # the HDF5 that exist because of the fallback" rather than "times
            # the fallback was reached for".
            if used_fb:
                fb_here += 1
                n_top_fallback += 1

            ok = np.isfinite(pts).all(axis=1)
            rel = apply(M, pts[ok]) - pts[i_cen]
            x_spread.append(rel[:, 0].max() - rel[:, 0].min())
            y_extent.append(rel[:, 1].max() - rel[:, 1].min())
            if np.isfinite(pts[i_bot]).all():
                b = (M @ np.r_[pts[i_bot], 1.0]) - pts[i_cen]
                d_bottom.append(b[1])  # +ve = below the centroid
        per_frame.append((lf.frame_idx, entries))
        fb_per_lf.append(fb_here)

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
        i_top=i_top, i_cen=i_cen, i_bot=i_bot, i_top_fb=i_top_fb,
        i_surround=i_surround, i_bounded=i_bounded,
        per_frame=per_frame, n_skipped=n_skipped, n_degenerate=n_degenerate,
        n_insufficient=n_insufficient,
        fb_per_lf=fb_per_lf, n_top_fallback=n_top_fallback,
        x_spread=x_spread, y_extent=y_extent, d_bottom=d_bottom,
    )


# ---------------------------------------------------------------------------
# Crossings: geometric hull overlap within a frame
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pass 2: render and write
# ---------------------------------------------------------------------------


# A convex polygon needs three corners. Everything below is that one fact,
# enforced at the two places it can fail.
MIN_HULL_KEYPOINTS = 3


class CropGeometryError(ValueError):
    """This instance cannot produce an id-image. Base of the two causes below.

    Both are SKIPS, never aborts: build_id_images drops the instance, writes no
    HDF5 row for it, and carries on. One unusable animal in one frame must not
    cost the run the other several thousand.
    """


class InsufficientKeypointsError(CropGeometryError):
    """Fewer than MIN_HULL_KEYPOINTS finite keypoints to build a crop from.

    Raised BEFORE any hull is attempted -- there is nothing for Graham scan or
    cv2.convexHull to consume, so the question is answered by counting rather
    than by geometry.
    """


class DegenerateHullError(CropGeometryError):
    """Enough points, but no polygon -- they are collinear or coincident.

    Raised by graham_scan rather than returned as a sentinel so it cannot be
    ignored by accident. build_id_images catches it and SKIPS the instance
    (no HDF5 row, exactly like a missing alignment keypoint); it is never
    allowed to abort a run, because one unusable animal in one frame is not a
    reason to lose the other several thousand.
    """


def graham_scan(points: np.ndarray) -> np.ndarray:
    """Convex hull of `points` by Graham scan. Returns the hull vertices, (k, 2).

    Used for the FALLBACK hull only -- the one built when a surrounding keypoint
    is missing and the silhouette has to be rebuilt from whatever nodes survive.
    The primary path keeps cv2.convexHull, which is what it always used.

    The scan proper: take the lowest point as pivot (lowest y, then lowest x --
    guaranteed to be a hull vertex), sort the rest by the angle the pivot sees
    them at, then sweep that order maintaining a stack, popping any vertex that
    a new point proves to be a right turn. What survives is the hull, CCW in
    standard axes. Points are deduplicated first because repeated coordinates
    make the angular sort ambiguous, and ties in angle are broken by distance so
    that collinear runs are entered nearest-first and collapse cleanly.

    Note these are IMAGE coordinates (+y down), so the CCW winding looks
    clockwise on screen. Nothing downstream cares -- cv2.fillConvexPoly accepts
    either -- but the cross-product sign convention below is the standard one.
    """
    P = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(P) < 3:
        raise DegenerateHullError(
            f"convex hull needs 3 distinct points, got {len(P)}")

    pivot = P[np.lexsort((P[:, 0], P[:, 1]))[0]]
    rest = P[~np.all(P == pivot, axis=1)]
    if len(rest) < 2:
        raise DegenerateHullError(
            f"convex hull needs 3 distinct points, got {len(P)}")

    d = rest - pivot
    order = np.lexsort((np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0])))
    ordered = rest[order]

    def cross(o, a, b) -> float:
        """> 0 left turn, < 0 right turn, == 0 collinear."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    stack = [pivot, ordered[0]]
    for pt in ordered[1:]:
        # <= 0 rather than < 0 so collinear points are dropped, leaving only
        # true corners. A hull with interior-edge points would still fill
        # correctly but carries vertices that mean nothing.
        while len(stack) > 1 and cross(stack[-2], stack[-1], pt) <= 0:
            stack.pop()
        stack.append(pt)
    hull = np.asarray(stack)
    if len(hull) < 3:
        # Every point collinear: the "hull" is a line segment, which
        # fillConvexPoly would render as a 1px streak rather than a silhouette.
        raise DegenerateHullError(
            f"all {len(P)} points are collinear; no polygon exists")
    return hull


def hull_points(pts: np.ndarray, i_surround, i_bounded) -> tuple[np.ndarray, bool]:
    """The keypoints forming one instance's silhouette, and whether we fell back.

    Primary: the SURROUNDING_KEYPOINTS alone, when every one of them is present.
    Fallback: every available keypoint, surrounding and bounded alike, when any
    surrounding one is missing -- the outline has a gap, so the hull is rebuilt
    from all surviving evidence rather than from a set known to be incomplete.
    """
    surround = pts[i_surround]
    if np.isfinite(surround).all():
        src, fallback = surround, False
    else:
        both = pts[list(i_surround) + list(i_bounded)]
        src, fallback = both[np.isfinite(both).all(axis=1)], True

    # The count gate. Checked here, before graham_scan or cv2.convexHull sees
    # anything, because with under three points there is no hull to attempt --
    # the failure is missing information, not bad geometry, and it deserves to
    # say so rather than surface as a confusing degeneracy error.
    if len(src) < MIN_HULL_KEYPOINTS:
        raise InsufficientKeypointsError(
            f"not enough keypoint information to generate a crop: "
            f"{len(src)} finite keypoint{'' if len(src) == 1 else 's'} available "
            f"out of {len(i_surround) + len(i_bounded)} configured, but at least "
            f"{MIN_HULL_KEYPOINTS} are needed to form a convex hull"
        )
    return src, fallback


def canonical_crop(image, pts, M, geo) -> tuple[np.ndarray, bool]:
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

    # Which nodes outline this animal, and by which route (see hull_points).
    src, fallback = hull_points(pts, geo["i_surround"], geo["i_bounded"])
    xy = apply(A, src)
    hull = (graham_scan(xy) if fallback
            else cv2.convexHull(xy.astype(np.int32))).astype(np.int32)
    if len(hull) < 3:
        # Only reachable on the cv2 path, and only when the int32 rounding that
        # the area clamp's downscale forces collapses otherwise-distinct
        # keypoints onto one pixel. graham_scan raises this itself.
        raise DegenerateHullError(
            f"hull has {len(hull)} vertices after rounding; no polygon to fill")

    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull.reshape(-1, 1, 2), 255)
    return cv2.bitwise_and(crop, crop, mask=mask), fallback


def main(slp_path: Path = SLP, out_dir: Path = OUT_DIR) -> None:
    t0 = time.time()
    print(f"loading {slp_path.name}")
    labels = sio.load_slp(str(slp_path))
    episodes, n_frames, video_path, _ = episodes_from_slp(slp_path)

    print(f"  {len(labels.labeled_frames)} frames, "
          f"{sum(len(lf.instances) for lf in labels.labeled_frames)} instances, "
          f"{len(episodes)} episodes")

    # Preconditions first: cheapest possible failure, before the video is
    # opened or a single crop is built. Raises PreflightError on violation.
    preflight(labels)
    print(f"  preflight: <= N_ANIMALS={N_ANIMALS} per frame, "
          f"keypoints sufficient")

    # COLOR_MODE is checked against frame 0 here, before pass 1 and before a
    # single HDF5 file exists, so the overwhelmingly common case -- the mode is
    # simply set wrong for this video -- costs one frame decode and leaves
    # nothing behind to clean up. The per-frame check in pass 2 stays anyway:
    # this one cannot see a video whose channel count changes partway through.
    probe = labels.labeled_frames[0]
    check_frame(probe.image, probe.frame_idx, Path(video_path).name)
    print(f"  colour: COLOR_MODE={COLOR_MODE} ({N_CHANNELS} channel"
          f"{'' if N_CHANNELS == 1 else 's'}), frame 0 agrees")

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
    if TOP_NODE_FALLBACK is not None:
        print(f"  top-node fallback: {geo['n_top_fallback']} instance(s) aligned on "
              f"{TOP_NODE_FALLBACK!r} because {TOP_NODE!r} was NaN "
              "-- these would have had no row at all before")
    if geo["n_skipped"]:
        print(f"  WARNING: {geo['n_skipped']} instances lack an alignment keypoint"
              + (f" ({TOP_NODE!r} AND {TOP_NODE_FALLBACK!r} both NaN, or "
                 f"{CENTROID_NODE!r} NaN)" if TOP_NODE_FALLBACK is not None else ""))
    if geo["n_insufficient"]:
        print(f"  WARNING: {geo['n_insufficient']} instances have fewer than "
              f"{MIN_HULL_KEYPOINTS} finite keypoints -- not enough information to "
              "generate a crop; skipped, no row (InsufficientKeypointsError)")
    if geo["n_degenerate"]:
        print(f"  WARNING: {geo['n_degenerate']} instances have enough keypoints "
              "but they are collinear -- no polygon; skipped, no row "
              "(DegenerateHullError)")

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
    # Rows in this episode that exist only because TOP_NODE_FALLBACK was used.
    # Aggregated per episode rather than reported globally-or-not-at-all (the
    # -1 convention used by n_no_alignment below), because a reader holding one
    # episode file should be able to say how many of ITS rows are fallback rows.
    fb_counts = np.zeros(len(episodes), dtype=int)
    fb_per_lf = geo["fb_per_lf"]
    for k, (fi, entries) in enumerate(per_frame):
        ep_i = frame_to_ep[fi]
        counts[ep_i] += sum(x is not None for x in entries)
        n_inst[ep_i] += len(entries)
        fb_counts[ep_i] += fb_per_lf[k]
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
        img_shape = (n, H, W) if N_CHANNELS == 1 else (n, H, W, N_CHANNELS)
        d = dict(
            # (n, H, W) in grayscale, (n, H, W, 3) in RGB. The spatial
            # dimensions are identical either way -- COLOR_MODE widens the
            # pixel, it does not change the crop.
            id_images=fh.create_dataset("id_images", img_shape, np.uint8,
                                        maxshape=(None,) + img_shape[1:]),
            # True = the surrounding outline had a gap, so this row's hull
            # was rebuilt by Graham scan over every surviving keypoint.
            hull_fallback=fh.create_dataset("hull_fallback", (n,), bool, maxshape=(None,)),
            # Zero-filled, not absent: 0 is upstream's "unassigned" sentinel
            # (list_of_fragments.py:372) and require_dataset there needs a
            # matching shape. No identity information is written here.
            identities=fh.create_dataset("identities", (n,), np.int64, maxshape=(None,)),
            frame_numbers=fh.create_dataset("frame_numbers", (n,), np.int64, maxshape=(None,)),
            global_index=fh.create_dataset("global_index", (n,), np.int64, maxshape=(None,)),
            local_index=fh.create_dataset("local_index", (n,), np.int64, maxshape=(None,)),
            instance_idx=fh.create_dataset("instance_idx", (n,), np.int64, maxshape=(None,)),
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
            # Stamped so a reader never has to infer the layout from ndim,
            # and so a session built under one mode is identifiable as such.
            color_mode=COLOR_MODE, n_channels=int(N_CHANNELS),
            surrounding_keypoints=list(SURROUNDING_KEYPOINTS),
            bounded_keypoints=list(BOUNDED_KEYPOINTS),
            top_node=TOP_NODE, centroid_node=CENTROID_NODE, bottom_node=BOTTOM_NODE,
            # "" rather than None: h5py has no null attribute, and an empty
            # string reads unambiguously as "no fallback configured".
            top_node_fallback=("" if TOP_NODE_FALLBACK is None
                               else str(TOP_NODE_FALLBACK)),
            # Rows in THIS file aligned on the fallback node, and the session
            # total. Both are always populated, so a fallback run is never
            # mistakable for a run without one.
            n_top_fallback=int(fb_counts[e]),
            n_top_fallback_total=int(geo["n_top_fallback"]),
            video_path=str(video_path), slp_path=str(slp_path),
            # Instances in this episode that SLEAP predicted but that have no
            # row here, because they had no alignment keypoint. n_instances is
            # the SLEAP total, so n_instances - n = n_skipped by construction.
            # n_skipped is the TOTAL absent rows; the two causes are split out
            # so a reader never has to guess which applies.
            n_skipped=int(skipped[e]), n_instances=int(n_inst[e]),
            n_no_alignment=int(geo["n_skipped"]) if len(episodes) == 1 else -1,
            n_degenerate_hull=int(geo["n_degenerate"]) if len(episodes) == 1 else -1,
            n_insufficient_keypoints=(int(geo["n_insufficient"])
                                      if len(episodes) == 1 else -1),
            min_hull_keypoints=MIN_HULL_KEYPOINTS,
        )
        files.append(fh)
        dsets.append(d)

    n_fallback = 0
    n_degenerate_late = 0
    try:
        for k, (frame_idx, entries) in enumerate(per_frame):
            lf = labels.labeled_frames[k]
            assert lf.frame_idx == frame_idx
            # Validates AND canonicalises: (H, W, 1) -> (H, W) under GRAYSCALE,
            # (H, W, 3) passed through under RGB, ColorModeError on a contradiction.
            # Checked on every frame, not just frame 0 -- a video that changes
            # channel count partway through would otherwise write rows of the wrong
            # layout into a dataset already allocated for the other one.
            image = check_frame(lf.image, frame_idx, Path(video_path).name)

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
                try:
                    crop, fell_back = canonical_crop(image, pts, M, geo)
                except CropGeometryError as exc:
                    # Pass 1 already rejected the degenerate cases it can see, so
                    # this only fires when int32 rounding collapses the hull. Skip
                    # the image rather than abort the run; the pre-allocated tail
                    # is trimmed below so no zero row is left behind.
                    n_degenerate_late += 1
                    print(f"  SKIPPED frame {frame_idx} instance {i}: {exc}")
                    continue
                d["id_images"][row] = crop
                d["hull_fallback"][row] = fell_back
                n_fallback += fell_back
                d["frame_numbers"][row] = frame_idx
                d["global_index"][row] = off + row
                d["local_index"][row] = row
                d["instance_idx"][row] = i
                cursors[e] += 1

            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(per_frame)} frames  ({time.time() - t0:.0f}s)")
    except ColorModeError:
        # The files are open and partly written by now. A half-populated
        # id_images_*.h5 is indistinguishable from a complete one downstream --
        # every reader trusts the allocated row count, and the tail would read
        # as legitimate all-zero images -- so the session is torn down rather
        # than left for the next stage to misread.
        for fh in files:
            fh.close()
        for f in out_dir.glob("id_images_*.h5"):
            f.unlink()
        print(f"\nABORTED: removed {len(files)} partly-written file(s) "
              f"from {out_dir}")
        raise

    if n_degenerate_late:
        # Shrink each dataset to the rows actually written. Without this the tail
        # would read as legitimate all-zero images with identity 0 -- the very
        # confusion the sparse-row convention exists to avoid.
        for e, (fh, d) in enumerate(zip(files, dsets)):
            if cursors[e] < counts[e]:
                for name in d:
                    fh[name].resize(int(cursors[e]), axis=0)
                fh.attrs["n_degenerate_skipped"] = int(counts[e] - cursors[e])
        print(f"  trimmed {n_degenerate_late} degenerate row(s) from the "
              "pre-allocation")
        counts = cursors.copy()

    for fh in files:
        fh.close()

    assert (cursors == counts).all(), "row counts do not match the pre-allocation"

    total = int(counts.sum())
    print(f"\nwrote {len(episodes)} files, {total} images ({W}x{H}) to {out_dir}")
    print(f"hulls: {total - n_fallback}/{total} from SURROUNDING_KEYPOINTS "
          f"{SURROUNDING_KEYPOINTS}; {n_fallback} ({100 * n_fallback / total:.2f}%) "
          f"fell back to Graham scan over all available keypoints")
    if TOP_NODE_FALLBACK is not None:
        n_fb = int(geo["n_top_fallback"])
        print(f"top node: {total - n_fb}/{total} rows aligned on {TOP_NODE!r}; "
              f"{n_fb} ({100 * n_fb / total:.2f}%) on the fallback "
              f"{TOP_NODE_FALLBACK!r}")
    print(f"episode rows: {counts.tolist()}")
    print(f"index offsets: {offsets.tolist()}")
    if skipped.any():
        print(f"skipped per episode: {skipped.tolist()} "
              f"({int(skipped.sum())}/{int(n_inst.sum())} instances have no row; "
              "surviving rows keep their SLEAP instance_idx)")
    print(f"elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
