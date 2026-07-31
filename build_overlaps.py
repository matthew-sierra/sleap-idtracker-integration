"""Add the `is_overlapping` flag and the frame-to-frame links to the id-image files.

Stage 3c. Runs *after* ``build_id_images.py`` and edits its output in place:
every file is opened ``r+``, three datasets are added, a few attributes are
recorded, and nothing already present is touched (verified by digest, see
``snapshot()``).

    is_overlapping    (n,)   bool    True  = this instance is confusable
    next_global_index (n,)   int64   global_index of its successor, -1 if none
    centroid          (n, 2) float64 CENTROID_NODE position, video coordinates

WHAT THE FLAG MEANS
-------------------
Upstream idtracker.ai calls this field ``is_an_individual`` and uses it to ask
"does this blob contain exactly one animal?" -- a segmentation question, because
thresholded blobs merge when animals touch (``blob.py``/``crossing_detector.py``).
SLEAP has already answered that: one instance is one animal, always. Ported
literally the flag would be a constant True, which is why it is named for what
it actually measures here instead:

    is_overlapping[r] == True
        -> at least one alternative frame-to-frame assignment costs within
           SIMILARITY_THRESHOLD of the optimal one (measured on the scale set by
           GAP_SCALE), and this instance's partner changes under it. Some other
           animal is nearly as good an explanation for this detection, so
           propagating identity across this link is unsafe.

This is the *negation* of upstream's field: ``is_an_individual == not
is_overlapping``. The inversion is applied once, at the boundary where upstream
``Fragment`` objects are constructed (``build_fragments.py``), so idtracker.ai's
own code keeps its own polarity and needs no edit.

THE LINKS
---------
``next_global_index`` is the identity graph, built in the *same* pass as the
cost matrix rather than in a second sweep: the Hungarian solution sigma that the
ambiguity test already needs is exactly the frame-to-frame correspondence, so it
is returned rather than recomputed.

sigma indexes instances 0..n-1 *within a frame* and carries no global meaning on
its own. It is resolved into a real link through the frame number the pair came
from: instance i of frame t links to instance sigma[i] of frame t+1, and that
(frame, instance) key maps to a unique ``global_index``. Centroids are written
alongside so the pairing can be inspected and replayed in video coordinates.

Per spec, a link is only kept when both endpoints agree on ``is_overlapping`` --
a stretch of clean tracking and a stretch of huddling must not end up in the
same fragment. This mirrors upstream's chain condition in
``fragmentation.py:68``, which likewise refuses to extend across a change in the
flag.

THE ALGORITHM
-------------
For each consecutive frame pair (t, u):

    S[i, j] = similarity between instance i of t and instance j of u
              (SIMILARITY_METHOD: bbox IoU, OKS, or centroid decay)
    C       = 1 - S                          (cost; Hungarian minimises)
    sigma   = linear_sum_assignment(C)       (the committed assignment)
    opt     = C[sigma].sum()

A plain Hungarian solve stops here and reports only the winner. It cannot tell a
decisive assignment from a coin-flip, because it never looks at the runner-up
*permutation*. So for every off-assignment cell (i, j) we solve the assignment
problem again with i->j forced:

    forced(i, j) = C[i, j] + opt(rows != i, cols != j)
    gap(i, j)    = forced(i, j) - opt        (>= 0 by optimality of sigma)

If gap <= SIMILARITY_THRESHOLD the alternative is "nearly as good". Every
instance whose partner differs between sigma and that alternative is flagged --
not just i and j, because the two permutations differ by a cycle and every
animal on that cycle is mutually confusable.

Under GAP_SCALE = "per_edge" the comparison is gap / n_changed rather
than gap, which is a ratio and so is NOT what a Hungarian solve minimises. It is
still answered exactly, in the same one-solve-per-cell budget, by shifting the
cost matrix first; see the comment in ``assignment_ambiguity``.

Doing this naively is n^2 Hungarian solves per frame pair. The prune is exact,
not heuristic: removing a column can only make an assignment problem more
expensive, so

    opt(rows != i, cols != j) >= opt(rows != i, all cols) =: L[i]

giving the valid lower bound ``gap(i, j) >= C[i, j] + L[i] - opt``. L costs n
solves, and any cell whose bound already exceeds the ceiling is skipped without
a solve. On this dataset that removes >99% of the work.

The cost matrix is padded to square with UNMATCHED_COST before solving, so every
row is assigned and the bound above stays valid even when the two frames hold
different numbers of instances (removing a column would otherwise be able to
*lower* the cost by leaving a row unassigned).

SIMILARITY METRIC
-----------------
Selected by ``config.SIMILARITY_METHOD``; all three are ports of the SLEAP
tracker's own scoring, imported from ``assignment_margin.py``:

    "bounding_box"  ``iou_matrix`` / ``poses_to_bboxes``  -- matches
        ``sleap_nn.tracking.utils.compute_iou`` and ``get_bbox`` exactly,
        including the ``+1`` inclusive-pixel convention.
    "keypoint"      ``oks_matrix``                        -- matches
        ``sleap_nn.evaluation.compute_oks``, cocoeval normalization.
    "centroid"      ``centroid_matrix`` / ``poses_to_centroids`` -- the centroid
        is upstream's ``get_centroid`` (nanmedian over OVERLAP_NODES, NOT
        CENTROID_NODE). The SIMILARITY is not upstream's, and cannot be:
        ``compute_euclidean_distance`` returns a raw NEGATIVE distance, which is
        unbounded and would break ``C = 1 - S`` against UNMATCHED_COST padding.
        Mapped instead through ``exp(-d / body_length)``, so the unit is body
        lengths travelled and no pair is ever hard-excluded.

Because all three are SLEAP's, SIMILARITY_THRESHOLD is on the scale the SLEAP
tracker uses -- but the three do NOT share a scale as each other, which is why
``config.SIMILARITY_THRESHOLD`` is a dict keyed by method. The smallest per-edge
gap on this clip is 0.7856 (bbox), 0.5820 (keypoint), 0.4908 (centroid): a
threshold below its method's minimum flags nothing at all.

Run:
    /opt/anaconda3/envs/sleap_id/bin/python src/sleap_idtracker/build_overlaps.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import sleap_io as sio
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from assignment_margin import (  # noqa: E402
    centroid_matrix, iou_matrix, oks_matrix, poses_to_bboxes, poses_to_centroids)
from build_id_images import node_index  # noqa: E402

BODY_NODES = config.BODY_NODES
CENTROID_NODE = config.CENTROID_NODE

REPO = config.REPO
SLP = config.SLP
OUT_DIR = config.OUT_DIR

# The knobs that decide what counts as an overlap. All set in config.py, which
# carries the full explanation of each.
SIMILARITY_METHOD = config.SIMILARITY_METHOD
# config.SIMILARITY_THRESHOLD is a dict keyed by method -- the three do not
# share a scale. Resolved once here so the rest of the module sees a float.
if SIMILARITY_METHOD not in config.SIMILARITY_THRESHOLD:
    raise ValueError(
        f"SIMILARITY_METHOD {SIMILARITY_METHOD!r} has no threshold in "
        f"config.SIMILARITY_THRESHOLD (has {sorted(config.SIMILARITY_THRESHOLD)})")
SIMILARITY_THRESHOLD = float(config.SIMILARITY_THRESHOLD[SIMILARITY_METHOD])
OVERLAP_NODES = config.OVERLAP_NODES
OVERLAP_DIRECTION = config.OVERLAP_DIRECTION
MAX_FRAME_GAP = config.MAX_FRAME_GAP
GAP_SCALE = config.GAP_SCALE
UNMATCHED_COST = config.UNMATCHED_COST
REPORT_CEILING = config.REPORT_CEILING
TIE_TOL = config.TIE_TOL
SWEEP = config.SWEEP


# ---------------------------------------------------------------------------
# The ambiguity test
# ---------------------------------------------------------------------------


def assignment_ambiguity(
    C: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
    ceiling: float = REPORT_CEILING,
    normalization: str = GAP_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flag rows/cols involved in a near-optimal alternative assignment.

    `C` is a (n, m) cost matrix, finite and >= 0. Returns
    ``(row_flags, col_flags, row_gap, sigma)`` where `row_gap[i]` is the exact
    cost increase of the cheapest alternative that re-partners row i, or `inf`
    if that increase provably exceeds `ceiling`, and `sigma[i]` is the column
    row i is assigned to (>= m meaning "matched only to padding", i.e. no
    partner exists).

    `sigma` is returned rather than left internal because it IS the
    frame-to-frame correspondence the caller needs for the identity graph.
    Recomputing it outside would risk the two disagreeing on a padded or
    degenerate matrix.
    """
    if normalization not in ("per_edge", "total"):
        raise ValueError(f"unknown GAP_SCALE {normalization!r}")
    n, m = C.shape
    k = max(n, m)
    P = np.full((k, k), UNMATCHED_COST, dtype=float)
    P[:n, :m] = C

    idx = np.arange(k)
    r, c = linear_sum_assignment(P)
    sigma = np.empty(k, dtype=int)
    sigma[r] = c
    opt = float(P[r, c].sum())

    # Exact lower bound (see module docstring): dropping row i and any column
    # costs at least as much as dropping row i alone.
    L = np.empty(k)
    for i in range(k):
        rows = np.delete(idx, i)
        sub = P[rows]
        rr, cc = linear_sum_assignment(sub)
        L[i] = sub[rr, cc].sum()

    row_flag = np.zeros(k, dtype=bool)
    col_flag = np.zeros(k, dtype=bool)
    row_gap = np.full(k, np.inf)

    # Which cost matrix to search, and what "near-optimal" means in it.
    #
    # "total" searches P directly and flags when cost(alt) - opt <= threshold.
    #
    # "per_edge" cannot do that, because minimising total cost through a forced
    # cell does NOT minimise cost-per-changed-edge -- that is a ratio objective
    # and the Hungarian does not optimise ratios. Subtracting `threshold` from
    # every off-assignment cell converts it into one, exactly: for any
    # permutation pi,
    #
    #     cost_Q(pi) - opt = (cost_P(pi) - opt) - threshold * n_changed(pi)
    #
    # since pi pays the discount once per animal it re-partners. So
    # ``cost_Q(pi) <= opt`` is precisely ``(cost_P(pi) - opt) / n_changed(pi)
    # <= threshold``. One solve per cell, same cost as the total case, and no
    # iterative ratio search.
    if normalization == "per_edge":
        Q = P - threshold * (idx[None, :] != sigma[:, None])
        prune_limit = flag_limit = 0.0
    else:
        Q = P
        prune_limit, flag_limit = max(threshold, ceiling), threshold
    ref = float(Q[idx, sigma].sum())  # == opt; sigma pays no discount

    # Exact lower bound (see module docstring): dropping row i and any column
    # costs at least as much as dropping row i alone.
    L = np.empty(k)
    for i in range(k):
        rows = np.delete(idx, i)
        sub = Q[rows]
        rr, cc = linear_sum_assignment(sub)
        L[i] = sub[rr, cc].sum()
    bound = Q + L[:, None] - ref

    for i in range(k):
        for j in range(k):
            if j == sigma[i] or bound[i, j] > prune_limit + TIE_TOL:
                continue
            rows = np.delete(idx, i)
            cols = np.delete(idx, j)
            sub = Q[np.ix_(rows, cols)]
            rr, cc = linear_sum_assignment(sub)  # square -> rr == arange(k-1)
            score = float(Q[i, j] + sub[rr, cc].sum()) - ref

            alt = np.empty(k, dtype=int)
            alt[i] = j
            alt[rows] = cols[cc]
            differ = alt != sigma
            n_changed = int(differ.sum())  # >= 2; alt[i] != sigma[i] by construction

            # Report in the units the threshold is quoted in, not shifted ones.
            total = float(P[idx, alt].sum()) - opt
            gap = total / n_changed if normalization == "per_edge" else total
            if gap < row_gap[i]:
                row_gap[i] = gap

            if score > flag_limit + TIE_TOL:
                continue
            # The two permutations differ by a cycle; everyone on it is
            # mutually confusable, so flag the whole cycle, not just (i, j).
            row_flag |= differ
            col_flag[sigma[differ]] = True
            col_flag[alt[differ]] = True

    # An instance paired only with padding has no partner at all; that is not
    # evidence of a clean link, so treat it as ambiguous.
    row_flag[:n] |= sigma[:n] >= m
    matched_cols = np.zeros(k, dtype=bool)
    matched_cols[sigma[:n]] = True
    col_flag[:m] |= ~matched_cols[:m]

    return row_flag[:n], col_flag[:m], row_gap[:n], sigma[:n]


def two_swap_margin(
    S: np.ndarray, sigma: np.ndarray, normalization: str = GAP_SCALE
) -> np.ndarray:
    """Per-row cost increase of the cheapest pairwise swap, fully vectorised.

    Every 2-cycle is a valid alternative permutation, so this is an exact upper
    bound on the true minimum gap and is what the reported sweep is built from
    (the constrained solves above are used for the flags themselves). For
    assignment problems the minimising alternative is a 2-swap in all but
    pathological cases, so the two agree in practice.
    """
    n = len(sigma)
    if n < 2:
        return np.full(n, np.inf)
    matched = S[np.arange(n), sigma]          # similarity actually committed to
    cross = S[:, sigma]                       # cross[i, j] = S[i, sigma[j]]
    delta = matched[:, None] + matched[None, :] - cross - cross.T
    np.fill_diagonal(delta, np.inf)
    if normalization == "per_edge":
        delta = delta / 2.0                   # a swap re-partners exactly two
    return delta.min(axis=1)


# ---------------------------------------------------------------------------
# Driving it over the movie
# ---------------------------------------------------------------------------


def frame_geometry(labels, node_idx, cen_idx, method: str = SIMILARITY_METHOD):
    """Per-frame matching features and centroids, in original video coordinates.

    Both come off the same pose array in one pass.

    `features` is whatever SIMILARITY_METHOD needs -- boxes, poses, or upstream
    centroids -- and is what the cost matrix is built from.

    `centroids` is separate and always CENTROID_NODE, regardless of method: it
    is the reference build_id_images.py centres its crops on and the value
    written to the h5 `centroid` column, so a fragment's trajectory and its
    images agree on where the animal is. The "centroid" *method* uses a
    different quantity entirely (nanmedian over OVERLAP_NODES, upstream's
    definition) -- do not conflate the two.
    """
    lfs = sorted(labels.labeled_frames, key=lambda lf: lf.frame_idx)
    frames, features, centroids = [], [], []
    for lf in lfs:
        poses = np.stack([inst.numpy() for inst in lf.instances]).astype(float)
        centroids.append(poses[:, cen_idx, :].copy())
        sel = poses if node_idx is None else poses[:, node_idx, :]
        frames.append(int(lf.frame_idx))
        if method == "bounding_box":
            features.append(poses_to_bboxes(sel))
        elif method == "keypoint":
            features.append(sel)
        elif method == "centroid":
            features.append(poses_to_centroids(sel))
        else:
            raise ValueError(f"unknown SIMILARITY_METHOD {method!r}")
    return frames, features, centroids


def similarity(A: np.ndarray, B: np.ndarray, method: str = SIMILARITY_METHOD,
               scale: float | None = None) -> np.ndarray:
    """Frame-to-frame similarity in [0, 1], all-NaN instances degraded to zero.

    Dispatches on SIMILARITY_METHOD. Every branch returns [0, 1] so that the
    caller's `C = 1 - S` stays a valid cost against UNMATCHED_COST padding.
    """
    if method == "bounding_box":
        S = iou_matrix(A, B)
    elif method == "keypoint":
        S = oks_matrix(A, B)
    elif method == "centroid":
        if scale is None:
            raise ValueError("centroid similarity needs a body-length scale")
        S = centroid_matrix(A, B, scale)
    else:
        raise ValueError(f"unknown SIMILARITY_METHOD {method!r}")
    return np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)


def body_length_from_h5(out_dir: Path) -> float:
    """Median body length in px, read from the id-image files.

    build_id_images.py already derives this from the whole population
    (`collect_geometry`'s y_extent, measured in the egocentric frame so it is
    the animal's own length rather than an orientation-dependent projection).
    Reusing it keeps the centroid metric's scale identical to the one the crops
    were built with, instead of recomputing a subtly different number here.

    Reads the `body_length` attribute, NOT `height`. The two are equal only
    while config.MAX_CROP_AREA leaves the crops alone; once the area clamp
    engages, `height` is the reduced crop's own dimension while `body_length`
    stays in video pixels -- and video pixels are what the centroids being
    compared here are measured in. Falling back to `height` keeps files written
    before the clamp existed readable, where the two were the same number.
    """
    files = sorted(out_dir.glob("id_images_*.h5"))
    if not files:
        raise FileNotFoundError(
            f"no id_images_*.h5 in {out_dir}; the centroid similarity needs the "
            "body length that build_id_images.py measures")
    with h5py.File(files[0], "r") as fh:
        if "body_length" in fh.attrs:
            return float(fh.attrs["body_length"])
        return float(fh.attrs["height"])


def compute_flags(labels, node_idx, cen_idx, threshold: float,
                  method: str = SIMILARITY_METHOD, scale: float | None = None):
    """Walk consecutive frame pairs, collecting ambiguity AND the links together.

    One pass over the movie produces both outputs, because they come from the
    same solve: sigma is the assignment whose robustness the ambiguity test is
    measuring, and simultaneously the frame-to-frame correspondence.
    """
    frames, features, centroids = frame_geometry(labels, node_idx, cen_idx, method)
    amb_fwd = {f: np.zeros(len(b), dtype=bool) for f, b in zip(frames, features)}
    amb_bwd = {f: np.zeros(len(b), dtype=bool) for f, b in zip(frames, features)}
    # succ[t][i] = index of the instance in frame succ_frame[t] that instance i
    # of frame t was matched to; -1 where no partner exists.
    succ = {f: np.full(len(b), -1, dtype=int) for f, b in zip(frames, features)}
    succ_frame: dict[int, int] = {}

    margins: list[np.ndarray] = []
    exact_gaps: list[float] = []
    n_pairs = 0

    for (ft, A), (fu, B) in zip(zip(frames, features), zip(frames[1:], features[1:])):
        if fu - ft > MAX_FRAME_GAP:
            continue
        n_pairs += 1
        S = similarity(A, B, method, scale)
        C = 1.0 - S

        row_flag, col_flag, row_gap, sigma = assignment_ambiguity(C, threshold)
        amb_fwd[ft] |= row_flag
        amb_bwd[fu] |= col_flag

        # The link, straight off the same solve. sigma >= len(B) means the row
        # was matched only to padding, i.e. frame fu has no partner for it.
        paired = sigma < len(B)
        succ[ft][paired] = sigma[paired]
        succ_frame[ft] = fu

        finite = row_gap[np.isfinite(row_gap)]
        exact_gaps.extend(finite.tolist())

        if paired.all():
            margins.append(two_swap_margin(S, sigma))
        else:
            m = np.full(len(A), np.inf)
            if paired.sum() >= 2:
                m[paired] = two_swap_margin(S[paired], sigma[paired])
            margins.append(m)

    return dict(
        frames=frames, centroids=centroids, amb_fwd=amb_fwd, amb_bwd=amb_bwd,
        succ=succ, succ_frame=succ_frame,
        margins=np.concatenate(margins) if margins else np.empty(0),
        exact_gaps=np.asarray(exact_gaps), n_pairs=n_pairs,
    )


def resolve(res: dict, direction: str) -> dict[int, np.ndarray]:
    """Combine the two link directions into one per-frame `is_overlapping` array.

    A missing neighbour (movie start/end, or a gap wider than MAX_FRAME_GAP)
    contributes no evidence and therefore never flags -- absence of a link is
    not the same as an ambiguous one.
    """
    out = {}
    for f in res["frames"]:
        bad = np.zeros(len(res["amb_fwd"][f]), dtype=bool)
        if direction in ("both", "forward"):
            bad |= res["amb_fwd"][f]
        if direction in ("both", "backward"):
            bad |= res["amb_bwd"][f]
        out[f] = bad
    return out


# ---------------------------------------------------------------------------
# Writing, without disturbing anything already there
# ---------------------------------------------------------------------------

PRESERVE_NOTE = "datasets and attributes written by build_id_images.py"

# Namespace this stage owns. Everything outside it belongs to an earlier stage
# and is never touched.
ATTR_PREFIX = "overlap_"

# Names this stage wrote under a previous naming scheme. The per-run attribute
# cleanup cannot catch these, because the prefix itself changed -- so they are
# removed explicitly. This is a safety fix, not tidiness: a leftover
# `is_an_individual` sitting beside `is_overlapping` holds the OPPOSITE polarity,
# so a downstream reader that picks the wrong one gets exactly inverted
# semantics with no error.
LEGACY_DATASETS = ("is_an_individual",)
LEGACY_ATTR_PREFIXES = ("individual_",)

# Written by build_fragments.py from the links this stage produces. Rewriting the
# links invalidates it, so it is dropped rather than left to describe a graph
# that no longer exists.
DEPENDENT_DATASETS = ("fragment_identifier",)
DEPENDENT_ATTRS = ("n_fragments",)


def drop_stale(fh, path_name: str) -> None:
    """Remove anything this stage has invalidated, before the digest is taken."""
    dropped = []
    for name in LEGACY_DATASETS + DEPENDENT_DATASETS:
        if name in fh:
            del fh[name]
            dropped.append(name)
    for key in list(fh.attrs):
        if key.startswith(LEGACY_ATTR_PREFIXES) or key in DEPENDENT_ATTRS:
            del fh.attrs[key]
            dropped.append(key)
    if dropped:
        print(f"  {path_name}: dropped superseded {dropped}")


def snapshot(fh) -> dict:
    """Digest every existing dataset and attribute, to prove nothing changed."""
    digest = {}
    for key in sorted(fh.keys()):
        if key in ("is_overlapping", "next_global_index", "centroid"):
            continue
        digest[f"data:{key}"] = hashlib.sha1(
            np.ascontiguousarray(fh[key][...]).tobytes()
        ).hexdigest()
    for key in sorted(fh.attrs.keys()):
        # Skip this stage's OWN namespace, for the same reason the datasets above
        # are skipped: these are what it is here to rewrite. Digesting them made
        # the check pass only while the config never changed -- the first edit to
        # SIMILARITY_THRESHOLD turned a correct rewrite into a spurious
        # "pre-existing data was modified" failure.
        if key.startswith(ATTR_PREFIX):
            continue
        digest[f"attr:{key}"] = repr(fh.attrs[key])
    return digest


def main(slp_path: Path = SLP, out_dir: Path = OUT_DIR) -> None:
    t0 = time.time()
    print(f"loading {slp_path.name}")
    labels = sio.load_slp(str(slp_path))

    skel = labels[0].instances[0].skeleton
    if OVERLAP_NODES == "all":
        node_idx = None
    elif OVERLAP_NODES == "body":
        node_idx = [node_index(skel, n) for n in BODY_NODES]
    else:
        raise ValueError(f"unknown OVERLAP_NODES {OVERLAP_NODES!r}")
    if OVERLAP_DIRECTION not in ("both", "forward", "backward"):
        raise ValueError(f"unknown OVERLAP_DIRECTION {OVERLAP_DIRECTION!r}")
    if GAP_SCALE not in ("per_edge", "total"):
        raise ValueError(f"unknown GAP_SCALE {GAP_SCALE!r}")
    if REPORT_CEILING < SIMILARITY_THRESHOLD:
        raise ValueError("REPORT_CEILING must be >= SIMILARITY_THRESHOLD")

    cen_idx = node_index(skel, CENTROID_NODE)

    n_inst = sum(len(lf.instances) for lf in labels.labeled_frames)
    print(f"  {len(labels.labeled_frames)} frames, {n_inst} instances")
    # The centroid metric is the only one with a scale: it needs the median
    # body length to express distance in body lengths. Read from the id-image
    # files so it is the SAME number the crops were sized with.
    centroid_scale = (body_length_from_h5(out_dir)
                      if SIMILARITY_METHOD == "centroid" else None)
    if centroid_scale is not None:
        print(f"centroid similarity scale: {centroid_scale:.1f} px "
              "(median body length)")
    print(f"overlap: {SIMILARITY_METHOD}, {OVERLAP_NODES}-nodes, "
          f"{GAP_SCALE} gap <= {SIMILARITY_THRESHOLD} => overlapping, "
          f"direction={OVERLAP_DIRECTION}")

    print("scanning consecutive frame pairs (flags and links in one pass)")
    res = compute_flags(labels, node_idx, cen_idx, SIMILARITY_THRESHOLD,
                        SIMILARITY_METHOD, centroid_scale)
    overlapping = resolve(res, OVERLAP_DIRECTION)
    centroid_of = dict(zip(res["frames"], res["centroids"]))
    print(f"  {res['n_pairs']} frame pairs solved ({time.time() - t0:.0f}s)")

    # --- report -----------------------------------------------------------
    margins = res["margins"]
    units = (f"mean {SIMILARITY_METHOD} similarity given up per re-assigned animal"
             if GAP_SCALE == "per_edge" else "total assignment cost increase")
    print(f"\ncost of the cheapest alternative permutation, per instance")
    print(f"  measured as: {units}")
    print(f"  (2-swap; an exact UPPER bound on the true minimum gap. Flags "
          f"themselves come from the constrained solves, not from this.)")
    print(f"  min {margins.min():.4f}  p1 {np.percentile(margins, 1):.4f}  "
          f"median {np.median(margins):.4f}  max {margins.max():.4f}")
    print(f"  headroom: the smallest gap anywhere is "
          f"{margins.min() / SIMILARITY_THRESHOLD:.1f}x the threshold")
    if res["exact_gaps"].size:
        eg = res["exact_gaps"]
        print(f"  constrained solves under the {REPORT_CEILING} ceiling: "
              f"{eg.size}, min {eg.min():.4f}")
    else:
        print(f"  no cell came within {REPORT_CEILING} of optimal "
              "(every alternative was pruned by the exact bound)")

    print(f"\nthreshold sweep, over the {margins.size} instances that have a "
          f"forward link")
    print(f"  ({n_inst - margins.size} instances are in the final frame and have "
          f"no successor to compare against)")
    for tau in SWEEP:
        n_bad = int((margins <= tau).sum())
        print(f"  gap <= {tau:<5} -> {n_bad:6d} overlapping  "
              f"({100 * n_bad / margins.size:.2f}%)")

    # --- write ------------------------------------------------------------
    files = sorted(out_dir.glob("id_images_*.h5"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"no id_images_*.h5 in {out_dir}; run build_id_images first")

    # Links cross episode boundaries, so the (frame, instance) -> global_index
    # map has to cover every file before any of them can be written.
    gmap: dict[tuple[int, int], int] = {}
    for path in files:
        with h5py.File(path, "r") as fh:
            for f, i, g in zip(fh["frame_numbers"][...], fh["instance_idx"][...],
                               fh["global_index"][...]):
                key = (int(f), int(i))
                if key in gmap:
                    raise ValueError(f"{path.name}: (frame {key[0]}, instance {key[1]}) "
                                     "already claimed by another row")
                gmap[key] = int(g)

    total = n_over = n_linked = n_no_row = n_skipped = 0
    seen: set[tuple[int, int]] = set()
    print()
    for path in files:
        with h5py.File(path, "r+") as fh:
            # Before the digest, so removals are not misread as modifications.
            drop_stale(fh, path.name)
            before = snapshot(fh)

            frame_no = fh["frame_numbers"][...]
            inst_idx = fh["instance_idx"][...]
            n = len(frame_no)

            flags = np.zeros(n, dtype=bool)
            nxt = np.full(n, -1, dtype=np.int64)
            cent = np.full((n, 2), np.nan, dtype=np.float64)
            filled = np.zeros(n, dtype=bool)
            for row, (f, i) in enumerate(zip(frame_no, inst_idx)):
                f, i = int(f), int(i)
                if f not in overlapping:
                    raise KeyError(f"{path.name} row {row}: frame {f} absent from the .slp")
                if i >= len(overlapping[f]):
                    raise IndexError(f"{path.name} row {row}: instance {i} out of range "
                                     f"for frame {f} ({len(overlapping[f])} instances)")
                if (f, i) in seen:
                    raise ValueError(f"{path.name} row {row}: (frame {f}, instance {i}) "
                                     "already claimed by another row")
                seen.add((f, i))
                flags[row] = overlapping[f][i]
                cent[row] = centroid_of[f][i]

                # Resolve sigma's within-frame index into a global link. Per
                # spec, only keep it when both ends agree on is_overlapping --
                # a fragment must not span a change in the flag.
                j = int(res["succ"][f][i])
                if j >= 0:
                    fu = res["succ_frame"][f]
                    # The successor may exist in the .slp yet have no row here:
                    # build_id_images.py drops instances whose TOP_NODE or
                    # CENTROID_NODE is NaN, leaving `gmap` sparse in i (the
                    # surviving instances keep their SLEAP index, they are not
                    # renumbered). There is nothing to point at, so treat it
                    # exactly like "no partner" (j < 0) and let the chain end
                    # here. An unguarded gmap[...] raised KeyError instead,
                    # which took down the whole stage over one dropped instance.
                    successor = gmap.get((fu, j))
                    if successor is None:
                        n_no_row += 1
                    elif overlapping[fu][j] == overlapping[f][i]:
                        nxt[row] = successor
                filled[row] = True

            if not filled.all():
                raise AssertionError(f"{path.name}: {int((~filled).sum())} rows unlabelled")

            for name, data, dtype in (
                ("is_overlapping", flags, bool),
                ("next_global_index", nxt, np.int64),
                ("centroid", cent, np.float64),
            ):
                if name in fh:
                    del fh[name]
                fh.create_dataset(name, data=data, dtype=dtype)

            attrs = dict(
                overlap_metric=SIMILARITY_METHOD,
                overlap_nodes=OVERLAP_NODES,
                overlap_threshold=float(SIMILARITY_THRESHOLD),
                overlap_scale=(float(centroid_scale)
                               if centroid_scale is not None else float("nan")),
                overlap_direction=OVERLAP_DIRECTION,
                overlap_gap_scale=GAP_SCALE,
                overlap_max_frame_gap=int(MAX_FRAME_GAP),
            )
            # Files are reopened r+, so a knob that was renamed or dropped since
            # the last run would otherwise leave its old attribute sitting there
            # describing settings that are no longer in force. Clear the whole
            # `overlap_` namespace first so what is on disk is exactly what
            # produced the flags. Attributes owned by earlier stages are not in
            # this namespace and are left alone.
            for key in [k for k in fh.attrs if k.startswith(ATTR_PREFIX)]:
                if key not in attrs:
                    del fh.attrs[key]
            fh.attrs.update(attrs)

            after = snapshot(fh)
            stale = [k for k in before
                     if k.startswith(f"attr:{ATTR_PREFIX}") and k[5:] not in attrs]
            changed = [k for k in before
                       if k not in stale and before[k] != after.get(k)]
            if changed:
                raise AssertionError(f"{path.name}: pre-existing {PRESERVE_NOTE} "
                                     f"were modified: {changed}")
            if stale:
                print(f"  {path.name}: dropped stale attrs from an earlier config: "
                      f"{[k[5:] for k in stale]}")

            total += n
            n_skipped += int(fh.attrs.get("n_skipped", 0))
            n_over += int(flags.sum())
            n_linked += int((nxt >= 0).sum())
            print(f"  {path.name}: {n} rows, {int(flags.sum())} overlapping, "
                  f"{int((nxt >= 0).sum())} outgoing links")

    print(f"\nwrote is_overlapping / next_global_index / centroid to {len(files)} files")
    print(f"  overlapping : {n_over}/{total} ({100 * n_over / total:.2f}%)")
    print(f"  linked      : {n_linked}/{total} "
          f"({total - n_linked} with no successor"
          + (f", {n_no_row} of them because the successor has no row)"
             if n_no_row else ")"))
    # Rows + instances dropped for want of an alignment keypoint must account
    # for every SLEAP instance. Comparing rows to n_inst directly would fail on
    # the first dropped instance; the point of the check is that nothing is lost
    # SILENTLY, not that nothing is ever dropped. n_skipped is written by
    # build_id_images.py, which is the only stage that knows why a row is absent.
    if total + n_skipped != n_inst:
        raise AssertionError(
            f"{total} rows + {n_skipped} skipped != {n_inst} instances in the .slp")
    if len(seen) != total:
        raise AssertionError(f"{len(seen)} distinct (frame, instance) keys but {total} rows")
    if n_skipped:
        print(f"  every one of {n_inst} instances accounted for: {total} labelled, "
              f"{n_skipped} skipped for want of an alignment keypoint")
    else:
        print(f"  every one of {n_inst} instances labelled exactly once")
    print(f"elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
