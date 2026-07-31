"""Reconstruct SLEAP's per-frame association score/cost matrix and flag ambiguous
assignments — frames where a second track is nearly as good a match as the one the
tracker committed to (a candidate crossing/occlusion).

What it does
------------
SLEAP's simple tracker keeps only the *winning* similarity (`tracking_score`) and
throws the rest of the score matrix away, so the ".slp" cannot tell you whether an
assignment was a clear win or a coin-flip. This tool rebuilds that matrix from the
stored poses:

  for each frame t:
    detections D_t          = the instances at t (each already assigned a track)
    candidates for track j  = j's poses in frames [t-window, t-1]   (fixed window)
    S[i, j]                 = mean over j's window of OKS(D_i, candidate)   (score)
    C = -S                                                                  (cost)

Then for each detection it compares the score of its **finalized track** against the
best **alternative** track. If the runner-up is within `--threshold` (default 0.1)
of the winner, the assignment was ambiguous — the animal could almost as plausibly
be the other track — and the frame is flagged as a place to eyeball.

Fidelity caveat
---------------
This reproduces `features=keypoints, scoring_method=oks, scoring_reduction=mean,
window_size=5` exactly, BUT it does **not** apply optical flow. The `extended_vid`
run used `use_flow=True`, which motion-compensates candidates before scoring, so the
absolute scores here run a bit lower than the stored `tracking_score`. The tool
prints an agreement check (how often its top-scoring track equals the committed
track, and mean |recon - stored|) so you can judge the reconstruction. The *margin*
(gap between two tracks) is far less sensitive to flow than the absolute score,
which is what makes the ambiguity flags meaningful despite the caveat.

Run
---
    conda run -n sleap_id python -m sleap_idtracker.assignment_margin \
        --slp extended_vid.v020.slp.predictions.slp \
        --out assignment_margin_extended_vid --threshold 0.1
"""

from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import sleap_io as sio

# SLEAP `standard` palette (RGB) + human color names, for naming the individuals.
_PALETTE = [
    ("0,114,189", "blue"),
    ("217,83,25", "orange-red"),
    ("237,177,32", "amber"),
    ("126,47,142", "purple"),
    ("119,172,48", "olive-green"),
    ("77,190,238", "light-blue"),
    ("162,20,47", "dark-red"),
]


def oks_matrix(gt: np.ndarray, pr: np.ndarray, stddev: float = 0.025) -> np.ndarray:
    """Vectorized OKS between two pose sets, matching `sleap_nn.evaluation.compute_oks`.

    gt: (nG, N, 2), pr: (nP, N, 2)  ->  (nG, nP) in [0, 1] (cocoeval normalization,
    scale = gt bbox area). NaN keypoints are treated as missing exactly as upstream.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gx, gy = gt[..., 0], gt[..., 1]  # (nG, N)
        xr = np.nanmax(gx, 1) - np.nanmin(gx, 1)
        yr = np.nanmax(gy, 1) - np.nanmin(gy, 1)
    area = np.where(np.isfinite(xr) & np.isfinite(yr), xr * yr, np.nan)  # (nG,)

    disp = gt[:, None, :, :] - pr[None, :, :, :]  # (nG, nP, N, 2)
    d2 = (disp ** 2).sum(-1)  # (nG, nP, N)
    missing_pr = np.isnan(pr).any(-1)  # (nP, N)
    d2 = np.where(missing_pr[None, :, :], np.inf, d2)

    spread = (2 * stddev) ** 2
    scale_factor = 2 * (area + np.spacing(1))  # (nG,)
    norm = spread * scale_factor[:, None, None]  # (nG, 1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ks = np.exp(-(d2 / norm))  # (nG, nP, N)

    missing_gt = np.isnan(gt).any(-1)  # (nG, N)
    ks = np.where(missing_gt[:, None, :], 0.0, ks)
    n_vis = (~missing_gt).sum(-1).astype(float)  # (nG,)
    n_vis[n_vis == 0] = np.nan
    return ks.sum(-1) / n_vis[:, None]  # (nG, nP)


def poses_to_bboxes(poses: np.ndarray) -> np.ndarray:
    """(n, N, 2) poses -> (n, 4) `[xmin, ymin, xmax, ymax]`, matching `utils.get_bbox`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        xmin = np.nanmin(poses[..., 0], axis=1)
        ymin = np.nanmin(poses[..., 1], axis=1)
        xmax = np.nanmax(poses[..., 0], axis=1)
        ymax = np.nanmax(poses[..., 1], axis=1)
    return np.stack([xmin, ymin, xmax, ymax], axis=1)


def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Vectorized bbox IoU, matching `sleap_nn.tracking.utils.compute_iou`.

    A: (nA, 4), B: (nB, 4), each `[xmin, ymin, xmax, ymax]` -> (nA, nB) in [0, 1].
    Uses the same `+1` (inclusive-pixel) convention as upstream.
    """
    ax0, ay0, ax1, ay1 = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    bx0, by0, bx1, by1 = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
    ix0 = np.maximum(ax0[:, None], bx0[None, :])
    iy0 = np.maximum(ay0[:, None], by0[None, :])
    ix1 = np.minimum(ax1[:, None], bx1[None, :])
    iy1 = np.minimum(ay1[:, None], by1[None, :])
    inter = np.clip(ix1 - ix0 + 1, 0, None) * np.clip(iy1 - iy0 + 1, 0, None)
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    union = area_a[:, None] + area_b[None, :] - inter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return inter / union


def poses_to_centroids(poses: np.ndarray) -> np.ndarray:
    """(n, N, 2) poses -> (n, 2), matching `sleap_nn.tracking.utils.get_centroid`.

    Upstream's centroid is the *nanmedian over the instance's keypoints*, not a
    named node. That is deliberately not the same thing as this port's
    CENTROID_NODE (thorax), which centres the id-image crops and fills the h5
    `centroid` column: one is a robust average of the pose, the other is an
    anatomical landmark. Matching upstream here keeps `similarity_matrix` a
    faithful port of the SLEAP tracker's scoring.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(poses, axis=1)


def centroid_matrix(A: np.ndarray, B: np.ndarray, scale: float) -> np.ndarray:
    """Vectorized centroid similarity: ``exp(-d / scale)``. A: (nA,2), B: (nB,2).

    `sleap_nn.tracking.utils.compute_euclidean_distance` returns the NEGATIVE
    distance -- a score where higher is better, but unbounded below. The
    assignment solver here needs a similarity in [0, 1], because the cost is
    `1 - S` and must stay finite and non-negative against UNMATCHED_COST
    padding. So the distance is mapped through a decaying exponential rather
    than used raw.

    `scale` is the median body length, so the unit is "body lengths travelled":
    d = 0 -> 1.0, one body length -> 0.368, two -> 0.135. Scaling by a measured
    quantity rather than a pixel constant means the metric transfers to other
    animals without retuning. The exponential never reaches exactly 0, so no
    pair is ever hard-excluded and the solver sees no discontinuity.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"centroid similarity needs a positive scale, got {scale!r}")
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)  # (nA, nB)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.exp(-d / scale)


def track_color(idx: int) -> str:
    """`track_<idx>` -> 'name #RRGGBB' from the cycling standard palette."""
    rgb, name = _PALETTE[idx % len(_PALETTE)]
    r, g, b = (int(v) for v in rgb.split(","))
    return f"{name} #{r:02X}{g:02X}{b:02X}"


def compute_flags(
    labels: sio.Labels,
    window: int,
    threshold: float,
    stddev: float,
    feature: str = "keypoints",
) -> pd.DataFrame:
    """Walk frames in order, rebuild the score matrix, and flag ambiguous assignments.

    ``feature="keypoints"`` scores with OKS (SLEAP's default); ``feature="bboxes"``
    scores with bbox IoU (the ``features="bboxes", scoring_method="iou"`` path). Both
    similarities are in [0, 1], so the same margin/threshold logic applies.
    """
    # Track name -> palette index (position in the Labels track list).
    track_idx = {t.name: i for i, t in enumerate(labels.tracks)}

    lfs = sorted(labels.labeled_frames, key=lambda lf: lf.frame_idx)
    history: deque = deque(maxlen=window)  # each item: {track_name: (N,2) pose}

    rows: List[Dict] = []
    agree_hits = agree_tot = 0
    abs_diffs: List[float] = []

    for lf in lfs:
        dets = [(i.track.name, i.numpy(), i.tracking_score) for i in lf.instances if i.track]
        if dets and history:
            D = np.stack([d[1] for d in dets])  # (nD, N, 2) poses
            D_bbox = poses_to_bboxes(D) if feature == "bboxes" else None

            active = sorted({t for fr in history for t in fr})
            # Score matrix S[i, j] = mean over track j's windowed poses of the
            # chosen similarity (OKS for keypoints, IoU for bboxes).
            S = np.full((len(dets), len(active)), np.nan)
            for j, tname in enumerate(active):
                cand = [fr[tname] for fr in history if tname in fr]
                if not cand:
                    continue
                Cj = np.stack(cand)  # (kj, N, 2) poses
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if feature == "bboxes":
                        sim = iou_matrix(D_bbox, poses_to_bboxes(Cj))
                    else:
                        sim = oks_matrix(D, Cj, stddev)
                    S[:, j] = np.nanmean(sim, axis=1)

            col = {t: j for j, t in enumerate(active)}
            for i, (aname, _, stored_ts) in enumerate(dets):
                if aname not in col or np.isnan(S[i, col[aname]]):
                    continue  # committed track has no window history (a birth) -> skip
                assigned = S[i, col[aname]]
                # agreement / validation vs the stored (flow) score
                top_j = int(np.nanargmax(S[i]))
                agree_tot += 1
                if active[top_j] == aname:
                    agree_hits += 1
                # MAD vs the stored score is only meaningful when scoring with the
                # same metric SLEAP used (OKS); in bbox mode it's cross-metric.
                if (
                    feature == "keypoints"
                    and stored_ts is not None
                    and not np.isnan(stored_ts)
                ):
                    abs_diffs.append(abs(assigned - float(stored_ts)))

                # best alternative track (exclude the committed one)
                others = S[i].copy()
                others[col[aname]] = -np.inf
                best_j = int(np.nanargmax(others))
                best_other = others[best_j]
                if not np.isfinite(best_other):
                    continue
                gap = assigned - best_other
                if gap <= threshold:  # runner-up within threshold (or better)
                    n_close = int(
                        np.sum(
                            (np.nan_to_num(S[i], nan=-np.inf) >= assigned - threshold)
                        )
                        - 1  # exclude self
                    )
                    rows.append(
                        {
                            "frame_idx": int(lf.frame_idx),
                            "individual": aname,
                            "individual_color": track_color(track_idx.get(aname, 0)),
                            "assigned_score": round(float(assigned), 4),
                            "competitor": active[best_j],
                            "competitor_color": track_color(
                                track_idx.get(active[best_j], 0)
                            ),
                            "competitor_score": round(float(best_other), 4),
                            "gap": round(float(gap), 4),
                            "n_within_thresh": n_close,
                        }
                    )

        history.append({name: pose for name, pose, _ in dets})

    df = pd.DataFrame(rows)
    agree = agree_hits / agree_tot if agree_tot else float("nan")
    mad = float(np.mean(abs_diffs)) if abs_diffs else float("nan")
    df.attrs["agreement"] = agree
    df.attrs["mad_vs_stored"] = mad
    return df


def group_events(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse runs of consecutive flagged frames for the same (individual, competitor)
    pair into single events, ranked by the tightest (smallest) gap."""
    if df.empty:
        return df
    df = df.sort_values(["individual", "competitor", "frame_idx"]).reset_index(drop=True)
    key = (df["individual"] != df["individual"].shift()) | (
        df["competitor"] != df["competitor"].shift()
    ) | (df["frame_idx"].diff() != 1)
    df["event"] = key.cumsum()
    out = []
    for _, ev in df.groupby("event"):
        tight = ev.loc[ev["gap"].idxmin()]
        out.append(
            {
                "individual": ev["individual"].iloc[0],
                "individual_color": ev["individual_color"].iloc[0],
                "competitor": ev["competitor"].iloc[0],
                "competitor_color": ev["competitor_color"].iloc[0],
                "start": int(ev["frame_idx"].min()),
                "end": int(ev["frame_idx"].max()),
                "n_frames": len(ev),
                "min_gap": round(float(ev["gap"].min()), 4),
                "at_frame": int(tight["frame_idx"]),
            }
        )
    return pd.DataFrame(out).sort_values("min_gap").reset_index(drop=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--slp",
        default="/Users/matthewsierra/sleap_idtracker/extended_vid.v020.slp.predictions.slp",
    )
    ap.add_argument(
        "--out",
        default="/Users/matthewsierra/sleap_idtracker/assignment_margin_extended_vid",
    )
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="Flag when runner-up is within this of the committed score.")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--stddev", type=float, default=0.025)
    ap.add_argument("--feature", choices=["keypoints", "bboxes"], default="keypoints",
                    help="keypoints -> OKS similarity; bboxes -> bbox IoU.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = sio.load_slp(args.slp)
    df = compute_flags(labels, args.window, args.threshold, args.stddev, args.feature)

    scoring = "IoU (bboxes)" if args.feature == "bboxes" else "OKS (keypoints)"
    n_frames = len({int(lf.frame_idx) for lf in labels.labeled_frames})
    print(f"loaded {args.slp}")
    print(f"  frames: {n_frames} | tracks: {len(labels.tracks)} | scoring: {scoring}")
    print(f"  agreement (top-scoring track == committed track): "
          f"{df.attrs['agreement']*100:.1f}%")
    if args.feature == "keypoints":
        print(f"  mean |recon assigned_score - stored tracking_score|: "
              f"{df.attrs['mad_vs_stored']:.3f}  (gap from omitting optical flow)")

    if df.empty:
        print(f"  no assignments with a runner-up within {args.threshold}.")
        return

    df.to_csv(out / "ambiguous_instances.csv", index=False)
    events = group_events(df)
    events.to_csv(out / "ambiguous_events.csv", index=False)

    print(f"  flagged instances (runner-up within {args.threshold}): {len(df)}")
    print(f"  distinct ambiguous events: {len(events)}")
    print(f"  wrote {out/'ambiguous_instances.csv'} and {out/'ambiguous_events.csv'}")

    print("\n=== places to look (tightest 25 events; smaller gap = more ambiguous) ===")
    print(f"{'rank':>4}  {'frames':<13} {'n':>3} {'gap':>7}  individual  vs  competitor")
    print("-" * 78)
    for i, e in enumerate(events.head(25).itertuples(), 1):
        span = f"{e.start}-{e.end}" if e.start != e.end else f"{e.start}"
        print(
            f"{i:>4}  {span:<13} {e.n_frames:>3} {e.min_gap:>7.3f}  "
            f"{e.individual} ({e.individual_color})  vs  "
            f"{e.competitor} ({e.competitor_color})"
        )


if __name__ == "__main__":
    main()
