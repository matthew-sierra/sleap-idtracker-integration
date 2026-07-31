"""Write the identities back into a .slp file as SLEAP Tracks.

Stage 6. Runs after ``build_identities.py``.

WHAT THIS DOES
--------------
The .slp SLEAP produced has no tracks at all -- ``labels.tracks == []`` and
every ``instance.track is None``. It holds poses, not identities. This stage
fills that in from the port's own output and writes a NEW file:

    <name>.predictions.slp   ->   <name>.idtracker_predictions.slp

Everything else is copied through untouched: same frames, same instances, same
keypoints, same skeleton, same video. The only difference is that each instance
now carries a ``Track``.

HOW THE MAPPING WORKS
---------------------
Two columns of the id-image HDF5 are all that is needed:

    instance_idx   the instance's position in ``lf.instances`` -- the index
                   SLEAP itself gave it (build_id_images.py:372). The column is
                   sparse in i when an instance was dropped for want of an
                   alignment keypoint, never renumbered, which is precisely what
                   makes it usable as a key here.
    identities     the identity build_identities.py assigned, 1..n_animals,
                   with 0 meaning unassigned.

So: row (frame f, instance_idx i, identity k) means
``labels[f].instances[i].track = tracks[k - 1]``.

``n_animals`` tracks are created -- the number of clusters the contrastive step
sorted the animals into -- and named for the identity they carry, so
``track.name == "7"`` is idtracker identity 7. A track that no row claims is
still created and simply stays empty; the count is a property of the clustering,
not of what happened to be used.

INSTANCES THAT GET NO TRACK
---------------------------
Three cases, all left with ``track=None`` rather than guessed at:

1. No HDF5 row at all -- dropped by build_id_images.py because TOP_NODE or
   CENTROID_NODE was NaN. The instance is still written to the .slp with its
   keypoints intact; it just has no identity.
2. identity 0 -- upstream's "unassigned" sentinel.
3. A collision: two instances in one frame assigned the same identity. Crossing
   rows are predicted independently with no mutual-exclusion pass, so this is
   possible. BOTH are left untracked -- writing one of the two would invent a
   resolution the evidence does not support. Counted and reported. This rule was
   decided for the retired build_trajectories.py and carried over rather than
   re-decided, so the two outputs never disagreed about an ambiguous frame.

Run:
    /opt/anaconda3/envs/sleap_id/bin/python src/sleap_idtracker/build_sleap_tracks.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import sleap_io as sio  # noqa: E402
from sleap_io import Track  # noqa: E402

OUT_DIR = config.OUT_DIR
SLP = config.SLP
IDTRACKER_SLP = config.IDTRACKER_SLP


def read_columns(out_dir: Path) -> dict[str, np.ndarray]:
    """The three columns this stage needs, concatenated in episode order."""
    files = sorted(out_dir.glob("id_images_*.h5"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"no id_images_*.h5 in {out_dir}")
    cols: dict[str, list[np.ndarray]] = {
        k: [] for k in ("frame_numbers", "instance_idx", "identities")}
    for path in files:
        with h5py.File(path, "r") as fh:
            for name in cols:
                if name not in fh:
                    raise KeyError(f"{path.name} has no {name!r}; "
                                   "run build_identities.py first")
                cols[name].append(fh[name][...])
            if fh.attrs.get("identities_source") is None:
                raise KeyError(f"{path.name} has no identities_source attribute; "
                               "run build_identities.py first")
    return {k: np.concatenate(v).astype(np.int64) for k, v in cols.items()}


def drop_collisions(frame: np.ndarray, ident: np.ndarray) -> np.ndarray:
    """Boolean mask of rows to apply. False where a (frame, identity) collides.

    A cell claimed more than once is claimed by nobody, rather than resolved by
    write order. Inherited from the retired build_trajectories.py:112.
    """
    keep = ident > 0
    cell = frame[keep] * (int(ident.max()) + 1) + ident[keep]
    _, inverse, counts = np.unique(cell, return_inverse=True, return_counts=True)
    ok = ~(counts > 1)[inverse]
    out = np.zeros(len(frame), dtype=bool)
    out[np.flatnonzero(keep)] = ok
    return out


def apply_tracks(labels, cols: dict[str, np.ndarray], n_animals: int) -> dict:
    """Attach a Track to every instance the HDF5 gives an identity. In place."""
    frame, inst, ident = (cols["frame_numbers"], cols["instance_idx"],
                          cols["identities"])
    if ident.size and ident.max() > n_animals:
        raise ValueError(f"identity {int(ident.max())} exceeds n_animals={n_animals}")

    tracks = [Track(name=str(i + 1)) for i in range(n_animals)]
    labels.tracks = tracks

    by_frame = {lf.frame_idx: lf for lf in labels.labeled_frames}
    keep = drop_collisions(frame, ident)

    n_set = 0
    for row in np.flatnonzero(keep):
        f, i, k = int(frame[row]), int(inst[row]), int(ident[row])
        lf = by_frame.get(f)
        if lf is None:
            raise KeyError(f"row {row}: frame {f} is not in {SLP.name}")
        if i >= len(lf.instances):
            raise IndexError(f"row {row}: instance_idx {i} out of range for "
                             f"frame {f} ({len(lf.instances)} instances)")
        lf.instances[i].track = tracks[k - 1]
        n_set += 1

    n_instances = sum(len(lf.instances) for lf in labels.labeled_frames)
    used = {t.name for lf in labels.labeled_frames for t in
            (i.track for i in lf.instances) if t is not None}
    return {
        "n_instances": n_instances,
        "n_rows": len(frame),
        "n_tracked": n_set,
        "n_unassigned": int((ident <= 0).sum()),
        "n_collided": int(((ident > 0) & ~keep).sum()),
        "n_no_row": n_instances - len(frame),
        "n_tracks": len(tracks),
        "n_tracks_used": len(used),
    }


def main(slp_path: Path = SLP, out_dir: Path = OUT_DIR,
         dest: Path = IDTRACKER_SLP) -> None:
    t0 = time.time()
    print(f"reading identities from {out_dir}")
    cols = read_columns(out_dir)
    n_animals = int(config.N_ANIMALS)
    print(f"  {len(cols['frame_numbers'])} rows, n_animals={n_animals}")

    print(f"loading {slp_path.name}")
    labels = sio.load_slp(str(slp_path))
    before_tracks = len(labels.tracks)
    # Snapshot the poses so the "instances are untouched" claim is checked
    # against the source, not merely asserted.
    before_pts = np.stack([i.numpy() for lf in labels.labeled_frames
                           for i in lf.instances])
    print(f"  {len(labels.labeled_frames)} frames, {len(before_pts)} instances, "
          f"{before_tracks} tracks")

    stats = apply_tracks(labels, cols, n_animals)
    print(f"\nassigned {stats['n_tracked']}/{stats['n_instances']} instances "
          f"to {stats['n_tracks']} tracks")
    for label, key in (("no HDF5 row (no alignment keypoint)", "n_no_row"),
                       ("identity 0 (unassigned)", "n_unassigned"),
                       ("collided, both left untracked", "n_collided")):
        if stats[key]:
            print(f"  {stats[key]} instances untracked: {label}")
    if stats["n_tracks_used"] < stats["n_tracks"]:
        print(f"  NOTE: {stats['n_tracks'] - stats['n_tracks_used']} of "
              f"{stats['n_tracks']} tracks are empty")

    dest.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(dest), verbose=False)
    print(f"\nwrote {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")

    # --- validation -------------------------------------------------------
    print("\n=== validation ===")
    reloaded = sio.load_slp(str(dest))
    after_pts = np.stack([i.numpy() for lf in reloaded.labeled_frames
                          for i in lf.instances])

    # Rebuild the expected identity of every instance straight from the HDF5,
    # then compare against what actually round-tripped through the file.
    keep = drop_collisions(cols["frame_numbers"], cols["identities"])
    expected: dict[tuple[int, int], str] = {
        (int(cols["frame_numbers"][r]), int(cols["instance_idx"][r])):
            str(int(cols["identities"][r]))
        for r in np.flatnonzero(keep)}
    got: dict[tuple[int, int], str] = {
        (lf.frame_idx, i): inst.track.name
        for lf in reloaded.labeled_frames
        for i, inst in enumerate(lf.instances) if inst.track is not None}

    checks = {
        "source file untouched": slp_path.resolve() != dest.resolve(),
        "same frame count": len(reloaded.labeled_frames) == len(labels.labeled_frames),
        "same instance count": len(after_pts) == len(before_pts),
        "keypoints bit-identical to the source":
            np.array_equal(before_pts, after_pts, equal_nan=True),
        f"file declares {n_animals} tracks": len(reloaded.tracks) == n_animals,
        "every tracked instance matches its HDF5 identity": got == expected,
        "no frame has two instances on one track": all(
            len({i.track.name for i in lf.instances if i.track is not None})
            == sum(i.track is not None for i in lf.instances)
            for lf in reloaded.labeled_frames),
    }
    if got != expected:
        missing = set(expected) - set(got)
        extra = set(got) - set(expected)
        wrong = {k for k in set(expected) & set(got) if expected[k] != got[k]}
        print(f"    missing {len(missing)}, unexpected {len(extra)}, "
              f"mismatched {len(wrong)}")

    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        raise AssertionError("track export validation failed")

    per_track = {t.name: sum(1 for lf in reloaded.labeled_frames
                             for i in lf.instances
                             if i.track is not None and i.track.name == t.name)
                 for t in reloaded.tracks}
    counts = sorted(per_track.values())
    print(f"\n  instances per track: min {counts[0]}, max {counts[-1]}, "
          f"total {sum(counts)}")
    print(f"\nALL TRACK EXPORT CHECKS PASS   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
