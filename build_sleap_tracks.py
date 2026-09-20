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
3. A collision the evidence cannot settle. Two instances in one frame can be
   assigned the same identity -- crossing rows are predicted independently with
   no mutual-exclusion pass -- and `drop_collisions` arbitrates: a single-image
   fragment always loses to a longer competitor, otherwise the higher max(P1)
   wins. Only when that cannot separate them (every claimant a crossing row with
   no pooled vote, or an exact tie on max(P1)) are ALL claimants left untracked,
   rather than resolving by write order. Counted and reported.

   The original rule dropped BOTH claimants unconditionally, inherited from the
   retired build_trajectories.py. It was replaced because it let a one-image
   crossing fragment blank a 3459-image individual fragment (measured on
   Pletcher_10fly, 1-based frame 16834, identity 1): both were discarded, so a
   stray singleton cost the real track a frame. build_trajectories.py still
   holds the old rule, so the two now disagree on contested frames by design.

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
        k: [] for k in ("frame_numbers", "instance_idx", "identities",
                        "fragment_identifier", "is_overlapping")}
    p1: list[np.ndarray] = []
    for path in files:
        with h5py.File(path, "r") as fh:
            for name in cols:
                if name not in fh:
                    raise KeyError(f"{path.name} has no {name!r}; "
                                   "run build_identities.py first")
                cols[name].append(fh[name][...])
            # Written by build_identities.py. Absent in sessions built before
            # the collision tiebreak existed -- fall back to all-NaN, which
            # makes every contest unresolvable and restores the old drop-both.
            p1.append(fh["p1_max"][...] if "p1_max" in fh
                      else np.full(len(fh["identities"]), np.nan))
            if fh.attrs.get("identities_source") is None:
                raise KeyError(f"{path.name} has no identities_source attribute; "
                               "run build_identities.py first")
    out = {k: np.concatenate(v).astype(np.int64) for k, v in cols.items()
           if k != "is_overlapping"}
    out["is_overlapping"] = np.concatenate(cols["is_overlapping"]).astype(bool)
    out["p1_max"] = np.concatenate(p1).astype(np.float64)
    return out


def drop_collisions(frame: np.ndarray, ident: np.ndarray,
                    frag_id: np.ndarray | None = None,
                    p1_max: np.ndarray | None = None,
                    overlapping: np.ndarray | None = None,
                    stats: dict | None = None) -> np.ndarray:
    """Boolean mask of rows to apply, resolving (frame, identity) contests.

    When two or more rows in one frame claim the same identity:

    0. BOTH IN AN OVERLAPPING FRAGMENT -> BOTH [none]. If every claimant is an
       `is_overlapping` row, all of them are left untracked. Set by explicit
       instruction, and kept as its own rule rather than left to fall out of
       step 2: it currently would, because those rows carry NaN p1_max and step
       2 cannot pick a winner among NaNs, but that is a consequence of how
       p1_max happens to be filled. If crossing rows ever gain a per-row
       confidence the guarantee would vanish silently. This does not depend on
       it.

    1. LENGTH, but only against a ONE-IMAGE claimant. A claimant carrying a
       single image of evidence loses to any competitor carrying more, with no
       score compared. Two things count as one image:

         - an `is_overlapping` row, whatever its fragment's length. Those rows
           are different animals predicted one at a time with no pooling, so the
           evidence behind any one row is exactly one network call. A five-row
           crossing fragment is not five images of evidence for any one of its
           identities.
         - a row of an INDIVIDUAL fragment holding one image, which pooled a
           single vote.

       Skipped when every claimant is a one-image claimant, since then there is
       no longer competitor to prefer.

    2. max(P1). Among what survives step 1, the highest max(P1) wins -- the
       fragment whose own images voted most decisively. P1 is normalised to sum
       to 1, so it is comparable across fragments of different lengths.

    Overlapping rows have no pooled vote and so no P1: they carry NaN and lose
    to any competitor that has one. Every claimant is left untracked when step 0
    applies, when no survivor has a finite max(P1), or when the maximum is tied
    -- never resolved by write order.

    Passing neither `frag_id` nor `p1_max` reproduces the old unconditional
    drop-both behaviour, so sessions written before `p1_max` existed still load.
    """
    keep = ident > 0
    rows = np.flatnonzero(keep)
    out = np.zeros(len(frame), dtype=bool)
    if rows.size == 0:
        return out
    cell = frame[rows] * (int(ident.max()) + 1) + ident[rows]
    _, inverse, counts = np.unique(cell, return_inverse=True, return_counts=True)
    contested = (counts > 1)[inverse]
    out[rows[~contested]] = True                      # uncontested: keep

    n_contest = n_both_ov = n_len = n_p1 = n_unres = 0
    if contested.any():
        legacy = frag_id is None or p1_max is None
        if not legacy:
            uf, cf = np.unique(frag_id, return_counts=True)
            lookup = dict(zip(uf.tolist(), cf.tolist()))
            size = np.array([lookup[int(f)] for f in frag_id[rows]], dtype=float)
            score = p1_max[rows]
            # Prefer the recorded flag; fall back to "has no pooled vote" for
            # files written before build_overlaps.py added the column.
            over = (np.asarray(overlapping, dtype=bool)[rows]
                    if overlapping is not None else np.isnan(score))
            eff = np.where(over, 1.0, size)
        order = np.argsort(inverse, kind="stable")
        bounds = np.flatnonzero(np.diff(inverse[order])) + 1
        for grp in np.split(order, bounds):
            if grp.size < 2:
                continue
            n_contest += 1
            if legacy:
                n_unres += 1
                continue
            if over[grp].all():                        # step 0
                n_both_ov += 1
                continue
            cand = grp
            if eff[grp].max() > 1:                     # step 1
                cand = grp[eff[grp] > 1]
            if cand.size == 1:
                out[rows[cand[0]]] = True
                n_len += 1
                continue
            sc = score[cand]                           # step 2
            fin = np.isfinite(sc)
            if not fin.any():
                n_unres += 1
                continue
            best = sc[fin].max()
            top = cand[fin][sc[fin] == best]
            if top.size == 1:
                out[rows[top[0]]] = True
                n_p1 += 1
            else:
                n_unres += 1
    if stats is not None:
        stats.update(n_contested_cells=n_contest, n_all_overlapping=n_both_ov,
                     n_resolved_by_length=n_len, n_resolved_by_p1=n_p1,
                     n_unresolved=n_unres)
    return out


def apply_tracks(labels, cols: dict[str, np.ndarray], n_animals: int) -> dict:
    """Attach a Track to every instance the HDF5 gives an identity. In place."""
    frame, inst, ident = (cols["frame_numbers"], cols["instance_idx"],
                          cols["identities"])
    if ident.size and ident.max() > n_animals:
        raise ValueError(f"identity {int(ident.max())} exceeds n_animals={n_animals}")

    # CLEAR ANY IDENTITY THE INPUT ALREADY CARRIES, before installing ours.
    #
    # Not defensive tidying -- required. The line below REPLACES labels.tracks,
    # but an instance holds its own reference to a Track object, and every
    # instance this stage does not re-point keeps pointing at one of the old
    # ones. Those objects are then reachable from the frames and absent from
    # labels.tracks, and sleap-io's writer resolves each instance's track by
    # `labels.tracks.index(inst.track)` (slp.py:1784), so it dies with
    #     ValueError: Track(name='...') is not in list
    # the moment one such instance is written. MEASURED on this input: the
    # corrected oline .slp carries 581 SLEAP tracks on 529048 instances.
    #
    # Clearing here rather than in a per-dataset preparation script means the
    # guarantee holds for ANY input: the port derives identity from scratch, and
    # the output declares exactly the n_animals tracks this function created --
    # which is what the validation in main() already asserts ("file declares
    # n_animals tracks", "no frame has two instances on one track"). The
    # contract was always this; it is now enforced rather than assumed.
    n_pre_existing = sum(1 for lf in labels.labeled_frames
                         for i in lf.instances if i.track is not None)
    if n_pre_existing:
        print(f"  input carries identity already: {n_pre_existing} instance(s) "
              f"on {len(labels.tracks)} track(s) -- cleared, since identity here "
              "is derived from scratch")
        # `pi`, NOT `inst`: this function unpacked `inst = cols["instance_idx"]`
        # at the top, and a loop variable named `inst` would leave that name
        # bound to the last PredictedInstance for the rest of the function --
        # which then fails in the assignment loop below with
        #   IndexError: Invalid indexing argument for skeleton: 0
        # because `inst[row]` indexes an instance by node instead of an array by
        # position. Caught in preflight; the name is deliberate.
        for lf in labels.labeled_frames:
            for pi in lf.instances:
                pi.track = None

    tracks = [Track(name=str(i + 1)) for i in range(n_animals)]
    labels.tracks = tracks

    by_frame = {lf.frame_idx: lf for lf in labels.labeled_frames}
    coll: dict = {}
    keep = drop_collisions(frame, ident, cols.get("fragment_identifier"),
                           cols.get("p1_max"), cols.get("is_overlapping"), coll)
    if coll.get("n_contested_cells"):
        print(f"  collisions: {coll['n_contested_cells']} contested (frame, identity) "
              f"cells -> {coll['n_all_overlapping']} all-overlapping (untracked), "
              f"{coll['n_resolved_by_length']} settled on length, "
              f"{coll['n_resolved_by_p1']} on max(P1), "
              f"{coll['n_unresolved']} otherwise untracked")

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
    keep = drop_collisions(cols["frame_numbers"], cols["identities"],
                           cols.get("fragment_identifier"), cols.get("p1_max"),
                           cols.get("is_overlapping"))
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
