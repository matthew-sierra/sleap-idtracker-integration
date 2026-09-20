"""Build idtracker.ai's ``ListOfGlobalFragments`` from our fragment files.

Stage 3f. Runs after ``build_fragments.py``.

A *global fragment* is a stretch where every animal is simultaneously in its own
individual fragment -- the only stretches where identities can be learned without
ambiguity, and what the accumulation protocol trains on.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
Upstream's constructor is ``ListOfGlobalFragments.from_fragments(blobs_in_video,
fragments, num_animals)`` (``list_of_global_fragments.py:87``), and its first
argument is a list of ``Blob`` objects. This port has no blobs -- that is the
whole point of it -- so that one classmethod cannot be called.

Everything else is upstream's and is used unmodified:

    GlobalFragment(fragments)          builds one global fragment
    ListOfGlobalFragments(gfs)         splits accumulable / non-accumulable
    ListOfGlobalFragments.save(path)   upstream's own JSON format

Only the core detection is re-expressed against our arrays instead of blobs, and
it follows ``get_global_fragment_core`` (``list_of_global_fragments.py:221``)
condition for condition.

THE CORE TEST
-------------
Upstream:

    is_clean_frame  = len(blobs_in_frame) == n_animals
                      and all(b.is_an_individual for b in blobs_in_frame)
    was_clean_frame = ... same, for the previous frame
    same_ids        = {fragment ids now} == {fragment ids before}
    core            = is_clean_frame and (same_ids or not was_clean_frame)

``is_clean_frame`` is just "this frame holds the right number of individuals",
which is the whole test for whether a frame *qualifies*. The second clause does
something different: it decides where one global fragment ENDS and the next
BEGINS. A run of clean frames can span a fragment turnover -- one animal's
individual fragment ending and another starting -- and those are two different
global fragments even though every frame in the run is clean. Without the clause
the run would collapse into one global fragment holding a stale set of Fragment
objects, since a GlobalFragment stores fragment references rather than frames.

``is_an_individual`` is ``not is_overlapping`` here; see build_overlaps.py.

WHEN THERE ARE NO GLOBAL FRAGMENTS
----------------------------------
A video where the animals are never all separate at once yields none. That is a
supported outcome, not a failure -- identification then runs on the contrastive
embedding alone, clustered by K-Means seeded with ``k-means++`` and ``k`` equal
to the user's ``N_ANIMALS`` (set in build_fragments.py). idtracker.ai already
does this unprompted: ``first_gfrag=None`` makes ``ContrastiveLearning`` drop the
global-fragment seeding (contrastive.py:435) and fall back to ``k-means++``
(contrastive.py:854), and ``contrastive_step`` returns a ratio of ``inf``
(tracker.py:276) which makes the caller skip the accumulation protocol outright
(tracker.py:110). None of that needed an edit.

What this module adds is refusing to report that case as an ordinary success.
Every structural check here has the form ``all(... for g in allgf)``, and
``all()`` of an empty iterable is ``True``, so an empty result would otherwise
print a full row of PASS and the "ALL CHECKS PASS" headline having examined
nothing. Instead the empty case is branched out and validated on its own terms:
what k-means++ actually needs is co-occurring animals (contrastive.py:291), not
global fragments, and that is what gets checked. See ``report_kmeans_fallback``.

Run:
    /opt/anaconda3/envs/sleap_id/bin/python src/sleap_idtracker/build_global_fragments.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# config must be imported BEFORE idtrackerai: it is what puts the bundled
# idtrackerai/src on sys.path, so a fresh clone resolves these imports without
# idtracker.ai having been pip-installed first.
import config  # noqa: E402

from idtrackerai.globalfragment import GlobalFragment  # noqa: E402
from idtrackerai.list_of_fragments import ListOfFragments  # noqa: E402
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments  # noqa: E402
from idtrackerai.utils import conf  # noqa: E402

REPO = config.REPO
OUT_DIR = config.OUT_DIR
FRAGMENTS_JSON = config.FRAGMENTS_JSON
GLOBAL_FRAGMENTS_JSON = config.GLOBAL_FRAGMENTS_JSON

# Set in config.py. Applied by pushing it into upstream's config singleton via
# upstream's own setter, so the filter that consumes it is idtracker.ai's,
# untouched (``ListOfGlobalFragments.__init__``, list_of_global_fragments.py:72).
MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION = (
    config.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION)


# ---------------------------------------------------------------------------
# Core detection, blob-free
# ---------------------------------------------------------------------------


def per_frame_state(out_dir: Path) -> tuple[np.ndarray, list[frozenset], np.ndarray]:
    """Read the id-image files into per-frame views.

    Returns ``(frames, fragment_ids_per_frame, n_individuals_per_frame,
    is_empty)``, all indexed by a DENSE frame axis spanning the first to the
    last frame that carries a row. ``is_empty[k]`` marks a frame that carries no
    row at all, which is distinct from a frame whose rows fail the clean test.
    """
    files = sorted(out_dir.glob("id_images_*.h5"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    frame, frag, over = [], [], []
    for path in files:
        with h5py.File(path, "r") as fh:
            for name in ("fragment_identifier", "is_overlapping"):
                if name not in fh:
                    raise KeyError(f"{path.name} has no {name!r}; "
                                   "run build_fragments.py first")
            frame.append(fh["frame_numbers"][...])
            frag.append(fh["fragment_identifier"][...])
            over.append(fh["is_overlapping"][...])
    frame = np.concatenate(frame)
    frag = np.concatenate(frag)
    over = np.concatenate(over)

    frames = np.arange(int(frame.min()), int(frame.max()) + 1)

    order = np.argsort(frame, kind="stable")
    frame, frag, over = frame[order], frag[order], over[order]
    bounds = np.searchsorted(frame, frames)
    bounds = np.append(bounds, len(frame))

    # A frame SLEAP predicted nothing in has no row here, so `searchsorted`
    # gives it lo == hi and it falls out of the loop as an empty fragment set
    # with zero individuals. That is the truth about such a frame, and it is
    # already what the core scan wants: `clean` is False there, so the frame can
    # neither be a core nor continue one, and a run of clean frames spanning it
    # is correctly split.
    #
    # Emptiness is therefore handled whether or not it is named. `is_empty`
    # names it because `clean` goes False for two unrelated reasons -- no data at
    # all, or data showing the wrong number of individuals -- and only the second
    # says anything about the animals. Collapsing them loses the distinction at
    # exactly the point a reader wants it, and it is the difference between
    # "SLEAP found nothing" and "SLEAP found the wrong thing". Derived from
    # `lo == hi` rather than recomputed from the frame numbers, so the label and
    # the empty set cannot disagree.
    ids_per_frame, n_individuals = [], np.zeros(len(frames), dtype=int)
    is_empty = np.zeros(len(frames), dtype=bool)
    for k in range(len(frames)):
        lo, hi = bounds[k], bounds[k + 1]
        is_empty[k] = lo == hi
        ids_per_frame.append(frozenset(frag[lo:hi].tolist()))
        n_individuals[k] = int((~over[lo:hi]).sum())

    if is_empty.any():
        print(f"  {int(is_empty.sum())}/{len(frames)} frames labelled EMPTY "
              "(no id-image row; SLEAP predicted nothing there)")
    return frames, ids_per_frame, n_individuals, is_empty


def core_flags(ids_per_frame, n_individuals, n_per_frame, n_animals: int) -> np.ndarray:
    """Port of ``get_global_fragment_core`` (list_of_global_fragments.py:221)."""
    n = len(ids_per_frame)
    clean = (n_per_frame == n_animals) & (n_individuals == n_animals)

    core = np.zeros(n, dtype=bool)
    # Upstream seeds index 0 to False and evaluates from the first pair onward.
    for k in range(1, n):
        if not clean[k]:
            continue
        same_ids = ids_per_frame[k] == ids_per_frame[k - 1]
        core[k] = same_ids or not clean[k - 1]
    return core


def first_frames_of_cores(core: np.ndarray) -> list[int]:
    """Indices where a core starts, i.e. the [False, True] transitions."""
    return [k for k in range(1, len(core)) if core[k] and not core[k - 1]]


# ---------------------------------------------------------------------------
# The no-global-fragment case
# ---------------------------------------------------------------------------


def report_kmeans_fallback(lof: ListOfFragments, n_animals: int) -> bool:
    """Check that the k-means++ identification path can actually run. -> ok?

    An empty global fragment set is a legitimate outcome, not a failure. When it
    happens, idtracker.ai does not accumulate at all; it identifies purely from
    the contrastive embedding, clustered by K-Means seeded with ``k-means++``:

        tracker.py:75     first_global_fragment = None
        contrastive.py:435    -> gfrag_loader = None
        contrastive.py:854    -> kmeans_init() = {"n_init": 20, "init": "k-means++"}
        contrastive.py:797    -> MiniBatchKMeans(n_animals, **kmeans_init())
        tracker.py:276    -> ratio = inf -> accumulation protocol skipped entirely

    Nothing above needs an edit -- it is already the behaviour we want. What that
    path DOES need is preconditions that are cheap to check here and expensive to
    discover halfway through training, so they are checked here.

    The binding one is co-occurrence. Contrastive learns by pushing apart animals
    seen at the same time; with no co-occurring pairs it raises outright
    (contrastive.py:291). Global fragments are not required, but co-occurrence is.
    """
    min_len = conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION
    long_fragments = [f for f in lof.fragments
                      if f.is_an_individual and f.n_images >= min_len]

    # Negative pairs, counted exactly as ContrastiveLearning.__init__ does
    # (contrastive.py:274) so this check cannot drift from the thing it guards.
    n_negative_pairs = sum(
        1
        for frag in long_fragments
        for coex in frag.coexisting_individual_fragments
        if coex.identifier > frag.identifier
        and coex.is_an_individual
        and coex.n_images >= min_len
    )
    n_images = sum(f.n_images for f in long_fragments)

    print(f"  identification will cluster with k = n_animals = {n_animals}")
    print(f"  {len(long_fragments)} individual fragments of >= {min_len} frames, "
          f"{n_images} images to cluster")

    # Note what is NOT checked: that n_animals matches anything observed in the
    # video. n_animals is the identity count, i.e. k, and k-means++ will happily
    # form k clusters from any embedding with at least k points. The binding
    # requirement is co-occurrence -- animals seen together, which is the only
    # signal contrastive can learn "different" from.
    checks = {
        f"at least {n_animals} images to form {n_animals} clusters":
            n_images >= n_animals,
        "some pair of animals appears together at some point "
        "(contrastive.py:291)": n_negative_pairs > 0,
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    if n_negative_pairs:
        connectivity = lof.get_connectivity()
        # Mirrors the threshold contrastive_step warns on (tracker.py:248).
        verdict = "good enough" if connectivity >= 0.5 else "LOW -- animals too isolated"
        print(f"  fragment connectivity {connectivity:.2f} ({verdict})")

    return all(checks.values())


def main(out_dir: Path = OUT_DIR, fragments_json: Path = FRAGMENTS_JSON,
         json_path: Path = GLOBAL_FRAGMENTS_JSON) -> None:
    t0 = time.time()

    # Upstream's own setter, so upstream's own filter reads it.
    conf.set_parameters(
        MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION=(
            MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION),
    )
    print(f"MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION = "
          f"{conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION}")

    lof = ListOfFragments.load(fragments_json)
    fragments = lof.fragments
    n_animals = lof.n_animals
    print(f"loaded {len(fragments)} fragments, n_animals={n_animals}")

    frames, ids_per_frame, n_individuals, is_empty = per_frame_state(out_dir)
    n_per_frame = np.array([len(s) for s in ids_per_frame])
    # Quoted over the frames that HAVE data: including the empties would drag
    # the minimum to 0 and say nothing about how many animals were found.
    live = n_per_frame[~is_empty]
    print(f"  {len(frames)} frames, {int(live.min())}-{int(live.max())} fragments "
          f"alive per frame over the {int((~is_empty).sum())} non-empty ones")

    core = core_flags(ids_per_frame, n_individuals, n_per_frame, n_animals)
    starts = first_frames_of_cores(core)
    clean = (n_per_frame == n_animals) & (n_individuals == n_animals)
    print(f"  clean frames (right number of individuals): {int(clean.sum())}/{len(frames)}")
    print(f"    not clean: {int((~clean & ~is_empty).sum())} with data, "
          f"{int(is_empty.sum())} empty")
    print(f"  core frames: {int(core.sum())}, distinct cores: {len(starts)}")

    global_fragments = [
        GlobalFragment([fragments[fid] for fid in sorted(ids_per_frame[k])])
        for k in starts
    ]
    logf = ListOfGlobalFragments(global_fragments)

    print(f"\n{len(logf.global_fragments)} accumulable, "
          f"{len(logf.non_accumulable_global_fragments)} non-accumulable")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    logf.save(json_path)
    print(f"saved {json_path}")

    # --- validation -------------------------------------------------------
    print("\n=== validation ===")
    allgf = logf.global_fragments + logf.non_accumulable_global_fragments

    # Every check below is `all(... for g in allgf)`, and all() over an empty
    # iterable is True. On a video with no global fragments they would therefore
    # every one of them "PASS" without having examined anything, and the module
    # would announce success over an empty JSON. Branch before that can happen:
    # zero global fragments is a real, supported outcome with its own downstream
    # path, so it gets its own report rather than a borrowed one.
    if not allgf:
        print("  no global fragments -- no frame holds all "
              f"{n_animals} animals as separate individuals at once")
        print("  this is a supported outcome, not a failure; identification "
              "falls back to k-means++\n")
        ok = report_kmeans_fallback(lof, n_animals)

        reloaded = ListOfGlobalFragments.load(json_path, fragments)
        rt = not reloaded.global_fragments
        print(f"  {'PASS' if rt else 'FAIL'}  empty set round-trips through "
              "upstream's own loader")

        if not (ok and rt):
            raise AssertionError(
                "no global fragments AND the k-means++ fallback cannot run; "
                "see the FAIL lines above")

        print("\nNO GLOBAL FRAGMENTS -- IDENTIFICATION WILL USE K-MEANS++ "
              f"WITH k={n_animals}   ({time.time() - t0:.0f}s)")
        return

    checks = {
        "every global fragment holds n_animals fragments": all(
            len(g.fragments_identifiers) == n_animals for g in allgf),
        "no duplicate fragment within a global fragment": all(
            len(set(g.fragments_identifiers)) == n_animals for g in allgf),
        "all member fragments are individuals": all(
            f.is_an_individual for g in allgf for f in g),
        "accumulable split matches the threshold": all(
            g.min_n_images_per_fragment
            >= conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION
            for g in logf.global_fragments)
            and all(g.min_n_images_per_fragment
                    < conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION
                    for g in logf.non_accumulable_global_fragments),
        "members coexist at the core frame": all(
            all(f.start_frame <= g.first_frame_of_the_core < f.end_frame for f in g)
            for g in allgf),
        "is_unique(n_animals) before identities assigned": all(
            len(set(g.fragments_identifiers)) == n_animals for g in allgf),
        # The invariant that makes labelling a frame EMPTY safe rather than
        # merely tidy. An empty frame has n_per_frame == 0, so `clean` is False
        # and core_flags skips it -- meaning it can never reach `starts` and can
        # never have GlobalFragment() built over its (empty) fragment set. If
        # this ever fails, a global fragment was constructed from no fragments.
        "no empty frame is a core": not bool(core[is_empty].any()),
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    # Two things separate the full rule from a bare "right number of individuals".
    # They are reported apart because only one of them is inert on this dataset.
    #
    # (a) The seed. Upstream forces index 0 to False (the `[False] + [...]` in
    #     from_fragments) so that a core which is already running at frame 0 still
    #     produces a [False, True] transition. Without it, a movie that starts
    #     clean yields NO global fragments at all -- which is this movie.
    unseeded = [k for k in range(1, len(clean)) if clean[k] and not clean[k - 1]]
    print(f"  seed matters here: clean-frames-without-the-seed finds "
          f"{len(unseeded)} core(s) vs {len(starts)} with it")

    # (b) The same_fragment_identifiers clause, which splits a clean run at a
    #     fragment turnover. Compared with the seed applied to both sides, so
    #     this isolates the clause itself.
    seeded = clean.copy()
    seeded[0] = False
    naive_starts = [k for k in range(1, len(seeded)) if seeded[k] and not seeded[k - 1]]
    same = naive_starts == starts
    print(f"  {'PASS' if same else 'DIFFER'}  given the same seed, the "
          f"same_fragment_identifiers clause changes nothing here "
          f"({len(naive_starts)} vs {len(starts)} cores) -- no fragment turnover "
          f"occurs in this clip")

    reloaded = ListOfGlobalFragments.load(json_path, fragments)
    rt = len(reloaded.global_fragments) == len(logf.global_fragments)
    print(f"  {'PASS' if rt else 'FAIL'}  round-trips through upstream's own loader")

    if not (all(checks.values()) and rt):
        raise AssertionError("global fragment validation failed")

    # Global fragments existed but none survived the accumulation threshold.
    # Downstream this is indistinguishable from having none at all: tracker.py:69
    # tests `list_of_global_fragments.global_fragments`, the ACCUMULABLE ones, so
    # first_global_fragment is None and the same k-means++ path runs. The checks
    # above were real here, but the headline must not claim accumulation is on.
    if not logf.global_fragments:
        print(f"\n  {len(logf.non_accumulable_global_fragments)} global fragment(s) "
              "found, but none is long enough to accumulate on "
              f"(< {conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION} frames)")
        print("  identification falls back to k-means++\n")
        if not report_kmeans_fallback(lof, n_animals):
            raise AssertionError(
                "no accumulable global fragments AND the k-means++ fallback "
                "cannot run; see the FAIL lines above")
        print("\nNO ACCUMULABLE GLOBAL FRAGMENTS -- IDENTIFICATION WILL USE "
              f"K-MEANS++ WITH k={n_animals}   ({time.time() - t0:.0f}s)")
        return

    for g in logf.global_fragments[:3]:
        print(f"\n  core starts at frame {g.first_frame_of_the_core}, "
              f"{len(g.fragments_identifiers)} fragments, "
              f"min {g.min_n_images_per_fragment} images each, "
              f"min distance travelled {g.minimum_distance_travelled:.0f} px")

    print(f"\nALL GLOBAL FRAGMENT CHECKS PASS   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
