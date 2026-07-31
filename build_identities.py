"""Assign an identity to every instance in the video.

Stage 4. Runs after ``build_global_fragments.py``.

WHAT THIS DOES
--------------
Two populations, handled differently, per the design decision recorded in
UPDATES.md:

    INDIVIDUAL fragments  -- the tracker was confident here. These are what the
        contrastive model trains on, and their identity is decided at FRAGMENT
        level by pooling every image's vote (upstream's P1 vector). One identity
        for the whole fragment, so a single bad frame cannot flip it.

    CROSSING fragments    -- a merged group holding every animal that was
        mutually ambiguous over some stretch (see build_fragments.py). These are
        excluded from training entirely: they contribute no images to any batch.
        At prediction time each ROW is pushed through the trained model on its
        own and assigned to its nearest cluster centre, so a crossing fragment
        ends up holding several identities because its rows do.

Deliberately NOT done, per the MVP scope: no accumulation protocol, no P2
exclusion cascade, no impossible-velocity correction, no crossing interpolation.
Those are upstream's robustness layers and each is a separate decision.

HOW IDENTITY IS READ OFF
------------------------
``ContrastiveLearning.predict`` (contrastive.py:748) already restricts itself to
``frag.is_an_individual``, so the crossing rows are skipped for free -- no filter
of ours is needed. It sets each individual fragment's ``P1_vector`` via
``set_identification_statistics`` (fragment.py:374), which is upstream's
base-2-softmax vote pooling. ``argmax(P1_vector) + 1`` is that fragment's
identity; the ``+1`` is upstream's 1-indexing, where 0 means "unassigned"
(list_of_fragments.py:372).

For crossing rows we call ``IdentifierContrastive.forward`` (models.py:260),
which is the same embedding followed by ``cdist`` to the cluster centres and the
``reciprocal(d + 0.01) ** 7`` weighting -- identical arithmetic to ``predict``,
applied one row at a time instead of pooled.

Images are preprocessed exactly as ``collate_fun`` (contrastive.py:874) does it:
``load_id_images(..., dtype=np.float32)`` then ``unsqueeze(1) / 255``. Any
divergence here would silently feed the model differently-scaled images than it
trained on.

OUTPUT
------
Writes into the existing ``identities`` dataset, which build_id_images.py created
zero-filled precisely so upstream's ``require_dataset`` finds a matching shape.
0 stays the "unassigned" sentinel.

Run:
    /opt/anaconda3/envs/sleap_id/bin/python src/sleap_idtracker/build_identities.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from idtrackerai.base.network.device import DEVICE  # noqa: E402
from idtrackerai.base.tracker.contrastive import ContrastiveLearning  # noqa: E402
from idtrackerai.list_of_fragments import ListOfFragments  # noqa: E402
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments  # noqa: E402
from idtrackerai.utils import conf  # noqa: E402
from idtrackerai.utils.py_utils import load_id_images  # noqa: E402

OUT_DIR = config.OUT_DIR
FRAGMENTS_JSON = config.FRAGMENTS_JSON
GLOBAL_FRAGMENTS_JSON = config.GLOBAL_FRAGMENTS_JSON
ACCUMULATION_DIR = config.SESSION_DIR / "accumulation"

# --- Short-training knobs --------------------------------------------------
# These exist to make the stage runnable end to end in seconds so the plumbing
# can be validated. They are NOT tracking-quality settings.
#
# The loop (contrastive.py:633) validates every `check_every` batches, starting
# after `skipped_validations` of them, and stops once the score has failed to
# improve for `patience` steps -- or, with the target set to 0.0, once any
# positive score has been reached and two steps pass without improvement. That
# last path is what makes this terminate quickly.
#
# For a real run: delete these and let train() use its own defaults
# (check_every = max(5*n_animals, 100), CONTRASTIVE_PATIENCE = 30,
# CONTRASTIVE_SILHOUETTE_TARGET = 0.91).
SHORT_TRAINING = True

# Skip training entirely when a checkpoint from a previous run is on disk.
REUSE_CHECKPOINT = True

# macOS spawns DataLoader workers instead of forking them, which deadlocks on
# the preloaded image array. See single_process_loaders() below.
SINGLE_PROCESS_LOADERS = sys.platform == "darwin"

TRAIN_CHECK_EVERY = 10
TRAIN_SKIPPED_VALIDATIONS = 1
TRAIN_PATIENCE = 2
TRAIN_TARGET_SILHOUETTE = 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def single_process_loaders(contrastive: ContrastiveLearning) -> None:
    """Rebuild both DataLoaders with ``num_workers=0``. Required on macOS.

    ``build_dataloaders`` (contrastive.py:404) picks 1 or 3 workers depending on
    whether images were preloaded. On Linux those workers are forked and inherit
    memory for free. macOS spawns instead, so every worker re-imports the module
    AND has to receive `collate_fn` over a pipe -- and `collate_fn` is
    ``partial(collate_fun, images_sources=...)`` holding the whole preloaded
    image array (151 MB here). With two persistent loaders that stalls the run
    indefinitely at near-zero CPU, which is exactly what it did.

    num_workers=0 loads batches in the main process: no spawn, no pickling, no
    stall. The images are already in RAM, so there is nothing to overlap with
    anyway and the workers were buying nothing at this size.

    Done here rather than as an edit to idtrackerai/ -- the upstream default is
    right for the platform it targets, and this is a port-side platform fix.
    The batch_sampler object is REUSED, not rebuilt: ``train_step`` mutates
    ``train_loader.batch_sampler.n_batches`` on every call (contrastive.py:538),
    so a fresh sampler would silently break the batch count.
    """
    tl, vl = contrastive.train_loader, contrastive.val_loader
    contrastive.train_loader = DataLoader(
        dataset=tl.dataset, batch_sampler=tl.batch_sampler,
        collate_fn=tl.collate_fn, num_workers=0)
    contrastive.val_loader = DataLoader(
        dataset=vl.dataset, batch_size=vl.batch_size,
        collate_fn=vl.collate_fn, num_workers=0)


def train(lof: ListOfFragments, first_gfrag) -> tuple[ContrastiveLearning, float]:
    """Train the contrastive model on individual fragments only."""
    ACCUMULATION_DIR.mkdir(parents=True, exist_ok=True)
    contrastive = ContrastiveLearning(
        lof, saving_folder=ACCUMULATION_DIR, first_gfrag=first_gfrag
    )
    if SINGLE_PROCESS_LOADERS:
        single_process_loaders(contrastive)

    # Training is the expensive step (~15 min on MPS for this clip, because it
    # runs until the silhouette score plateaus rather than for a fixed budget).
    # `train()` leaves its best weights in contrastive_checkpoint.pt, and
    # set_model() will pick that file up when handed the folder
    # (contrastive.py:499). Reusing it makes downstream iteration cheap.
    # Delete the checkpoint, or set REUSE_CHECKPOINT=False, to retrain.
    checkpoint = ACCUMULATION_DIR / contrastive.checkpoint_filename
    if REUSE_CHECKPOINT and checkpoint.is_file():
        print(f"  reusing weights from {checkpoint.name} (skipping training)")
        contrastive.set_model(ACCUMULATION_DIR)
        return contrastive, float("nan")

    contrastive.set_model()

    if SHORT_TRAINING:
        print(f"  SHORT TRAINING: check_every={TRAIN_CHECK_EVERY}, "
              f"patience={TRAIN_PATIENCE}, target={TRAIN_TARGET_SILHOUETTE}")
        score = contrastive.train(
            check_every=TRAIN_CHECK_EVERY,
            skipped_validations=TRAIN_SKIPPED_VALIDATIONS,
            patience=TRAIN_PATIENCE,
            target_silhouette_score=TRAIN_TARGET_SILHOUETTE,
        )
    else:
        score = contrastive.train()
    return contrastive, score


# ---------------------------------------------------------------------------
# Reading identities off the two populations
# ---------------------------------------------------------------------------


def seed_missing_P1(lof: ListOfFragments) -> int:
    """Give every individual fragment a P1_vector, so P2 can be computed.

    ``predict`` only sets P1 on fragments with at least
    MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION images (contrastive.py:769);
    shorter ones are left with the attribute unset, and ``compute_P2_vector``
    would raise on them -- both for the fragment itself and for every fragment
    that coexists with it, since it reads their P1 vectors too.

    Upstream never hits this because assign_remaining_fragments predicts on
    *all* non-accumulated individual fragments with no length filter
    (assigner.py:145). This port reuses contrastive.predict instead, which does
    filter, so the gap has to be closed here.

    Zeros are the right filler rather than a placeholder: a zero P1 contributes
    (1 - 0) = 1 to every coexisting product, i.e. excludes nothing, which is
    correct for a fragment carrying no evidence. Its own P2 numerator is then 0,
    the denominator is 0, and compute_P2_vector returns a zero vector -- which
    assign_identity reads as an n-way tie and answers with identity 0. A
    fragment we never predicted on ends up explicitly unassigned, which is the
    honest outcome.
    """
    n = 0
    for frag in lof.fragments:
        if not frag.is_an_individual:
            continue
        if getattr(frag, "P1_vector", None) is None:
            frag.P1_vector = np.zeros(lof.n_animals)
            n += 1
    return n


def assign_p2_identities(lof: ListOfFragments) -> tuple[dict[int, int], dict]:
    """fragment_identifier -> identity, via upstream's P2 cascade.

    This is assigner.py:157-167 with one step dropped: upstream re-runs the
    network over every non-accumulated fragment first, because in its flow the
    fragments have not been predicted on yet. Ours have -- contrastive.predict
    already called set_identification_statistics, which is the same P1 the
    cascade would compute. So the prediction is reused and the assignment half
    is run verbatim, using upstream's own methods rather than a reimplementation
    of the arithmetic.

    The ordering matters and is upstream's: get_fragments_to_identify
    (list_of_fragments.py:344) repeatedly yields the unassigned fragment with
    the highest P2 certainty, so the most confident calls are made first. Note
    that P2 is computed ONCE before the loop, not recomputed after each
    assignment -- the coexistence exclusion is baked into the vectors up front.
    The trailing compute_P2_vectors() refreshes them to reflect the one-hot P1
    vectors assign_identity leaves behind.
    """
    lof.compute_P2_vectors()
    for frag in lof.get_fragments_to_identify():
        frag.assign_identity(lof.n_animals, lof.id_to_exclusive_roi)
    lof.compute_P2_vectors()

    out: dict[int, int] = {}
    ambiguous: list[int] = []
    for frag in lof.fragments:
        if not frag.is_an_individual or frag.identity is None:
            continue
        out[frag.identifier] = int(frag.identity)
        if frag.identity == 0:
            ambiguous.append(frag.identifier)

    stats = {
        "n_assigned": sum(1 for v in out.values() if v > 0),
        "n_ambiguous": len(ambiguous),
        "ambiguous_fragments": ambiguous,
        "n_fixed": sum(1 for f in lof.fragments
                       if f.is_an_individual and f.identity_is_fixed),
    }
    return out, stats


def individual_identities(lof: ListOfFragments) -> dict[int, int]:
    """fragment_identifier -> identity, pooled over the fragment (P1 argmax)."""
    out = {}
    for frag in lof.fragments:
        if not frag.is_an_individual:
            continue
        if getattr(frag, "P1_vector", None) is None or not np.any(frag.P1_vector):
            # too short for predict()'s MIN_N_FRAMES filter -> stays unassigned
            continue
        out[frag.identifier] = int(np.argmax(frag.P1_vector)) + 1
    return out


@torch.inference_mode()
def crossing_identities(
    identifier, lof: ListOfFragments, image_sources, batch: int = 256
) -> dict[int, list[int]]:
    """fragment_identifier -> per-ROW identity, for merged crossing fragments.

    One forward pass per row, no pooling: the rows of a crossing fragment are
    different animals, so pooling them would be exactly wrong.
    """
    identifier.to(DEVICE)
    identifier.eval()
    out: dict[int, list[int]] = {}
    for frag in lof.fragments:
        if frag.is_an_individual:
            continue
        locations = list(frag.image_locations)
        preds: list[int] = []
        for i in range(0, len(locations), batch):
            chunk = locations[i: i + batch]
            imgs = load_id_images(image_sources, chunk, verbose=False, dtype=np.float32)
            # identical preprocessing to collate_fun (contrastive.py:885)
            tensor = torch.from_numpy(imgs).unsqueeze(1).to(DEVICE) / 255
            prob = identifier(tensor)
            preds += (prob.argmax(1) + 1).tolist()
        out[frag.identifier] = preds
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_identities(lof: ListOfFragments, per_fragment: dict[int, int],
                     per_row: dict[int, list[int]]) -> np.ndarray:
    """Fill the `identities` dataset. Returns the concatenated identity array."""
    files = [Path(p) for p in lof.id_images_file_paths]
    sizes = {}
    for e, path in enumerate(files):
        with h5py.File(path, "r") as fh:
            sizes[e] = len(fh["identities"])
    buf = {e: np.zeros(n, dtype=np.int64) for e, n in sizes.items()}

    for frag in lof.fragments:
        locs = list(frag.image_locations)
        if frag.is_an_individual:
            ident = per_fragment.get(frag.identifier)
            if ident is None:
                continue
            for img, ep in locs:
                buf[int(ep)][int(img)] = ident
        else:
            preds = per_row.get(frag.identifier, [])
            for (img, ep), ident in zip(locs, preds):
                buf[int(ep)][int(img)] = int(ident)

    for e, path in enumerate(files):
        with h5py.File(path, "r+") as fh:
            fh["identities"][...] = buf[e]
            fh.attrs["identities_source"] = "build_identities.py"
    return np.concatenate([buf[e] for e in range(len(files))])


def main() -> None:
    t0 = time.time()
    conf.set_parameters(
        MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION=(
            config.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION),
    )

    lof = ListOfFragments.load(FRAGMENTS_JSON)
    logf = ListOfGlobalFragments.load(GLOBAL_FRAGMENTS_JSON, lof.fragments)
    n_ind = sum(f.is_an_individual for f in lof.fragments)
    n_cross = len(lof.fragments) - n_ind
    print(f"{len(lof.fragments)} fragments: {n_ind} individual, {n_cross} crossing")
    print(f"n_animals={lof.n_animals}, device={DEVICE}")

    first_gfrag = (
        max(logf.global_fragments, key=lambda g: g.minimum_distance_travelled)
        if logf.global_fragments else None
    )
    print(f"first global fragment: "
          f"{'none -> k-means++' if first_gfrag is None else f'starts at frame {first_gfrag.first_frame_of_the_core}'}")

    print("\n=== training (individual fragments only) ===")
    contrastive, silhouette = train(lof, first_gfrag)
    print(f"  silhouette score: {silhouette:.4f}")

    print("\n=== prediction ===")
    # reference_gfrag=None ON PURPOSE -- the default, reorder-free path.
    #
    # Passing the global fragment here would make predict() renumber the k-means
    # clusters onto a "canonical" identity ordering via Hungarian matching
    # (contrastive.py:811). That anchor only carries meaning when it comes from
    # somewhere real -- a knowledge-transfer folder holding identities from a
    # previously tracked session, or exclusive ROIs. With neither, upstream falls
    # back to `identities = np.arange(n_animals)` (identity_transfer.py:42),
    # i.e. "identity k = the k-th fragment in the first global fragment", which
    # recovers no information and merely picks a convention.
    #
    # It also requires every fragment of that global fragment to carry a
    # `temporary_id`, which is set by the accumulation-protocol step this port
    # deliberately skips -- so passing it raises `assert frag.temporary_id is not
    # None`.
    #
    # Skipping the reorder leaves identities arbitrary but internally consistent:
    # fly "3" is the same fly throughout the video, just not tied to anything
    # outside it. That is the correct guarantee for an unsupervised MVP. Revisit
    # when identities need to be stable ACROSS videos -- this is the hook.
    print("  reference_gfrag=None -> cluster numbering is arbitrary but consistent")
    contrastive.predict(lof, None)
    identifier = contrastive.get_identification_model()
    print(f"  cluster_centers: {tuple(identifier.cluster_centers.shape)}")

    if config.USE_P2_ASSIGNMENT:
        # Snapshot the coexistence-free answer BEFORE the cascade runs.
        # assign_identity collapses P1_vector to a one-hot (fragment.py:449), so
        # after the cascade argmax(P1) just echoes the assignment and the
        # comparison would be vacuous.
        p1_only = individual_identities(lof)
        n_seeded = seed_missing_P1(lof)
        if n_seeded:
            print(f"  {n_seeded} individual fragment(s) too short for predict() "
                  "-> zero P1, will come out ambiguous")
        per_fragment, p2_stats = assign_p2_identities(lof)
        print(f"  P2 cascade: {p2_stats['n_assigned']} fragments assigned, "
              f"{p2_stats['n_ambiguous']} ambiguous (identity 0), "
              f"{p2_stats['n_fixed']} fixed above FIXED_IDENTITY_THRESHOLD")
        if p2_stats["ambiguous_fragments"]:
            print(f"    ambiguous fragment ids: {p2_stats['ambiguous_fragments']}")
        # What P2 actually bought, measured rather than asserted.
        changed = [k for k, v in per_fragment.items()
                   if k in p1_only and p1_only[k] != v]
        print(f"    differs from argmax(P1) on {len(changed)}/{len(per_fragment)} "
              f"fragments{': ' + str(changed) if changed else ''}")
    else:
        per_fragment = individual_identities(lof)
    print(f"  individual fragments identified: {len(per_fragment)}/{n_ind}")

    image_sources = contrastive.preload_images(lof.id_images_file_paths, None)
    per_row = crossing_identities(identifier, lof, image_sources)
    n_rows = sum(len(v) for v in per_row.values())
    print(f"  crossing fragments: {len(per_row)}, rows predicted individually: {n_rows}")
    for fid, preds in per_row.items():
        frag = lof.fragments[fid]
        print(f"    fragment {fid}: frames {frag.start_frame}-{frag.end_frame - 1}, "
              f"{len(preds)} rows -> identities {preds}")

    identities = write_identities(lof, per_fragment, per_row)
    print(f"\nwrote identities to {len(lof.id_images_file_paths)} files")

    # --- validation -------------------------------------------------------
    print("\n=== validation ===")
    assigned = identities > 0
    checks = {
        f"identities within 1..{lof.n_animals}": bool(
            (identities[assigned] >= 1).all()
            and (identities[assigned] <= lof.n_animals).all()),
        "crossing rows got per-row identities": all(
            len(v) == lof.fragments[k].n_images for k, v in per_row.items()),
    }
    # "every instance has an identity" is a HARD check only without P2. Under
    # P2 an n-way tie on max P2 yields identity 0 by design (fragment.py:431) --
    # a refusal, not a plumbing failure -- so failing the stage on it would
    # forbid the very behaviour that was just asked for. Reported below instead.
    if not config.USE_P2_ASSIGNMENT:
        checks["every instance has an identity"] = bool(assigned.all())
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"  assigned {int(assigned.sum())}/{len(identities)} instances")
    if config.USE_P2_ASSIGNMENT and not assigned.all():
        n_un = int((~assigned).sum())
        print(f"  {n_un} instances left at identity 0 by the P2 cascade "
              f"({100 * n_un / len(identities):.2f}%) -- ambiguous, not failed")

    # Reported, NOT enforced. With SHORT_TRAINING the embedding is deliberately
    # undertrained, so k-means can leave a cluster unclaimed and two fragments
    # can land on one identity. That is a quality signal, not a plumbing failure,
    # and failing the stage on it would hide whether the pipeline itself works.
    used = np.unique(identities[assigned])
    print(f"\n  quality (not enforced -- SHORT_TRAINING={SHORT_TRAINING}):")
    print(f"    distinct identities used: {len(used)}/{lof.n_animals}")
    if len(used) < lof.n_animals:
        missing = sorted(set(range(1, lof.n_animals + 1)) - set(used.tolist()))
        print(f"    identities never assigned: {missing}")
    counts = {int(i): int((identities == i).sum()) for i in used}
    print(f"    rows per identity: min {min(counts.values())}, "
          f"max {max(counts.values())} (even split would be {len(identities)//lof.n_animals})")

    if not all(checks.values()):
        raise AssertionError("identity validation failed")
    print(f"\nALL IDENTITY CHECKS PASS   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
