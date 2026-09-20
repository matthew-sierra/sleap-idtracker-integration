"""Retrain the contrastive model from scratch under a hard 30-minute budget.

A one-off experiment, not a pipeline stage. Everything it writes is new: it
never touches ``session/accumulation/`` (which holds a 24-hour checkpoint that
cannot be regenerated) and never touches ``session/id_images/``.

WHAT THIS DOES
--------------
1. Builds a ``ContrastiveLearning`` over the same fragments stage 4 uses, but
   with ``saving_folder`` pointed at ``accumulation_retrain30/`` so the
   checkpoint it saves lands somewhere disposable.
2. Trains from random init -- ``set_model()`` with NO argument -- for at most
   30 minutes, logging one CSV row per validation.
3. Reloads the best checkpoint, predicts, and runs the SAME P2 cascade as
   ``build_identities.py`` so the result is directly comparable to the
   baseline produced by the long run.
4. Writes identities into ``session/id_images_retrain30/`` and calls the
   existing stage 6 to export a .slp.

WHY THE TRAINING LOOP IS HAND-DRIVEN
------------------------------------
``ContrastiveLearning.train()`` (contrastive.py:593-706) has no time limit. It
runs until the silhouette score plateaus for ``patience`` validations or until
the target is reached, and on this clip that took a day. There is no argument,
no callback and no attribute that caps wall clock, so the only way to impose
one without editing ``idtrackerai/`` -- which stays byte-identical to upstream
-- is to drive the loop from outside using the public pieces ``train()`` itself
calls: ``train_step``, ``validate``, and ``torch.save`` to
``model_checkpoint_path``.

The loop below reproduces upstream's control flow exactly, with one addition
and one omission:

    ADDED     a wall-clock stop, checked after each validation row is recorded.
    OMITTED   the "patience exhausted -> retry KMeans with k-means++ instead of
              the global fragment" fallback (contrastive.py:673-685). That path
              only fires after 30 validations without improvement, which a
              30-minute budget cannot reach, and reproducing it would mean
              mutating ``gfrag_loader`` state the rest of this script depends on.

The improvement rule is upstream's verbatim: a BARE ``>`` against the best score
so far, with no minimum delta. A validation that beats the best by 1e-6 counts
as an improvement and resets the patience counter. That is a deliberate copy of
upstream's actual behaviour, not an oversight -- changing it would make the
curve incomparable to the baseline.

Every other hyperparameter is left at the value the existing pipeline uses:
batch_size 400, lr 0.001, 8 embedding dimensions, check_every 50,
skipped_validations 1, patience 30, target silhouette 0.91. Nothing is tuned;
the ONLY variable under test is how far 30 minutes gets you.

WHAT THE TWO LOSS FRACTIONS MEAN
--------------------------------
``train_step`` returns ``(positive_losses, negative_losses)``: the fraction of
sampled positive / negative pairs that still carried nonzero loss over the block.
The criterion pulls positive pairs (two images of the SAME fragment) to
distance 1 and pushes negative pairs (images of two COEXISTING fragments, hence
necessarily different animals) out to distance 10. So the positive fraction is
"images of one animal still too far apart" and the negative fraction is "images
of two animals still too close". Both should fall. They are the training-error
signal; silhouette is the validation signal and the thing early stopping and
checkpointing actually key on.

THE macOS DATALOADER CAVEAT
---------------------------
``build_dataloaders`` picks 1 or 3 worker processes. On Linux those are forked.
macOS spawns instead, so each worker must receive ``collate_fn`` over a pipe --
and that ``partial`` holds the entire preloaded image array. With two persistent
loaders the run wedges indefinitely at near-zero CPU. ``build_identities`` ships
the fix as ``single_process_loaders()``, which rebuilds both loaders with
``num_workers=0`` while REUSING the existing ``batch_sampler`` object (``train_step``
mutates ``batch_sampler.n_batches``, so a fresh sampler would silently break the
batch count). It is imported and called here rather than reimplemented; the
module's ``main()`` is ``__main__``-guarded, so importing it is free.

Run under ``caffeinate -s``: an earlier run lost 99.4% of its wall clock to
system sleep, which a wall-clock budget obviously cannot survive.

Run:
    export SLEAP_IDTRACKER_SESSION=.../sleap_id_mice_predictions/session
    export SLEAP_IDTRACKER_SLP=.../predictions/untracked_clip.v001.slp.predictions.slp
    caffeinate -s /opt/anaconda3/envs/sleap_id/bin/python \\
        src/sleap_idtracker/retrain30.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402  -- headless, before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import build_identities  # noqa: E402  (safe: main() is __main__-guarded)
import build_identities_silhouette  # noqa: E402
import build_sleap_tracks  # noqa: E402
from idtrackerai.base.network.device import DEVICE  # noqa: E402
from idtrackerai.base.tracker.contrastive import ContrastiveLearning  # noqa: E402
from idtrackerai.list_of_fragments import ListOfFragments  # noqa: E402
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments  # noqa: E402
from idtrackerai.utils import conf  # noqa: E402

# --- Paths. ALL NEW. Nothing here overwrites an existing artefact. ----------
RUN_DIR = config.SESSION_DIR.parent / "retrain30"
# NOT session/accumulation/ -- that holds the irreplaceable 24-hour checkpoint.
# ContrastiveLearning.train/save write to `saving_folder / checkpoint_filename`,
# so redirecting the folder is what keeps the old checkpoint safe.
ACCUM_DIR = config.SESSION_DIR.parent / "accumulation_retrain30"
ID_IMAGES_OUT = config.SESSION_DIR / "id_images_retrain30"
CSV_PATH = RUN_DIR / "training_curve.csv"
PNG_PATH = RUN_DIR / "training_curve.png"
SLP_DEST = config.SLP.parent / "untracked_clip.v001.slp.tracked_predictions.RETRAIN30.slp"

# --- Training budget and hyperparameters -----------------------------------
TIME_BUDGET_S = 30 * 60
CHECK_EVERY = build_identities.TRAIN_CHECK_EVERY              # 50
SKIPPED_VALIDATIONS = build_identities.TRAIN_SKIPPED_VALIDATIONS  # 1
PATIENCE = build_identities.TRAIN_PATIENCE                    # 30
TARGET_SILHOUETTE = build_identities.TRAIN_TARGET_SILHOUETTE  # 0.91

CSV_COLUMNS = ["batch", "wall_seconds", "silhouette", "best_silhouette",
               "positive_loss_frac", "negative_loss_frac",
               "steps_without_improvement", "is_new_best"]


def train_with_budget(cl: ContrastiveLearning) -> tuple[float, list[dict], str, int]:
    """Upstream's loop (contrastive.py:593-706) with a wall-clock stop.

    Returns ``(best_score, rows, stop_reason, batch_counter)`` and leaves the
    best weights loaded into ``cl.model``, as upstream's ``train()`` does.
    """
    rows: list[dict] = []
    best_score = 0.0
    steps_without_improvement = 0
    batch_counter = 0
    stop_reason = ""
    t0 = time.time()

    fh = CSV_PATH.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    fh.flush()

    try:
        while True:
            positive_losses, negative_losses = cl.train_step(
                n_batches=CHECK_EVERY, starting_batch_number=batch_counter)
            batch_counter += CHECK_EVERY

            if batch_counter < SKIPPED_VALIDATIONS * CHECK_EVERY:
                continue

            silhouette = cl.validate()

            # Upstream's rule verbatim: a bare `>`, no minimum delta.
            is_new_best = silhouette > best_score
            if is_new_best:
                best_score = silhouette
                torch.save(cl.model.state_dict(), cl.model_checkpoint_path)
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            elapsed = time.time() - t0
            row = {
                "batch": batch_counter,
                "wall_seconds": round(elapsed, 2),
                "silhouette": f"{silhouette:.6f}",
                "best_silhouette": f"{best_score:.6f}",
                "positive_loss_frac": f"{positive_losses:.6f}",
                "negative_loss_frac": f"{negative_losses:.6f}",
                "steps_without_improvement": steps_without_improvement,
                "is_new_best": int(is_new_best),
            }
            rows.append(row)
            writer.writerow(row)
            fh.flush()  # partial results must survive an interruption
            print(f"  batch {batch_counter:6d} | {elapsed:7.1f}s | "
                  f"silhouette {silhouette:.4f}{'  *NEW BEST*' if is_new_best else ''}"
                  f"  | best {best_score:.4f} | +pairs too far {positive_losses:6.2%}"
                  f" | -pairs too close {negative_losses:6.2%} | "
                  f"no-improve {steps_without_improvement}", flush=True)

            if elapsed >= TIME_BUDGET_S:
                stop_reason = (f"wall-clock budget reached "
                               f"({elapsed:.0f}s >= {TIME_BUDGET_S}s)")
                break
            if steps_without_improvement > PATIENCE:
                stop_reason = (f"no improvement for {steps_without_improvement} "
                               f"validations (patience={PATIENCE})")
                break
            if best_score > TARGET_SILHOUETTE and steps_without_improvement > 1:
                stop_reason = (f"target silhouette {TARGET_SILHOUETTE} reached "
                               "and 2 validations without improvement")
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted by user"
    finally:
        fh.close()

    # Upstream reloads the best weights before predicting.
    if cl.model_checkpoint_path.is_file():
        cl.model.load_state_dict(
            torch.load(cl.model_checkpoint_path, weights_only=True))
        print(f"  reloaded best checkpoint (silhouette {best_score:.6f}) from "
              f"{cl.model_checkpoint_path}")
    return best_score, rows, stop_reason, batch_counter


def plot_curve(rows: list[dict]) -> None:
    """Silhouette on the left axis, both loss fractions on the right."""
    if not rows:
        print("  no validation rows -- skipping plot")
        return
    batch = [int(r["batch"]) for r in rows]
    sil = [float(r["silhouette"]) for r in rows]
    pos = [float(r["positive_loss_frac"]) for r in rows]
    neg = [float(r["negative_loss_frac"]) for r in rows]
    best_x = [int(r["batch"]) for r in rows if int(r["is_new_best"])]
    best_y = [float(r["silhouette"]) for r in rows if int(r["is_new_best"])]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(batch, sil, color="#1f77b4", lw=1.8, label="silhouette (validation)")
    ax.scatter(best_x, best_y, s=34, facecolors="none", edgecolors="#d62728",
               lw=1.4, zorder=5, label="new best (checkpoint saved)")
    ax.set_xlabel("training batch")
    ax.set_ylabel("silhouette score", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(batch, pos, color="#2ca02c", lw=1.2, ls="--",
             label="positive pairs still too far apart")
    ax2.plot(batch, neg, color="#ff7f0e", lw=1.2, ls=":",
             label="negative pairs still too close")
    ax2.set_ylabel("fraction of sampled pairs carrying loss")

    lines, labels = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lab2, loc="center right", fontsize=8)
    ax.set_title("Contrastive retrain from scratch, 30-minute budget")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"  wrote {PNG_PATH}")


def main() -> None:
    t_start = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ACCUM_DIR.mkdir(parents=True, exist_ok=True)

    conf.set_parameters(
        MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION=(
            config.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION),
    )

    lof = ListOfFragments.load(config.FRAGMENTS_JSON)
    logf = ListOfGlobalFragments.load(config.GLOBAL_FRAGMENTS_JSON, lof.fragments)
    n_ind = sum(f.is_an_individual for f in lof.fragments)
    print(f"{len(lof.fragments)} fragments: {n_ind} individual, "
          f"{len(lof.fragments) - n_ind} crossing")
    print(f"n_animals={lof.n_animals}, device={DEVICE}")

    first_gfrag = (
        max(logf.global_fragments, key=lambda g: g.minimum_distance_travelled)
        if logf.global_fragments else None
    )
    print(f"first global fragment: "
          f"{'none -> k-means++' if first_gfrag is None else f'starts at frame {first_gfrag.first_frame_of_the_core}'}")

    print(f"\n=== training from scratch, budget {TIME_BUDGET_S}s ===")
    print(f"  saving_folder={ACCUM_DIR}  (NOT session/accumulation/)")
    cl = ContrastiveLearning(lof, saving_folder=ACCUM_DIR, first_gfrag=first_gfrag)
    # MUST happen before any training on macOS. See module docstring.
    build_identities.single_process_loaders(cl)
    print(f"  loaders rebuilt with num_workers=0 "
          f"(train={cl.train_loader.num_workers}, val={cl.val_loader.num_workers})")
    cl.set_model()  # no argument -> random init, existing checkpoint untouched
    print(f"  random init; batch_size={cl.batch_size}, lr={cl.learning_rate}, "
          f"embedding_dimensions={cl.embedding_dimensions}")
    print(f"  check_every={CHECK_EVERY}, skipped_validations={SKIPPED_VALIDATIONS}, "
          f"patience={PATIENCE}, target={TARGET_SILHOUETTE}")

    best_score, rows, stop_reason, total_batches = train_with_budget(cl)
    train_seconds = time.time() - t_start
    print(f"\n  STOP: {stop_reason}")
    print(f"  best silhouette {best_score:.6f} over {len(rows)} validations, "
          f"{total_batches} batches")
    plot_curve(rows)

    print("\n=== prediction ===")
    print("  reference_gfrag=None -> cluster numbering arbitrary but consistent")
    cl.predict(lof, None)
    identifier = cl.get_identification_model()
    print(f"  cluster_centers: {tuple(identifier.cluster_centers.shape)}")

    p1_only = build_identities.individual_identities(lof)
    n_seeded = build_identities.seed_missing_P1(lof)
    if n_seeded:
        print(f"  {n_seeded} individual fragment(s) too short for predict() "
              "-> zero P1, will come out ambiguous")
    per_fragment, p2_stats = build_identities.assign_p2_identities(lof)
    print(f"  P2 cascade: {p2_stats['n_assigned']} fragments assigned, "
          f"{p2_stats['n_ambiguous']} ambiguous (identity 0), "
          f"{p2_stats['n_fixed']} fixed above FIXED_IDENTITY_THRESHOLD")
    changed = [k for k, v in per_fragment.items()
               if k in p1_only and p1_only[k] != v]
    print(f"    differs from argmax(P1) on {len(changed)}/{len(per_fragment)} fragments")
    print(f"  individual fragments identified: {len(per_fragment)}/{n_ind}")

    image_sources = cl.preload_images(lof.id_images_file_paths, None)
    per_row = build_identities.crossing_identities(identifier, lof, image_sources)
    print(f"  crossing fragments: {len(per_row)}, rows predicted individually: "
          f"{sum(len(v) for v in per_row.values())}")

    # NOT build_identities.write_identities -- that one writes in place, into
    # the pristine session/id_images/. This one copies first, then writes.
    identities, files = build_identities_silhouette.write_identities(
        lof, per_fragment, per_row, ID_IMAGES_OUT)
    assigned = identities > 0
    print(f"\nwrote identities to {len(files)} files in {ID_IMAGES_OUT}")
    print(f"  assigned {int(assigned.sum())}/{len(identities)} instances")
    used = np.unique(identities[assigned])
    print(f"  distinct identities used: {len(used)}/{lof.n_animals}")

    print("\n=== stage 6: export .slp ===")
    build_sleap_tracks.main(slp_path=config.SLP, out_dir=ID_IMAGES_OUT, dest=SLP_DEST)

    print(f"\nTOTAL WALL TIME {time.time() - t_start:.0f}s "
          f"(training {train_seconds:.0f}s)")
    print(f"STOP REASON: {stop_reason}")
    print(f"BEST SILHOUETTE: {best_score:.6f}")
    print(f"SLP: {SLP_DEST}")


if __name__ == "__main__":
    main()
