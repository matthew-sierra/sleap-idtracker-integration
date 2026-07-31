# sleap-idtracker

Runs idtracker.ai's identity assignment on SLEAP pose predictions.

SLEAP tracks well frame to frame but its identities drift across occlusions — it offers only simple
and optical-flow association. idtracker.ai is more reliable at holding an identity, but it is built
around *blobs*: it segments animals by intensity thresholding and works from contours. This port
keeps idtracker.ai's identity machinery and replaces its input stage, deriving every per-animal
image from SLEAP keypoints instead.

**No thresholding of any kind.** No intensity cutoffs, no grabCut, no pixel-value segmentation. All
geometry comes from keypoints. Crops are non-square convex silhouettes and are not rescaled; the
identity network was modified to accept them (`# SLEAP-PORT:` edits in `idtrackerai/`).

## Install

```bash
git clone <this repo> sleap-idtracker && cd sleap-idtracker
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -e .
```

`pip install -e .` resolves dependencies only — it installs no modules. The stages are run as
scripts from the checkout. idtracker.ai is bundled at `idtrackerai/` and put on `sys.path` by
`config.py`, so there is nothing else to install. See `PROVENANCE.md`.

## Point it at data

Videos and `.slp` files are not in this repo. The default data root is `./data`; override it:

```bash
export SLEAP_IDTRACKER_DATA=/path/to/your/experiment
```

Expected under that root:

```
predictions/<your>.predictions.slp     # SLEAP output    -> SLEAP_IDTRACKER_SLP
session_fourfly/                       # written by the pipeline -> SLEAP_IDTRACKER_SESSION
```

Then set the population in `config.py` — `N_ANIMALS` is the number of animals in the arena, and
`CENTROID_NODE` / `TOP_NODE` are the two skeleton nodes used to orient each crop.

## Run

In order — each stage reads what the previous one wrote:

```bash
python build_id_images.py        # Stage 3  crops -> id-image HDF5 (+ episodes, Stage 1)
python build_overlaps.py         # Stage 3c is_overlapping flag + the identity graph
python build_fragments.py        # Stage 3d Fragment objects
python build_global_fragments.py # Stage 3f ListOfGlobalFragments
python build_identities.py       # Stage 4  contrastive training -> identities
python build_sleap_tracks.py     # Stage 6  identities back into a .slp as Tracks
```

The last step writes a sibling `.idtracker_predictions.slp` next to your input, leaving SLEAP's own
file canonical. Open it in the SLEAP GUI — that is the readout.

## Layout

| Module | Role |
|---|---|
| `config.py` | every user-set knob, and the only definition of where things live |
| `episodes.py` | Stage 1 — episode segmentation |
| `build_id_images.py` | Stage 3 — keypoint crops to id-image HDF5 |
| `build_overlaps.py` | Stage 3c — overlap detection, identity graph |
| `assignment_margin.py` | similarity/margin scoring used by Stage 3c |
| `build_fragments.py` | Stage 3d — `Fragment` objects |
| `build_global_fragments.py` | Stage 3f — `ListOfGlobalFragments` |
| `build_identities.py` | Stage 4 — contrastive identity assignment |
| `build_sleap_tracks.py` | Stage 6 — identities back into a `.slp` |
| `rotate_boxes.py` | egocentric rotation helpers, imported by Stage 3 |
| `idtrackerai/` | idtracker.ai 6.0.14 with surgical edits — see `PROVENANCE.md` |

Superseded frame-0 exploratory modules (`crops.py`, `make_boxes.py`, `canonical_hulls.py`) and the
retired `build_trajectories.py` are untracked; see `.gitignore`.

## Caveats worth knowing before you trust output

- The convex hull covers only *visible* keypoints — it under-covers where nodes are missing and
  over-covers splayed legs.
- Canonical-frame sizing uses central statistics, so it clips a fraction of animals at the extremes.
- The crossing/overlap threshold is scale-sensitive; `config.py` documents the three knobs.
