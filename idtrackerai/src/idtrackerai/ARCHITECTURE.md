# idtracker.ai — Architecture Guide

This document is a map of the `idtrackerai` Python package: what the system does, how the
pieces fit together, and what each file is for. It was written to support a project that
adapts idtracker.ai's identification approach for use inside SLEAP's tracker, so it leans
toward explaining the **algorithm and its data flow**, not just listing files.

---

## 1. What idtracker.ai does

idtracker.ai is a multi-animal tracking system. Given a video of `N` unmarked animals, it
outputs, for every frame, the (x, y) position of each animal **and a persistent numeric
identity (1..N)** that stays correct across the whole video — even through occlusions,
animals crossing paths, and animals leaving/re-entering the frame. It does this with no
markers/tags, using classical computer vision for detection and a convolutional neural
network for re-identification, trained *on the video itself* (no pretrained weights
required, though transfer learning is supported). In the current 6.x line, that
re-identification network is **primarily a self-supervised contrastive embedding**
(a ResNet-18), with the older supervised `N`-way identity-classifier CNN (`IdCNN`) retained
only as a fallback — see step 4 below.

The system does **not** track by simple frame-to-frame proximity/motion (a "simple flow"
tracker). Instead it:

1. Detects all animal blobs in every frame independently (segmentation).
2. Links blobs across consecutive frames purely by pixel overlap into short reliable chains
   called **fragments** (a fragment is *not yet* identified — it's just "this is probably one
   consistent animal for these N frames").
3. Finds moments where all `N` animals are simultaneously visible and separated
   (**global fragments**) — the single richest one is used to seed a bootstrap identity
   assignment (0..N-1), and, if a prior model is supplied, to run identity transfer against it.
4. Trains a **contrastive embedding** (ResNet-18), pulling together images from the same
   fragment and pushing apart images from any two simultaneously-visible fragments (which must
   be different individuals) — this training signal comes from ordinary fragment coexistence,
   not from full `N`-way global fragments, which are only needed for the bootstrap seed above.
   Clusters the embedding with K-means and aligns cluster labels to the bootstrap identities.
   This **contrastive protocol is the primary identification method** (the 2025 "representation
   learning" reframing of idtracker.ai). Only if it doesn't confidently account for enough of the
   video does the system fall back to the older supervised approach: bootstrap a CNN identity
   classifier from the global fragments, then **iteratively grow the training set** by accepting
   increasingly-less-obvious fragments the network classifies with high, self-consistent
   confidence — the "cascade of training and identification protocols" the earlier idtracker.ai
   papers are known for.
5. Uses the final trained identifier (contrastive+K-means, or the supervised CNN if it ran) to
   identify every remaining fragment, resolves physically-impossible identity jumps, interpolates
   through crossings/occlusions, and writes out per-frame trajectories.

### The pipeline, stage by stage

The whole run is orchestrated by `base/run.py`'s `RunIdTrackerAi.track_video()`, which calls
five stage APIs in sequence (each stage's output is checkpointed to disk so a run can be
inspected/resumed):

| # | Stage | Entry point | Input → Output |
|---|-------|-------------|-----------------|
| 1 | **Animal detection** (segmentation) | `animals_detection_API(session)` | video frames → `ListOfBlobs` |
| 2 | **Crossing detection** | `crossings_detection_API(session, list_of_blobs)` | `ListOfBlobs` → same, with `Blob.is_an_individual` classified |
| 3 | **Fragmentation** | `fragmentation_API(session, list_of_blobs)` | `ListOfBlobs` → `ListOfFragments`, `ListOfGlobalFragments` |
| 4 | **Tracking** (identification) | `tracker_API(session, list_of_blobs, list_of_fragments, list_of_global_fragments)` | fragments → per-fragment `identity`, trained `identifier_model` |
| 5 | **Trajectories** (post-processing + export) | `trajectories_API(session, list_of_blobs, list_of_fragments, identifier_model)` | identified fragments → trajectory files on disk |

### The core data model

```
Blob            one animal-or-crossing contour, in one frame
  .next / .previous     tuples of Blobs in adjacent frames that overlap in pixels
  .fragment_identifier  which Fragment this blob belongs to

Fragment        a maximal run of Blobs linked 1-to-1 (no merges/splits) across frames
                — either one individual animal, or one crossing (2+ touching animals),
                for as long as the linkage stays unambiguous
  .P1_vector, .P2_vector, .identity, .temporary_id, .certainty, .used_for_training, ...

GlobalFragment  a set of N (one per animal) coexisting *individual* Fragments, taken at a
                frame where all N animals are simultaneously visible and separated
                — the highest-confidence "everyone is alone" moments used to bootstrap
                and grow the identification training set

Session         per-run configuration + all state generated during tracking (video paths,
                thresholds, computed sizes, timers, results); saved/loaded as session.json
```

`ListOfBlobs`, `ListOfFragments`, and `ListOfGlobalFragments` are thin container/orchestration
classes around lists of the above (they own the save/load logic and the batch operations —
e.g. "connect all overlapping blobs", "compute P2 for every fragment" — that are too
expensive or too structural to live on a single instance).

### The identification cascade, in a bit more depth

This is the part most relevant to "how the model works":

1. **Bootstrap.** Pick the `GlobalFragment` whose animals moved the most (most distinguishable
   motion, least likely to be a segmentation artifact). Seed its `N` fragments with identities
   0..N-1 — either arbitrarily, or, if a previously-trained model is supplied
   (`knowledge_transfer_folder`), by **identity transfer**: run the old model on this
   bootstrap fragment and solve an optimal bipartite matching (Hungarian algorithm) between
   "this video's fragments" and "the old model's identity classes."
2. **Contrastive step (unsupervised, tried first).** Train a ResNet-18 embedding with a
   Siamese/margin contrastive loss: images from the *same* fragment are pulled together,
   images from *different, simultaneously-visible* fragments (which therefore **must** be
   different individuals) are pushed apart. Cluster the embedding space with K-means and align
   cluster labels to the bootstrap identities. If this alone confidently accounts for enough of
   the video, tracking is done — no supervised CNN needed at all.
3. **Supervised cascade (if contrastive isn't enough).** Train a small CNN classifier
   (`IdCNN`, `N`-way softmax) on the currently-accepted fragments. Predict on the rest.
   Convert each fragment's per-image predictions into a **P1 vector** (a base-2 softmax over
   *vote counts*, so confidence scales with how many images agreed, not just average
   probability). Greedily accept new fragments/global-fragments whose P1 is confident and
   internally consistent (no two simultaneously-visible fragments claiming the same identity),
   first trying an all-or-nothing "global" strategy per global fragment, falling back to a
   per-fragment "partial" strategy. Retrain and repeat until nothing new can be confidently
   accepted.
4. **Residual identification.** For every fragment never used in training, predict with the
   final model and compute a **P2 vector** — P1 reweighted by
   `∏(1 − coexisting_fragment.P1)` across identities, i.e. a soft Bayesian penalty for any
   identity that a simultaneously-visible fragment also claims. Resolve fragments
   most-confident-first, propagating each resolution into its neighbors' P2 immediately.
5. **Post-processing.** Fix physically-impossible velocity jumps at fragment boundaries;
   interpolate through crossings by iteratively eroding merged blobs into separable pieces and
   filling identity/position gaps from both ends inward; export final trajectories.

---

## 2. Background topics worth learning

To follow (or reimplement pieces of) this architecture, these are the concepts that come up
repeatedly:

**Computer vision / classical CV**
- Image thresholding & background subtraction (foreground/blob segmentation)
- Contour detection, connected components, and morphological operations (dilation/erosion) — OpenCV
- Bounding boxes, image moments (centroid/orientation via `cv2.moments`)
- Multi-object tracking-by-detection and frame-to-frame association by overlap ("tracklet"/fragment construction)

**Deep learning**
- CNN image classification basics (conv/pool/FC stacks, softmax, cross-entropy loss)
- Class-imbalanced training (inverse-frequency loss weighting)
- Early stopping / overfitting detection, learning-rate scheduling
- Contrastive / metric learning and Siamese networks (contrastive & triplet loss, embedding spaces, hard-negative mining)
- Transfer learning / weight reuse across models ("knowledge transfer" in this codebase)
- PyTorch fundamentals: `nn.Module`, `Dataset`/`DataLoader`, optimizers, schedulers, `torch.compile`

**Unsupervised learning & assignment**
- K-means clustering and the silhouette score (cluster-quality metric)
- The assignment problem / Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) — used both for identity transfer and for aligning unsupervised cluster labels to known identities

**Probability & constraint reasoning**
- Softmax and "confidence" calibration relative to sample size (the P1 vote-count softmax)
- Simple constraint propagation / soft Bayesian reweighting for mutual-exclusion constraints (the P2 vector, encoding "two simultaneously-visible animals can't share an identity")

**Systems/engineering patterns used throughout**
- Video I/O and frame-range partitioning for parallelism (`Episode`s, `multiprocessing.Pool`)
- HDF5/JSON/pickle for scientific data serialization, and version-compatibility shims across file formats
- Config/state objects with declarative attribute validation (`Session.set_parameters`, `ConfParams`)
- Qt (`qtpy`, PyQt/PySide) desktop GUI patterns, if the GUI layer is of interest

---

## 3. Repository layout

```
idtrackerai/
├── __init__.py, session.py, blob.py, fragment.py, globalfragment.py,
│   list_of_blobs.py, list_of_fragments.py, list_of_global_fragments.py   # core data model
├── base/                     # the tracking algorithm itself
│   ├── run.py                # top-level orchestrator (RunIdTrackerAi)
│   ├── animals_detection/    # stage 1: segmentation → Blobs
│   ├── crossings_detection/  # stage 2: individual vs. crossing CNN
│   ├── fragmentation/        # stage 3: Blobs → Fragments → GlobalFragments
│   ├── network/              # shared CNN architectures + generic training loop
│   ├── tracker/              # stage 4: the identification cascade
│   └── postprocess/          # stage 5: corrections + trajectory export
├── utils/                    # config, logging, I/O, misc helpers used everywhere
├── start/                    # CLI entry point (`idtrackerai` command) + arg parsing
├── GUI_tools/                # shared Qt widget framework used by all GUI apps
├── segmentation_app/         # GUI for setting up segmentation params before tracking
├── extra_tools/              # validator GUI, video generator, idmatcher.ai, cluster inspection
└── data/                     # bundled sample videos used by `idtrackerai_test`
```

---

## 4. File-by-file reference

### 4.1 Core data model (package root)

- **`__init__.py`** — Package entry point. Sets the multiprocessing start method, works
  around a PyQt6/OpenCV import-order quirk, and exposes the public API: `Session`, `Blob`,
  `Fragment`, `GlobalFragment`, `ListOfBlobs`, `ListOfFragments`, `ListOfGlobalFragments`,
  `IdtrackeraiError`, `conf`.

- **`session.py`** — `Session`: the central config/state object for one tracking run. Holds
  every user-facing parameter (video paths, thresholds, number of animals, output options),
  computed properties (video width/height/fps, processing `episodes` for parallelism,
  `id_image_size`), and paths to every artifact the pipeline produces (`blobs_path`,
  `fragments_path`, `id_images_folder`, etc., all derived from `session_folder`). Provides
  `prepare_tracking()` (validates and finalizes all parameters before a run starts),
  `save()`/`load()` (JSON, with extensive backward-compatibility handling for older
  idtracker.ai versions), and `delete_data()`/`compress_data()` (post-run cleanup governed by
  `data_policy`). Every other module reads configuration from a `Session` instance and many
  write results back onto it (e.g. `estimated_accuracy`, `silhouette_score`,
  `median_body_length`).

- **`blob.py`** — `Blob`: one segmented contour in one frame (an animal, or several touching
  animals = a "crossing"). Computes geometric properties from its contour (`area`, `centroid`,
  `orientation`, `bbox_corners`, all via OpenCV, mostly `cached_property`s), `overlaps_with()`
  another blob (pixel/contour intersection test used to link blobs across frames via
  `now_points_to()`), and `get_image_for_identification()` (produces the fixed-size,
  orientation-normalized, masked grayscale square image fed to every CNN in the codebase —
  crossing detector, identity classifier, contrastive network). Cached topological properties
  (`has_multiple_next/previous`, `has_a_next/previous_crossing`) walk the `.next`/`.previous`
  chain to detect nearby branch points, and back the heuristics `is_a_sure_individual()` /
  `is_a_sure_crossing()` used to bootstrap crossing-detector training labels. Also carries all
  the post-processing/validation fields (`identity`, `identity_corrected_solving_jumps`,
  `identities_corrected_closing_gaps`, `interpolated_centroids`, `user_generated_*`) and the
  `final_identities`/`final_centroids` properties that reconcile algorithmic output with manual
  corrections made in the Validator GUI.

- **`fragment.py`** — `Fragment`: a maximal run of blobs linked 1-to-1 across frames (see
  §1). Constructed from a list of image indices/centroids/episodes; computes
  `frame_by_frame_velocity` and start/end position immediately. Owns essentially the entire
  identification-cascade vocabulary: `P1_vector`/`set_P1_from_frequencies()`, `P2_vector`/
  `compute_P2_vector()`, `certainty`/`certainty_P2`, `temporary_id`, `identity`,
  `assign_identity()` (residual identification), `coexisting_individual_fragments`,
  `coexist_with()`, `is_inconsistent_with_coexistent_fragments()`,
  `compute_border_velocity()`/`get_neighbour_fragment()` (used by impossible-jump correction),
  and `reset()` (rolls a fragment back to an earlier pipeline stage, e.g. before re-tracking).

- **`globalfragment.py`** — `GlobalFragment`: wraps the `N` coexisting individual `Fragment`s
  at one "everyone visible and separated" moment. Exposes `minimum_distance_travelled`
  (used to pick the bootstrap global fragment), `is_unique()`/`is_partially_unique`
  (uniqueness checks used by the accumulation strategies), and `acceptable_for_training()`.

- **`list_of_blobs.py`** — `ListOfBlobs`: wraps `blobs_in_video: list[list[Blob]]`
  (one list per frame). Key methods: `compute_overlapping_between_subsequent_frames()`
  (the all-pairs overlap test that builds the `.next`/`.previous` graph — the foundation
  fragmentation is built on), `set_images_for_identification()` (parallel generation of every
  blob's identification image into per-episode HDF5 files), and pickle-based `save()`/`load()`
  (including migration from the old v4 `.npy` format).

- **`list_of_fragments.py`** — `ListOfFragments`: wraps `fragments: list[Fragment]`.
  `from_fragmented_blobs()` builds the fragment list from a blob graph (called by
  `fragmentation_API`). `connect_coexisting_fragments()` populates every fragment's
  `coexisting_individual_fragments` (an O(n log n)-ish sweep when fragments are sorted by
  start frame, else brute force) — this is what the entire P1/P2 constraint system depends on.
  `compute_P2_vectors()`, `get_fragments_to_identify()` (certainty-ordered generator used by
  residual identification), `manage_accumulable_non_accumulable_fragments()`,
  `update_blobs()` (pushes fragment-level identities down onto blobs), `get_stats()`
  (the end-of-run summary counters), and JSON save/load via a custom `FragmentsEncoder`.

- **`list_of_global_fragments.py`** — `ListOfGlobalFragments`: wraps
  `global_fragments`/`non_accumulable_global_fragments`. `from_fragments()` (called by
  `fragmentation_API`) scans the video frame-by-frame with `get_global_fragment_core()` to
  find the first frame of every "core" stretch (exactly `N` individual blobs, same fragment
  identifiers as the previous core frame) and builds one `GlobalFragment` per stretch. The
  constructor splits results into accumulable vs. non-accumulable based on minimum fragment
  length. `sort_by_distance_travelled()` / `sort_by_distance_to_the_frame()` support bootstrap
  selection and ordering in `tracker.py`.

### 4.2 `base/` — the tracking algorithm

- **`base/run.py`** — `RunIdTrackerAi`: the top-level orchestrator. `track_video()` calls the
  five stage APIs in order (see §1's table), saving `Session`/`ListOfBlobs`/`ListOfFragments`/
  `ListOfGlobalFragments` after each stage, logs the final estimated accuracy, and on any
  failure attaches a copy of the run's log file to the exception for debugging. This is the
  file to read first to see the whole pipeline as a sequence of calls.

#### `base/animals_detection/` — Stage 1: segmentation

- **`__init__.py`** — Re-exports `animals_detection_API` plus a few segmentation helpers for
  reuse by the GUI.
- **`animals_detection.py`** — `animals_detection_API(session)`: prepares bbox-image storage,
  computes or loads the background model if `use_bkg` is set, calls `segment()`, wraps the
  result in a `ListOfBlobs`, and runs `check_segmentation()` — a sanity check that warns/errors
  if frames have more blobs than the declared number of animals (a strong signal of noisy
  segmentation that will hurt every downstream stage).
- **`segmentation.py`** — The actual per-frame CV algorithm. `process_frame()`: grayscale →
  threshold (fixed intensity band, or background-subtraction `absdiff` if a background model
  is supplied) → optional ROI mask → `cv2.findContours` → filter by area thresholds.
  `get_blobs_in_frame()` wraps each surviving contour into a `Blob` and stores its cropped
  bounding-box image into an HDF5 file. `segment()` parallelizes this across the video's
  `Episode`s (frame chunks, for multiprocessing) via `segment_episode()` workers. Also hosts
  background-computation helpers (`compute_background`, `generate_frame_stack`,
  `generate_background_from_frame_stack`, `load_custom_background`) and the standalone
  `idtrackerai_background` CLI entry point.

#### `base/crossings_detection/` — Stage 2: individual vs. crossing classification

- **`__init__.py`** — `crossings_detection_API(session, list_of_blobs)`: computes the median
  body length (`model_area.compute_body_length`) to auto-size identification images
  (`session.set_id_image_size`), generates every blob's identification image, calls
  `list_of_blobs.compute_overlapping_between_subsequent_frames()` (builds the blob-linkage
  graph fragmentation depends on), then either trivially marks everything individual
  (single-animal case) or calls `detect_crossings()`.
- **`crossing_detector.py`** — `detect_crossings()`: first applies cheap heuristics
  (`_apply_area_and_unicity_heuristics`, using `ModelArea` from `model_area.py`) to label
  "sure" individuals/crossings; gathers a training set via `crossings_dataset.py`; if there's
  enough data, trains a small binary `IdCNN` (individual vs. crossing) using the shared
  training loop in `base/network/train.py`; if training data is too scarce or the model
  diverges, falls back to the heuristic labels alone. Predicts on the remaining "unknown"
  blobs and persists crossing labels into the identification-image HDF5 files.
- **`crossings_dataset.py`** — `get_train_validation_and_eval_blobs()`: builds the labeled
  training set for the crossing detector using topological heuristics on the blob-overlap
  graph — `Blob.is_a_sure_individual()`/`is_a_sure_crossing()` (defined in `blob.py`) plus
  "this frame has exactly N individual blobs" — rather than raw area alone, since a clean
  1:1-linked chain bookended by crossings is very likely a genuine single individual the whole
  way through.
- **`model_area.py`** — `ModelArea`: a simple statistical (median/std) model of blob area from
  "clean" frames (exactly `N` blobs), used as the first-pass individual/crossing heuristic
  before any CNN exists. `compute_body_length()`: analogous median-based estimate of animal
  size (bounding-box diagonal), feeding `Session.set_id_image_size()`.

#### `base/fragmentation/` — Stage 3: Blobs → Fragments → GlobalFragments

- **`__init__.py`** — Re-exports `fragmentation_API` and `find_exclusive_contours`.
- **`fragmentation.py`** — `fragmentation_API(session, list_of_blobs)`: optionally tags blobs
  with an exclusive ROI region (`set_blobs_ROI`, for videos with disjoint enclosures treated as
  separate identity pools), assigns every blob a `fragment_identifier` via
  `compute_fragment_identifier()` (walks the blob graph, cutting a new fragment whenever the
  1:1 linkage breaks or the individual/crossing label changes), builds `ListOfFragments` and
  `ListOfGlobalFragments` from the result, and partitions fragments into accumulable /
  non-accumulable sets. Also defines the ROI-mask contour utilities
  (`find_exclusive_contours`, `find_parent_ROI`) used when `exclusive_rois` is enabled.

#### `base/network/` — shared CNN architecture + generic training infrastructure

- **`__init__.py`** — Re-export hub for `DEVICE`, the model classes, and training utilities;
  enables `torch.backends.cudnn.benchmark`.
- **`models.py`** — `IdCNN`: the small custom CNN used both as the crossing-detector (2-way)
  and the identity classifier (N-way) — 3 conv+pool blocks → 2 FC layers, Xavier-init, with a
  `load()` that handles several legacy checkpoint formats/key-naming schemes. `ResNet18`: a
  1-channel-input, no-bias-output ResNet-18 used purely to produce embeddings for the
  contrastive step. `IdentifierBase`/`IdentifierIdCNN`/`IdentifierContrastive`: a common
  "image → per-animal probability vector" interface, letting the rest of the pipeline (residual
  identification, trajectory certainty computation) treat the supervised CNN and the
  unsupervised contrastive+K-means model interchangeably. `load_identifier_model()`: generic
  loader that tries the contrastive format first, then falls back to `IdCNN` — used both for
  loading the final trained model and for knowledge transfer from a prior session.
- **`train.py`** — Generic supervised training loop shared by the crossing detector and the
  identity CNN: `StopTraining` (adaptive early stopping — max epochs, NaN-loss divergence,
  overfitting-streak counter, loss-plateau detection, perfect validation accuracy),
  `train_loop`/`train`/`evaluate`, `ImageDataset`/`get_dataloader` (with 180°-rotation data
  augmentation for training), and `get_predictions`/`get_onthefly_dataloader` (memory-efficient
  lazy-loading inference used throughout the identification cascade).
- **`device.py`** — `_get_device()`: resolves the compute device (explicit user choice → CUDA
  → Apple MPS → CPU), exposed as the module-level `DEVICE` singleton used everywhere a tensor
  needs a device.

#### `base/tracker/` — Stage 4: the identification cascade (the core ML system)

- **`__init__.py`** — `tracker_API(session, list_of_blobs, list_of_fragments,
  list_of_global_fragments)`: dispatches to `track_without_identities()` (a no-CNN heuristic
  fallback that fills identity "slots" purely from fragment continuity) if requested, a trivial
  single-animal/single-fragment shortcut, or the full `run_tracker()` pipeline. Returns the
  trained `identifier_model`, later consumed by `trajectories_API`.
- **`tracker.py`** — `run_tracker()`/`fragment_identification()`: the main orchestrator of the
  cascade described in §1 — picks the bootstrap global fragment, calls `identity_transfer.py`
  to seed it, tries the unsupervised `contrastive_step()` first, and if that isn't sufficient,
  initializes an `IdCNN` (optionally via knowledge transfer) and runs the supervised
  accumulation loop (`accumulator.accumulation_step()`) until no more fragments can be
  confidently added. Also handles the penultimate-checkpoint safety rollback
  (`assigner.check_penultimate_model`) and final model persistence.
- **`accumulation_manager.py`** — `AccumulationManager`: the P1-based acceptance engine. Builds
  balanced (new + previously-used, capped, class-balanced) training batches each round
  (`get_old_and_new_images`), and implements the two acceptance strategies:
  `check_if_is_globally_acceptable_for_training()` (all-or-nothing per global fragment: every
  fragment must be certain, and a greedy most-confident-first pass must assign each a unique,
  coexistence-consistent identity) and `check_if_is_partially_acceptable_for_training()`
  (per-fragment fallback, requiring most coexisting fragments already trained). Also computes
  P1 vectors from raw predictions (`split_predictions_after_network_assignment`) and tracks
  extensive rejection-reason counters for diagnostics.
- **`accumulator.py`** — `accumulation_step()`: one round of the supervised cascade — build
  balanced dataloaders with inverse-frequency class weights, train with `train_loop`, commit
  newly-accepted fragments, checkpoint the model (keeping a penultimate copy), and — unless the
  accumulated-image ratio already exceeds the early-stop threshold — predict on the remaining
  pool and re-run `AccumulationManager.assign_identities()` to grow the accepted set for the
  next round.
- **`assigner.py`** — `assign_remaining_fragments()`: the residual-identification stage.
  Predicts identities for every fragment never used in training, computes P2 vectors for
  **all** individual fragments (`Fragment.compute_P2_vector`, encoding "coexisting fragments
  can't share an identity"), and resolves fragments in descending-P2-certainty order via
  `Fragment.assign_identity()`, propagating each resolution into its still-unresolved
  neighbors immediately. `check_penultimate_model()` compares the last two cascade checkpoints
  and rolls back if the final round actually regressed.
- **`contrastive.py`** — `ContrastiveLearning`: the unsupervised bootstrap/quick-track step.
  Builds positive pairs (same fragment) and negative pairs (coexisting, hence provably
  different, fragments), trains a `ResNet18` embedding with a margin-based contrastive loss and
  online hard-pair mining (`BatchSampler`, loss-score-weighted pair sampling), validates with a
  GPU-native silhouette-score implementation, and clusters the final embedding space with
  K-means (scikit-learn `MiniBatchKMeans`, `predict()`), aligning cluster labels to the
  bootstrap identities via the Hungarian algorithm. Feeds its results
  into the exact same `AccumulationManager.assign_identities()` machinery the supervised CNN
  uses — the two approaches are interchangeable inputs to one acceptance system.
- **`identity_transfer.py`** — `identify_first_global_fragment_for_accumulation()`: seeds the
  very first identities of a run, either arbitrarily (0..N-1) or, if a prior model is supplied,
  via `get_transferred_identities()` — running the old model on the new video's bootstrap
  fragment and solving a Hungarian assignment between "this video's fragments" and "the old
  model's identity classes." This is what lets idtracker.ai recognize *the same
  individually-known animals* across separate recording sessions.

#### `base/postprocess/` — Stage 5: corrections and trajectory export

- **`__init__.py`** — Re-exports `trajectories_API` and `produce_output_dict`.
- **`correct_impossible_jumps.py`** — `correct_impossible_velocity_jumps()`: scans fragments
  outward from the bootstrap global fragment (past and future) and, wherever a fragment's
  identity implies a physically-impossible instantaneous velocity to its same-identity
  neighbor, reassigns whichever fragment in the chain isn't `identity_is_fixed`, choosing among
  candidate identities by minimum implied speed intersected with above-random P2 confidence.
- **`erosion.py`** — `get_eroded_blobs()`: rasterizes all blobs in a frame, erodes the binary
  mask with a kernel sized from the video's typical animal half-width
  (`compute_erosion_disk()`), and re-extracts contours — a classical-CV trick for splitting
  touching/crossing animals into separable sub-blobs so their identities can be recovered.
- **`assign_them_all.py`** — `close_trajectories_gaps()`: the crossing/gap-closing engine.
  Iteratively (re-eroding more aggressively each pass) finds frame ranges where an identity is
  missing, interpolates where that animal should be from its bordering individual fragments,
  and matches the interpolated position against candidate (possibly eroded) blobs to fill in
  `identities_corrected_closing_gaps`/`interpolated_centroids` — this is how idtracker.ai
  recovers identities through crossings without ever training a per-crossing classifier.
- **`trajectories_creation.py`** — `trajectories_API()`: the final orchestrator — computes the
  velocity threshold used by the two files above, runs impossible-jump correction and gap
  closing, computes per-blob identity confidence (`compute_identity_probabilities`), flattens
  everything into a plain dict (`produce_output_dict`: trajectories array, per-identity
  confidence, area statistics, estimated accuracy, metadata), and hands it to
  `utils/trajectories_io.py` for saving.

### 4.3 `utils/` — configuration, logging, I/O, and general helpers

- **`__init__.py`** — Re-export surface most of the package imports from
  (`from idtrackerai.utils import ...`): `conf`, logging helpers, `Episode`/`Timer`/
  `LengthCalibration`/`IdtrackeraiError`, path/JSON/TOML helpers, progress-bar helpers, and
  trajectory I/O.
- **`confparams.py`** — `ConfParams` (a `dataclass` singleton, `conf`): every algorithm
  hyperparameter/threshold used across the pipeline (crossing-detector training params,
  accumulation thresholds like `CERTAINTY_THRESHOLD`/`FIXED_IDENTITY_THRESHOLD`, contrastive
  settings, device/compile flags). `set_parameters()` mirrors `Session.set_parameters()`'s
  pattern for CLI/TOML overrides.
- **`logging_utils.py`** — Logging setup and top-level error handling. `init_logger()`,
  the cross-process `LOGGING_QUEUE`/listener-thread machinery (so worker processes in
  `multiprocessing.Pool` calls log correctly), `wrap_entrypoint()` (the decorator every CLI
  entry point uses — sets up logging, runs telemetry/version-check threads, and turns
  exceptions into clean user-facing messages via `manage_exception()`).
- **`py_utils.py`** — General-purpose helpers and small data classes used everywhere:
  `IdtrackeraiError`, `Episode` (a video frame-chunk used for parallelization),
  `Timer` (stage timing, used pervasively in `run.py`), `LengthCalibration` (Validator
  pixel-to-real-unit calibration), `get_params_from_model_path()` (extracts n_classes/
  image_size from a saved model for knowledge transfer), `load_id_images()` (loads
  identification images from HDF5 given index/episode pairs), `build_ROI_mask_from_list()`,
  `resolve_path()`, JSON encode/decode hooks, and more.
- **`rich_utils.py`** — `track()`/`open_track()`: shared `rich`-based progress-bar wrappers
  used throughout the pipeline and GUIs for consistent progress reporting.
- **`telemetry.py`** — Anonymous usage reporting and PyPI version-check (`check_version()`,
  `report_usage()`), both opt-out via an env var or a GUI menu toggle.
- **`trajectories_io.py`** — `save_trajectories()`/`load_trajectories()`: serializes the plain
  dict from `produce_output_dict()` into any of `h5`, `npy`, `csv`, `csv_tidy`, `pickle`,
  `parquet` (and back). Deliberately has no dependency on `Blob`/`Fragment`/`Session` — it only
  ever touches the plain output dict, decoupling storage format from the tracking object model.
  Also hosts the `idtrackerai_format` CLI entry point for converting between formats.

### 4.4 `start/` — CLI entry point

- **`__main__.py`** — `main()` (the `idtrackerai` console command): merges parameters from
  `local_settings.toml`, `--load`ed TOML files, and CLI flags; applies them to `conf` and a new
  `Session`; if `--track` wasn't passed, launches the Segmentation GUI first so the user can
  interactively tune thresholds/ROI/background; then imports and runs
  `base.run.RunIdTrackerAi(session).track_video()` — this file is the bridge between the CLI/
  config layer and the core pipeline described in §4.2. Also defines `general_test()` (the
  `idtrackerai_test` smoke-test command, using the bundled `data/test_B.avi`).
- **`arg_parser.py`** — `get_parser()`/`parse_args()`: the full `argparse` CLI surface, with
  defaults auto-populated from `Session`/`ConfParams` and arguments grouped by category
  (General, Output, Background Subtraction, Parallel processing, Knowledge/identity transfer,
  Contrastive, Advanced hyperparameters, Deprecated).

### 4.5 `GUI_tools/` — shared Qt widget framework (used by every GUI app below)

- **`GUI_main_base.py`** — `GUIBase(QMainWindow)`: common base window class (theming, About/
  View menus, update-check, usage-analytics opt-in) shared by the Segmentation app, Validator,
  and video-generator GUI.
- **`themes.py`** — Dark/light `QPalette` definitions used by `GUIBase`'s theme toggle.
- **`widgets_utils/canvas.py`** — `Canvas`: the shared paint-surface widget for drawing video
  frames plus overlays (ROIs, blobs, trajectories); used by the Segmentation app and Validator.
- **`widgets_utils/custom_list.py`**, **`id_labels.py`**, **`other_utils.py`**,
  **`sliders.py`** — Reusable small widgets (removable list items, per-identity label/color
  management, misc buttons/dialogs, threshold range sliders) shared across the GUI apps.
- **`widgets_utils/video_paths_holder.py`** — `VideoPathHolder`: non-Qt helper wrapping
  `cv2.VideoCapture` across potentially multiple video-chunk files with cache-aware seeking.
- **`widgets_utils/video_player.py`** — `VideoPlayer`: the shared playback widget (built on
  `Canvas` + `VideoPathHolder`) with play/pause/speed control and async frame preloading.

### 4.6 `segmentation_app/` — pre-tracking GUI for segmentation setup

- **`main.py`** — `SegmentationGUI(GUIBase)`: the main window tying together video loading,
  live threshold preview, ROI editing, background computation, and tracking-interval
  selection, ultimately writing parameters back into the `Session` before (optionally)
  launching the tracker.
- **`widgets/area_ths.py`**, **`intensity_ths.py`** — Min/max threshold slider widgets
  mirroring `Session.area_ths`/`intensity_ths`.
- **`widgets/bkg_widget.py`** — Background-subtraction controls (statistic choice or custom
  image), running the (slow) computation off the UI thread.
- **`widgets/blob_info_widget.py`** — Per-frame blob-count chart, to help spot segmentation
  problems (compare against `check_segmentation()` in `animals_detection.py`).
- **`widgets/frame_analyzer.py`** — Runs the real `segmentation.process_frame()` live on the
  currently displayed frame so threshold edits are reflected immediately.
- **`widgets/open_video_widget.py`** — Video file(s) picker.
- **`widgets/ROI_widget.py`** — Interactive ROI polygon/ellipse editor, producing the
  string-encoded ROI list `py_utils.build_ROI_mask_from_list()` consumes.
- **`widgets/track_intervals_widget.py`** — Restricts tracking to specific frame ranges
  (`Session.tracking_intervals`).

### 4.7 `extra_tools/` — post-run utilities

- **`cluster_inspection.py`** — `idtrackerai_inspect_clusters` CLI: embeds a sample of
  identification images with the session's trained contrastive `ResNet18`, projects to 2D with
  t-SNE, and plots the result — a diagnostic for the contrastive step's cluster quality.
- **`idmatcherai.py`** — `idmatcherai` CLI: matches/re-identifies animals across *different*
  tracked sessions using each session's trained model and a Hungarian-algorithm assignment
  over cross-session prediction scores — the batch/offline counterpart to the
  `identity_transfer.py` mechanism used inside a single tracking run.
- **`validator/validation_GUI.py`** and **`validator/widgets/*`** — `ValidationGUI`: a full GUI
  for manually inspecting and correcting a completed tracking session frame-by-frame (fixing
  identities, adding/removing centroids, interpolating gaps, calibrating length units, managing
  identity groups) — this is what ultimately reads/writes the `Blob.user_generated_*` fields
  defined in `blob.py`.
- **`video_generator/*`** — `idtrackerai_video` CLI/GUI: renders annotated output videos from
  saved trajectories, either as one combined overview video (`general_video.py`) or a
  per-individual "miniframe" collage video (`individual_videos.py`).

### 4.8 `data/`

- **`test_A.avi`**, **`test_B.avi`** — Small sample videos bundled with the package, used by
  the `idtrackerai_test` smoke-test entry point (`start/__main__.py:general_test()`).

---

## 5. How to trace a value through the system (worked example)

As a concrete way to see the relationships above in action: to understand where a final
trajectory's identity for one animal in one frame comes from, follow this chain —

`Blob.identity` (or `.identity_corrected_solving_jumps` / `.identities_corrected_closing_gaps`
if postprocessing touched it) → set from `Fragment.identity` via `ListOfFragments.update_blobs()`
→ `Fragment.identity` set either during accumulation
(`AccumulationManager.assign_identities_to_fragments_used_for_training`, for fragments used in
training) or during residual identification (`Fragment.assign_identity`, driven by its
`P2_vector`) → `P2_vector` computed from `P1_vector` and every `coexisting_individual_fragments`
member's `P1_vector` (`Fragment.compute_P2_vector`) → `P1_vector` computed from raw per-image
CNN (or contrastive+K-means) predictions across the fragment's images
(`Fragment.set_P1_from_frequencies`) → those predictions come from `identifier_model` (an
`IdentifierIdCNN` or `IdentifierContrastive`), trained by the cascade in `base/tracker/`.
