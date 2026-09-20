# UPDATES — SLEAP → idTracker.ai port

Live working log for the sleap-idtracker implementation. Plan of record is
`Obsidian/Shaevitz_Lab/SLEAP/idtracker/sleap-idtracker-implementation.md`; this file records what
was actually *built*, what worked, and what didn't.

Newest entries at the top of each section.

---

## Repository layout

```
sleap_idtracker/
  vendor/                    read-only reference forks (do not edit)
    sleap/                   v1.6.3
    sleap-nn/                v0.3.0
    sleap-io/                v0.7.1
  idtrackerai/               upstream clone (gitlab polavieja_lab), surgical edits only
  src/sleap_idtracker/       our adapter code
    episodes.py              Stage 1 — episode segmentation
    crops.py                 Stage 2 — keypoint-derived crops / id-images
  predictions/               .slp prediction files
  UPDATES.md                 this file
```

**Editing policy.** `vendor/` is reference only — never edited. `idtrackerai/` gets *surgical* edits
at injection points only; every removal is commented out in place with a `# SLEAP-PORT:` marker
explaining why it is not needed, never deleted. New logic lives in `src/sleap_idtracker/`.

---

## 2026-07-21 — Session 1: forks, environment, data survey

### Forks pulled

| Package | Version | Source | Why this version |
|---|---|---|---|
| `sleap` | v1.6.3 | github.com/talmolab/sleap | requested |
| `sleap-nn` | v0.3.0 | github.com/talmolab/sleap-nn | requested (note: sleap 1.6.3 itself ships 0.2.0; 0.3.0 is the newer standalone release) |
| `sleap-io` | v0.7.1 | github.com/talmolab/sleap-io | "the default io package that comes with SLEAP" — `sleap==1.6.3` pins `sleap-io[all]>=0.7.0,<0.8.0` in `vendor/sleap/pyproject.toml:55`; 0.7.1 is the newest release in that range |

All three are shallow clones at the tag (detached HEAD).

### Environment

Created conda env **`sleap_id`** (python 3.11) with `sleap-io[all]==0.7.1`, `opencv-python`, `h5py`.

Rationale: no existing env had `sleap-io`. `sleap_env` is empty/broken (only `conda-meta` and `etc`
survive — no `bin/`). The `idtrackerai` env has h5py/numpy/torch/cv2 but no sleap-io, and was left
untouched deliberately so the reference idtracker install stays reproducible.

```
conda create -y -n sleap_id python=3.11
pip install 'sleap-io[all]==0.7.1' opencv-python h5py
```

### Working dataset

Switched from `extended_vid.v020.*` to the **fourfly clip** for this iteration (shorter, faster
turnaround, and its video reference resolves locally):

- Predictions: `predictions/fourfly_clip.mp4_labels.v001.slp.260721_104536.predictions.slp`
- Video: `/Users/matthewsierra/sleap_context/videos/fourfly_clip.mp4` (also copied to repo root)

Survey of the predictions file (via raw h5py):

| Property | Value |
|---|---|
| frames | 2000 (`frame_idx` 0–1999, contiguous, single video) |
| instances | 32000 → **exactly 16 per frame** |
| pred_points | 416000 → **13 nodes per instance** |
| skeleton | head, eyeL, eyeR, thorax, abdomen, forelegL/R, midlegL/R, hindlegL/R, wingL/R |
| tracks | **none** (`tracks_json` empty; every instance has `track == -1`) |
| video backend | `MediaVideo`, 2000 × 1536 × 1536, `grayscale=true`, fps 99.96 |

Absence of tracks is expected and correct — identity assignment is precisely the job we are handing
to idTracker.

### RESOLVED: can we work from the `.slp` alone?

**Partly — and the distinction matters.**

- **Episodes: yes, fully.** Episode segmentation needs only frame *counts* and indices. Pure `.slp`,
  no pixels touched.
- **Crops: no, not literally.** The predictions file contains **zero pixel data** — `points` is
  empty, `pred_points` holds coordinates only, and there is no embedded `videoN/video` dataset (the
  way `extended_vid.v020.pkg.slp` has one). Pixels have to come from the mp4.

  The practical saving grace: the `.slp`'s video reference is an **absolute local path that
  resolves** (`/Users/matthewsierra/sleap_context/videos/fourfly_clip.mp4`). So sleap-io opens the
  video transparently on `lf.image` — our code opens *only* the `.slp` and never names the mp4
  itself. That satisfies the spirit of "extraction only with the SLEAP file" while being honest that
  the bytes come off disk from the mp4.

  ⚠️ This makes the `.slp` **non-portable**: move it to another machine and the crops break. Worth
  recording for the writeup.

  Note this corrects the plan note's "We don't need the original mp4 — all we need are the frames
  that make up the video." The frames are not *in* the `.slp`.

### Reference points located in idtrackerai

- `Episode` dataclass — `idtrackerai/src/idtrackerai/utils/py_utils.py:176` (7 fields, `slots=True`,
  `length` property). Matches the plan note exactly.
- Episode construction — `idtrackerai/src/idtrackerai/session.py:964`, inside the static helper
  starting at `:871`. Builds "long episodes" from video-path changes ∪ tracking-interval changes,
  then subdivides each with `np.linspace` to respect `frames_per_episode`.
- Deserialization — `py_utils.py:528` (`Episode(**d)`).
- **`frames_per_episode` default is `500`** (`session.py:101`), *not* ~400. The plan note's
  validation gate says "chunks of ~400 frames"; going with upstream's 500 unless told otherwise.
  Flagged for Matthew.

### Not yet done / open

- Repo root is **not a git repo** (`idtrackerai/` is a nested clone with its own `.git`). No
  `git init` performed — surgical-change tracking would benefit from one, but it needs a decision on
  how to handle the nested clone (submodule vs. ignore vs. vendored-with-history). Flagged.

---

## Episodes (Stage 1) — WORKING

`src/sleap_idtracker/episodes.py`. Verbatim port of `Session.get_processing_episodes`
(`idtrackerai/src/idtrackerai/session.py:868`), reusing idtracker.ai's **real** `Episode` dataclass
rather than redefining it. Frame counts come from `.slp` metadata instead of `cv2.VideoCapture`; the
mp4 is never opened, because episode segmentation needs no pixels. `bbox_images` left `None` (crop
agent's territory). **Zero edits to `idtrackerai/` or `vendor/`.**

### Validated (real assertions, re-verified independently by the orchestrator)

2000 frames at `frames_per_episode=500` → **4 episodes**:

| idx | global_start | global_end | length |
|---|---|---|---|
| 0 | 0 | 500 | 500 |
| 1 | 500 | 1000 | 500 |
| 2 | 1000 | 1500 | 500 |
| 3 | 1500 | 2000 | 500 |

Tiles `[0, 2000)` with no gaps or overlaps; `sum(length) == 2000`; every `frame_idx` maps to exactly
one episode; `local_* == global_*` asserted rather than assumed.

### Upstream quirk worth knowing — do NOT "fix" it

`n_subepisodes = int((end - start) / (frames_per_episode + 1))`. The `+1` means episodes can
**exceed** `frames_per_episode`: at `n=1001, fpe=500` upstream yields `[500, 501]`. Confirmed by
sweep (63 of ~440 cases). Our 2000/500 case is clean. This is upstream behaviour and we match it
deliberately — matching boundary semantics is the whole point of the port.

Related: a bare `Episode` cannot round-trip through `json_default`. Not our import route — upstream
`session.py:350` deliberately pops `episodes` before saving.

### RESOLVED: how to import idtracker.ai

The agent hit `ModuleNotFoundError` (`toml`, `deprecated`, `requests` — pulled in by
`idtrackerai/__init__` → `logging_utils` → `telemetry`, none of which episode code touches) and built
a file-path fallback loader. **Superseded.** Fixed properly instead:

```bash
pip install toml deprecated requests
pip install -e ./idtrackerai --no-deps
```

`from idtrackerai.utils.py_utils import Episode` now works cleanly. Editable install is the right
shape for a surgical-fork project: our edits take effect immediately, imports are normal, no
`sys.path` manipulation. **TODO: the fallback loader in `episodes.py` is now dead weight and should
be deleted** — it also returns a distinct class *object*, so cross-route `isinstance` would fail.

---

## Crops / id-images (Stage 2) — PLUMBING WORKS, IMAGE QUALITY DOES NOT

`src/sleap_idtracker/crops.py`, output in `crops_frame0/` (33 files: raw + id-image per instance,
plus `overview.png`). Kept the `.slp`-only entry point — `lf.image` resolved and decoded fine, so
opening the mp4 directly was never needed. Zero edits to `idtrackerai/` or `vendor/`.

### What works

All 16 frame-0 instances produced valid, non-degenerate crops. Widths 53–99 px (median 65.5),
heights 65–101 px (median 81) — **non-square and unscaled**, as required. Zero missing keypoints,
zero edge-touching boxes, zero empty or degenerate hulls. `overview.png` confirms placement: four
flies per arena, every hull on a fly. Instances 6/7 are adjacent — a useful built-in overlap case.

### Bug found and fixed in our usage of `Instance.bounding_box()`

The plan note says to use `Instance.bounding_box()`. sleap-io 0.7.1 filters on the `visible` **flag**,
not on NaN — so a point that is NaN *while still flagged visible* propagates NaN through min/max and
poisons the entire box. Flag and finiteness agree on all 416,000 nodes in this clip, so the bug is
**latent, not active** — but one disagreeing point would kill a frame silently. Extent is now derived
from the same finite-filtered points that build the hull.

⚠️ The missing-keypoint path is therefore **unexercised by this dataset** (zero NaN in 32,000
instances) and is verified only synthetically.

### ⚠️ The id-images are not good silhouettes — do not train on these yet

Three problems, in descending severity. Orchestrator independently re-measured and confirms:

1. **The convex hull is mostly floor.** Measured background fraction inside the hull ranges
   **43%–77%** (worst: inst07 77%, inst15 72%, inst05 70%). A hull spanning splayed legs encloses
   large triangles of empty arena. Worse, the hull shape *encodes leg splay* — which is **pose, not
   identity**, exactly the nuisance variable the identity network should be invariant to.
2. **Polarity is inverted vs. idTracker's assumption.** These flies are dark-on-bright (body ≈52,
   floor ≈154); idTracker.ai assumes background-subtracted **bright-on-black**. Masked-out area
   becomes 0 (black), in-hull floor stays bright, so the animal is a *mid-grey* blob sandwiched
   between the two. This will not fail loudly — it will just train worse.
3. **Clipping at head/abdomen tips** — mild, ~1.5% of body pixels.

### Open decision: is convexity even the right shape model?

Flagged by the crop agent and I think it's right. A **capsule-along-skeleton-edges** mask (union of
thick line segments following the skeleton graph) would hug the animal far better, would not inflate
with leg splay, and stays purely geometric — **no thresholding**, so it respects the hard constraint.
The brief specified convex hull so convex hull was built, but this looks like the main limit on
id-image quality. **Matthew's call.**

Three smaller placeholders also awaiting a decision, currently `pad=0`, `dilate=0`, overlaps ignored:
crop padding, hull dilation, and overlapping-hull policy.

---

## Stage 2b — Rectangular crops (`make_boxes.py`) — WORKING

Scope correction: the masked id-images above ran ahead of what was actually wanted. The first
milestone is simply *boxes that contain the animals*. `crops_frame0_boxes/` holds 16 rectangular
crops (one per fly, frame 0), 52–98 × 64–100 px. Nothing else — no masking, no rescaling.

Kept from the earlier work because both are about getting the *box* right:

- extent from **finite** keypoints, not sleap-io's `visible` flag (the two can disagree; a NaN point
  still flagged visible poisons the box),
- boxes clipped to frame bounds.

`PAD = 0` gives the tight keypoint box, which clips slightly at head and abdomen tips since the
extent only covers labeled nodes.

Measured: **61–85% of each rectangle is arena floor** (median 69%) — worse than the convex hull's
43–77%, as expected.

---

## Stage 2c — Egocentric rotation (`rotate_boxes.py`) — WORKING

Aligns every animal to a common heading using two user-selected keypoints:

```python
TOP_NODE      = "head"    # ends up pointing up
CENTROID_NODE = "thorax"  # rotate about this
```

All 16 frame-0 instances rotated head-up. Output in `crops_frame0_rotated/` (+ `_montage.png`).

### Two things that needed care

1. **Image coordinates have +y pointing DOWN**, while "angle with the +y axis" is the y-up maths
   convention. The conversion is `angle = atan2(dx, -dy)`, derived in the docstring against cv2's
   rotation matrix. Rather than trust the sign, the code pushes both alignment keypoints through the
   matrix and **asserts** the top node lands above the centroid and vertically aligned with it.
   Passed for all 16.
2. **Rotation happens in the FULL FRAME, not in the tight crop.** Rotating an already-tight rectangle
   discards whatever swings past its corners — exactly the leg and wing tips the box exists to
   contain. Rotating about the centroid and cropping afterwards is the same operation with nothing
   lost. This was a deliberate deviation from the literal instruction, made to honour its stated
   intent ("all other information about the body is kept the same").

### Side benefit

Boxes became **more uniform**: width spread 52–98 → 50–93, with the wide outliers mostly collapsing.
A diagonal fly has a large axis-aligned box before rotation and a snug one after. Less variation that
has nothing to do with identity.

Useful fixed-point property: rotating *about* the centroid leaves the centroid's coordinates
unchanged, so downstream window extraction needs no coordinate bookkeeping.

---

## Stage 2d — Canonical-frame hull images (`canonical_hulls.py`) — WORKING, WITH KNOWN CLIPPING

Fixed-size frame so the identity network sees consistent input. Adds a third configurable node:

```python
BOTTOM_NODE = "wingR"     # vertical placement reference
```

Geometry, per spec:

| quantity | rule | value (frame 0) |
|---|---|---|
| WIDTH | median over flies of x-spread | **55** |
| HEIGHT | mean over flies of fly length | **87** |
| centroid height | median centroid→bottom distance, above the bottom edge | **58.9** (= 28 from top) |

Centroid is deliberately **not** vertically centred — the body extends ~2.2× further down than up
(median 59.3 down vs 27.4 up), so centring would waste the top third and clip the bottom. Horizontal
placement was unspecified; centroid centred at WIDTH/2 (**assumption, flagged**).

Output: `crops_frame0_canonical/`, 16 images at a uniform 55×87 + `_montage.png`.

### ⚠️ Clipping is substantial and was chosen with eyes open

**14/16 flies are clipped** (left 7, right 10, top 8, bottom 10). This is inherent to sizing from a
central statistic: a median is by definition exceeded by half the population, and clipping on *any*
of four sides counts.

Alternatives were measured and presented before building:

| sizing | frame | flies clipped |
|---|---|---|
| median/mean (**chosen**) | 55 × 87 | 14/16 |
| p90 | 86 × 99 | 5/16 |
| max | 92 × 104 | 0/16 |

Matthew chose the median/mean spec as written. Recorded here because the visible consequence is that
hulls get **cut flat against the frame edge** (flies 2, 4, 5, 7, 15), turning the mask boundary into
a straight line the network may read as a feature. Fly 10 is the only fly clipped nowhere.

Note for later: because everything outside the hull is masked to **black**, enlarging the frame adds
black pixels rather than arena floor — so generous sizing costs far less here than it would for
rectangular crops. That inverts the usual size/background tradeoff.

### Spec contradiction found and resolved before building

The request contained two incompatible frame definitions: "x = median x spread, y = mean fly length"
(opening) versus "x and y dimensions are the mean distance from centroid to bottom node" (later).

The second is **arithmetically impossible**: mean centroid→wingR is 57.9, so the frame would be 58
tall, but the centroid is to sit 58.9 px above the bottom edge — i.e. 1 px *above the top edge*. All
16 flies clip. It also contradicts the plan note's hard constraint "Non-square id-images." The
opening reading was used.

### Still true after masking

In-hull arena floor **survives** — black is only *outside* the hull. The pale wedges either side of
the abdomen are still floor, so the 43–77% background figure is unchanged by canonicalisation; the
boundary just moved. Combined with dark-on-bright polarity, the animal remains a mid-grey blob
between black padding and bright in-hull floor.

### Known-imperfect reference node

`wingR` is the lowest node in only **9/16** flies (the rest are bottomed by the left wing or a
hindleg), and being the *right* wing makes it asymmetric under wing pose. Kept as specified;
"lowest of all nodes" (median 59.3, near-identical but pose-symmetric) remains the obvious
alternative if this becomes a problem.

---

## Stage 3 — id-image HDF5 files (`build_id_images.py`) — WORKING, TWO FINDINGS

Builds the training inputs idtracker.ai's back half reads. One file per episode in
`session_fourfly/id_images/`, mirroring upstream's layout (`session.py:716`).

| Dataset | Shape | dtype | Source |
|---|---|---|---|
| `id_images` | (n, 83, 57) | uint8 | ours (upstream: square) |
| `crossings` | (n,) | bool | upstream name, our definition |
| `identities` | (n,) | int64 | upstream; all zeros until tracking |
| `frame_numbers` | (n,) | int64 | ours — for fragmentation graphs |
| `global_index` | (n,) | int64 | ours — non-resetting, per spec |
| `local_index` | (n,) | int64 | ours — upstream-compatible row index |
| `instance_idx` | (n,) | int64 | ours — position within the LabeledFrame |

4 episodes × 8000 images = **32,000** images in **13 s**.

### Verified, not assumed

- `global_index` is a bijection onto 0..31999 — unique, contiguous, gapless.
- `global_index - index_offset == local_index` in every file (the attribute makes the
  spec's non-resetting index convertible back to upstream's local one).
- Every frame number falls inside its file's `[global_start, global_end)`.
- All 2000 frames present, exactly 16 rows each.
- `identities` sums to 0 everywhere.

### Efficiency: one composed warp instead of rotate-then-crop

`canonical_hulls.py` rotates the full 1536×1536 frame and then cuts a 57×83 window. Doing
that 32,000 times is ~1000× more pixel work than needed. Here the rotation and the
translation-to-window are **composed into a single affine** handed to `warpAffine` with
`dsize=(W, H)`, so only the output pixels are ever computed. Output is identical.

### ⚠️ Finding 1 — the frame is too small; 11.6% of heads are cut off

Whole-movie medians give **57 × 83**, centroid 25 px from the top. But the median
centroid→head distance is ~27 px, so the top edge sits *below* the typical head.

| Side | Clipped | Share |
|---|---|---|
| left | 14,284 | 44.6% |
| right | 17,073 | 53.4% |
| top | 12,469 | 39.0% |
| bottom | 18,013 | 56.3% |
| **any** | **29,198** | **91.2%** |

**3,716 instances (11.6%) have the head keypoint itself outside the frame.** The head is
both the alignment reference and a strong identity cue, so this is worse than generic
edge clipping. Since everything outside the hull is black, enlarging the frame adds black
padding rather than arena floor — the fix is nearly free.

### ⚠️ Finding 2 — the crossing flag is firing on leg grazes, not occlusions

930/32,000 (2.9%) flagged, but the overlaps are all shallow. Across every overlapping
pair in episode 0, intersection area as a fraction of the smaller hull:

| p10 | p25 | p50 | p75 | p90 | max |
|---|---|---|---|---|---|
| 0.002 | 0.006 | 0.026 | 0.051 | 0.076 | **0.096** |

**No pair anywhere in the episode overlaps by more than 9.6%**, and 74% overlap by under
5%. These are convex hulls grazing at splayed leg and wing tips, not animals occluding
each other. The convex hull spans to the leg tips, so it reports contact long before the
bodies are near. As it stands the flag marks ~930 images as unusable that are in fact
perfectly clean.

Distribution is also clustered — 476 / 2 / 452 / 0 across the four episodes, and the same
pair persists for hundreds of consecutive frames — consistent with two flies resting near
each other rather than with transient crossing events.

### Configurable crossing threshold — three knobs

```python
CROSSING_NODES     = "body"   # "body" (BODY_NODES hull) | "all" (every keypoint)
CROSSING_METRIC    = "iou"    # "iou" | "iomin" (intersection / smaller hull)
CROSSING_THRESHOLD = 0.05
BODY_NODES = ["head", "eyeL", "eyeR", "thorax", "abdomen"]
```

All three are written into every file as attributes, so a built dataset records the
rule that produced it.

`iomin` exists for the size-asymmetric case: if one animal is much smaller, IoU
under-reports even total occlusion of the small one because the union is dominated by
the large hull. Not needed for same-size subjects; `iou` is the default.

### IoU sweep over the whole movie (2000 frames, 32,000 instances)

| IoU threshold | pairs | instances | % of all | frames |
|---|---|---|---|---|
| >0 (any contact) | 465 | 930 | 2.91% | 462 |
| 0.005 | 301 | 602 | 1.88% | 300 |
| 0.010 | 184 | 368 | 1.15% | 184 |
| 0.020 | 61 | 122 | 0.38% | 61 |
| 0.030 | 23 | 46 | 0.14% | 23 |
| 0.035 | 4 | 8 | 0.03% | 4 |
| **0.040** | **0** | **0** | **0.00%** | **0** |

**Maximum full-hull IoU anywhere in the movie: 0.0382.**
**Maximum body-hull IoU anywhere in the movie: 0.0000** — no two fly bodies ever
overlap, in any frame, at all.

Recommended: `body` + `iou` + `0.05`. Result is **0/32,000 crossings**.

### ⚠️ Read the threshold honestly

Any value ≥ 0.04 on full hulls returns zero, so 0.05 is not "tuned" so much as placed
above a noise floor. But **this dataset contains no positives**, so it cannot validate
a threshold — it only establishes that the flies never actually cross. Picking 0.05
because it yields zero here is circular; the number is defensible only because the
body-hull result (exactly 0.0000, always) independently confirms there is nothing to
detect. On a dataset with real crossings the threshold would need to be set against
labelled occlusions, not against this sweep.

The body-hull result is the load-bearing one. It makes the threshold nearly irrelevant
for this clip: at `CROSSING_NODES="body"` every value in [0, 1] gives the same answer.

### Semantics differ from upstream, deliberately

Upstream's crossing blob is **one blob containing N animals** — the thresholder could not
separate them. Ours is **N instances that happen to overlap** — SLEAP already separated
them. Downstream use (exclude from individual fragments) is the same, but the quantity is
not, so upstream's crossing-detector CNN (`crossing_detector.py:81`) is skipped entirely.

### Note: this `.slp` has no tracks

`labels.tracks == []` and every `instance.track is None`. "Instance ids native to SLEAP"
therefore resolve to **position within the LabeledFrame** — enumeration order, carrying no
identity across frames. Fine for indexing rows, and the crossing test is geometric anyway,
but nothing here inherits identity from SLEAP.

### Sizing statistic changed from mean-y to median-y

Stage 2d used median-x / **mean**-y; this stage was specified as "x and y medians". Both
are reported at run time (x: median 56.6 / mean 59.1; y: median 82.8 / mean 84.3), so the
difference is 1–2 px either way. `SIZE_STAT` is a one-line constant.

---

## Stage 3b — Diff against idtracker.ai's own id-image files

Upstream writes these files in three places: `id_images` at `list_of_blobs.py:251`,
`crossings` at `crossing_detector.py:74`, `identities` at `list_of_fragments.py:382`.
Pixel content comes from `Blob.get_image_for_identification` (`blob.py:525`). Compared
against those, not against documentation.

| | idtracker.ai | ours | risk |
|---|---|---|---|
| `id_images` shape | (n, S, S) square | (n, 83, 57) | 🔴 **breaks IdCNN** |
| `id_images` dtype | uint8 | uint8 | ✅ |
| `id_images` compression | gzip | none | 🟡 size only |
| `crossings` | bool, `is_a_crossing` | bool, same polarity | ✅ |
| `identities` | int, 0 = unassigned | int64, all 0 | ✅ |
| image index | resets per episode | global, non-resetting | 🔴 **silent misread** |
| orientation | moments, ±180° ambiguous | head-up, unambiguous | 🟡 ours is better |
| centering | `id_img[-S:, -S:]` corner cut | centroid-exact | 🟡 ours is better |
| mask | dilated 3×3 once | no dilation | 🟡 |
| extra datasets | — | 4 of ours | 🟡 attrs are fragile |

### 🟡 DANGER 1 — IdCNN cannot accept a non-square image — **DOWNGRADED, see Stage 3e**

`base/network/models.py:68`:

```python
nn.Linear(100 * (input_shape[1] // 4) ** 2, 100)
```

The flattened conv output is **squared**, i.e. height and width are assumed equal.
`Session.set_id_image_size` reinforces this: `id_image_size = [max_size, max_size, 1]`
(`session.py:543`). Feeding 83×57 gives a conv output of `100 · 20 · 14 = 28,000`
features into a layer expecting `100 · 20² = 40,000`. It fails loudly, not silently.

**Corrected in Stage 3e: this does not block the identity pipeline.** It was originally
recorded as the first surgical edit the port would need. That was wrong — `IdCNN` is not
the identity model in idtracker.ai 6.x. The contrastive step uses **`ResNet18`**
(`models.py:13`), whose `nn.AdaptiveAvgPool2d((1,1))` collapses the spatial dimensions
before `fc`, so **any** H×W is accepted. Verified: 83×57 batches build and flow.

`IdCNN` is only reached by (a) the crossing detector, which this port does not use because
crossings are geometric, and (b) the supervised accumulation cascade in `tracker.py:176`,
which `tracker_API` skips entirely when there are no Global Fragments (`tracker.py:116`
returns the contrastive identifier early). So the square assumption is only a problem if
the accumulation protocol is later brought into scope.

### 🔴 DANGER 2 — our global index is a valid-looking wrong answer

Upstream's `Fragment.image_locations` are `(image_index, episode)` pairs, and
`load_id_images` uses `image_index` directly as a row index into that episode's dataset
(`py_utils.py:468`). Our `global_index` runs 0..31,999 across four files of 8,000 rows.

For episodes 1–3 a global index is out of range and h5py raises — loud, fine. **The
dangerous case is a global index that happens to land in range**: with unequal episode
lengths, or after any subsetting, `global_index` can be a legal row number pointing at
the wrong animal in the wrong frame. No error, wrong training data.

Mitigations already in place: `local_index` is written as a dataset (upstream's
convention, directly usable), and `index_offset` is a file attribute. **Downstream code
must use `local_index`, never `global_index`, as an h5 row index.**

### 🟡 `compress_data` destroys our attributes

`session.py:1043` copies datasets only:

```python
for key, data in original_file.items():
    compressed_file.create_dataset(key, data=data, compression="gzip" if "image" in key else None)
```

`file.attrs` are not copied. Running upstream's compression silently drops
`index_offset`, `crossing_threshold`, node names — everything provenance-carrying. The
datasets survive, which is why `local_index` was written as a dataset rather than left
implied by the attribute. Any attribute we come to depend on needs the same treatment,
or `compress_data` needs patching.

Also note `"image" in key` gzips `id_images` only. Our four extra datasets survive
uncompressed, which is correct. Current size is 38 MB/episode, 152 MB total.

### 🟡 Upstream's images are worse than ours in two ways

- **Orientation is 180°-ambiguous.** `Blob.orientation` (`blob.py:150`) is the
  second-moment axis, `0.5·atan2(b, a−c)` ∈ [−π/2, π/2] — an *axis*, not a direction.
  Upstream then rotates by `orientation·180/π − 45` (`blob.py:568`; the −45 appears to
  be a legacy offset). So an upstream id-image may be head-up or tail-up, and the CNN
  must learn to absorb the flip. Ours uses the head keypoint, so heading is exact.
- **The crop is corner-cut, not centred.** `return id_img[-img_size:, -img_size:]`
  takes the bottom-right corner of the warped canvas. The commented-out "proposed
  future method" right below it searches for the origin maximising non-zero content,
  which tells you upstream knows this is wrong. Ours places the centroid analytically.

Both differences make our images *more* canonical. Flagged because they change the
input distribution the downstream contrastive loss was tuned against — better inputs
are still different inputs.

### 🟡 No mask dilation

Upstream dilates the blob mask by 3×3 once before masking (`blob.py:527`), recovering
a 1px rim the thresholder eroded. We have no thresholder and a convex hull that already
over-covers, so no dilation is applied. `dilate=0` remains an open placeholder.

---

## Stage 3c — `is_overlapping` + the identity graph (`build_overlaps.py`) — WORKING

Adds three per-row datasets to the existing id-image files. Runs *after* `build_id_images.py`
and edits its output in place (`r+`), so the 32,000 images are not re-rendered.

| dataset | dtype | meaning |
|---|---|---|
| `is_overlapping` | bool | `True` = this instance is confusable with another |
| `next_global_index` | int64 | `global_index` of its successor, `-1` if none |
| `centroid` | float64 (n,2) | `CENTROID_NODE` position, video coordinates |

Plus six attributes: `overlap_metric`, `overlap_nodes`, `overlap_threshold`,
`overlap_direction`, `overlap_gap_scale`, `overlap_max_frame_gap`.

> **Renamed from `is_an_individual` / `build_individuals.py`.** The old name was misleading:
> upstream's question is vacuous here (below), and what the flag actually measures is
> confusability. **The polarity is inverted** — `is_an_individual == not is_overlapping`. The flip
> is applied exactly once, at the boundary where upstream `Fragment` objects are constructed, so
> idtracker.ai keeps its own convention and needs no edit.

### What the flag means here vs. upstream

Upstream's `is_an_individual` answers "does this blob hold exactly one animal?" — a segmentation
question, because thresholded blobs merge on contact. SLEAP already answers that: one instance is
one animal, always. So the upstream question is **vacuous** for this port, and the flag would be a
constant `True` if ported literally.

The question that is still live is whether an instance can be **linked to the next frame** without
risk of a swap. That is what this computes. Downstream contract is unchanged (exclude from
individual fragments), so fragmentation consumes it as-is.

### The algorithm

For each consecutive frame pair, build `S[i,j]` = bbox IoU, cost `C = 1 - S`, solve the Hungarian.
A plain solve stops there and reports only the winner — it cannot distinguish a decisive assignment
from a coin-flip. So for every off-assignment cell `(i,j)` the problem is re-solved with `i→j`
forced; if that alternative costs within `SIMILARITY_THRESHOLD` of the optimum, **every animal whose
partner changes** is flagged, not just `i` and `j` — two permutations differ by a *cycle*, and
everyone on the cycle is mutually confusable.

`iou_matrix` / `poses_to_bboxes` are imported from `assignment_margin.py`, where they were already
written to match `sleap_nn.tracking.utils.compute_iou` / `get_bbox` exactly, including the `+1`
inclusive-pixel convention. So the threshold is literally on SLEAP's own bbox-similarity scale, not
a re-derivation of it.

### Two non-obvious correctness problems, both found by testing

**1. Tied optima were being pruned as "worse than optimal".** Two genuinely tied assignments — the
most ambiguous case that exists — produce a gap of exactly 0, which floating-point summation
returns as ~1e-16. A bare `>` then discarded them. Fixed with `TIE_TOL = 1e-9` on both comparisons.
Caught by a randomised brute-force cross-check, not by inspection.

**2. `per_edge` normalization is a ratio objective, which the Hungarian does not minimise.**
Minimising *total* cost through a forced cell does not minimise *cost per changed edge*. The first
implementation silently under-flagged (1186/1200 vs brute force). Fixed exactly, at no extra cost,
by shifting the matrix before solving: subtract `threshold` from every off-assignment cell, so for
any permutation π

    cost_Q(π) − opt = (cost_P(π) − opt) − threshold · n_changed(π)

since π pays the discount once per animal it re-partners. Then `cost_Q(π) ≤ opt` is *precisely*
`(cost_P(π) − opt)/n_changed(π) ≤ threshold`. One solve per cell, no iterative ratio search.

### ⚠️ The scale trap in the threshold

A permutation can never change exactly one pairing — the difference between two permutations is a
cycle of length ≥ 2. So a **total** cost gap lives on ~[0, 2], not IoU's [0, 1]. Two flies each
matching themselves at IoU 0.99 with no cross-overlap swap at a total cost of 1.98.

`GAP_SCALE` selects the scale. `"per_edge"` (default) divides by the number of
re-partnered animals, putting the gap back on IoU's own [0, 1] scale — which is the scale 0.3 was
specified on. `"total"` is the raw sum, on which 0.3 is ~6× stricter. `per_edge` also handles long
cycles correctly: 16 animals each shifting one place is genuinely ambiguous, and per-animal cost
catches it where a large total would not.

### Efficiency: an exact prune, not a heuristic

Naively this is n² Hungarian solves per frame pair. Removing a column can only make an assignment
problem more expensive, so `opt(rows≠i, cols≠j) ≥ opt(rows≠i, all cols) =: L[i]`, giving the valid
lower bound `gap(i,j) ≥ C[i,j] + L[i] − opt`. `L` costs n solves; any cell whose bound already
exceeds the limit is skipped without solving. On this dataset that removed >99% of the work —
1999 frame pairs in **2 s**.

The cost matrix is padded to square with `UNMATCHED_COST = 1.0` first. Without that, unequal
instance counts between frames would let removing a column *lower* the cost by leaving a row
unassigned, breaking the bound.

### Result: 32,000/32,000 are individuals, with 2.6× headroom

| per-edge gap | min | p1 | median | max |
|---|---|---|---|---|
| cheapest alternative | **0.7856** | 0.8743 | 0.9535 | 0.9943 |

Sweep: 0 flagged at every threshold up to 0.75; 100% at 1.0 (per-edge gap ≈ matched IoU ≈ 0.99, so
a threshold of 1.0 exceeds everything). The 0.3 threshold sits 2.6× below the smallest alternative
found anywhere in the movie.

Why it is this clean: matched-pair IoU has **min 0.687, median 0.992, never 0**, while the best
competing IoU maxes at **0.160**. The flies are far apart relative to their per-frame displacement,
so the assignment is never close.

### Validated three ways

1. **Against brute force.** `assignment_ambiguity` was cross-checked against exhaustive permutation
   enumeration on 3,000 random matrices (both normalizations, sizes 1–6, including deliberately
   tie-heavy integer-valued matrices). 3000/3000 agree. This is what caught both bugs above.
2. **Independent recomputation of the real data.** A separate script exhaustively enumerated every
   2-cycle and every 3-cycle of the optimal assignment for all 1999 frame pairs, scoring per-edge
   gap straight from the IoU matrix — no Hungarian sub-solves, no λ-shift, no pruning. Minimum found:
   0.7856 (2-cycles), 0.8374 (3-cycles). Agrees with the stored flags.
3. **Every instance labelled exactly once.** The `(frame, instance_idx)` keys across all four files
   form an exact bijection with the 32,000 instances in the `.slp` — no duplicates, no gaps, no
   unwritten rows. All 8 datasets in each file are the same length.

### Nothing pre-existing was disturbed

Files are opened `r+`, and `snapshot()` SHA-1s every pre-existing dataset and `repr()`s every
attribute before and after the write, raising if anything differs. Re-verified externally afterwards:
`local_index == arange`, `global_index − index_offset == local_index`, frames inside their episode
bounds, identities all zero, crossings all False, exactly 16 rows per frame.

One consequence of re-opening rather than rewriting: a knob that gets **renamed or dropped** leaves
its old attribute behind, still describing settings that are no longer in force. Caught in practice
when `individual_normalization` survived alongside its replacement `individual_gap_scale`. The write
now clears the whole `overlap_` namespace before repopulating it, so what is on disk is exactly
what produced the flags, and reports which stale keys it dropped. Attributes owned by earlier stages
(`crossing_*`, `top_node`, …) are outside that namespace and untouched.

⚠️ **That namespace sweep cannot catch a change to the prefix itself**, which bit immediately on the
`is_an_individual` → `is_overlapping` rename: the old dataset and all six `individual_*` attributes
survived the rewrite. Not merely untidy — a leftover `is_an_individual` beside `is_overlapping`
carries the **opposite polarity**, so a reader that picks the wrong one gets exactly inverted
semantics with no error. `drop_stale()` now removes named legacy datasets/prefixes explicitly, and
also drops `fragment_identifier` / `n_fragments`, which are computed *from* the links and are
invalidated whenever the links are rewritten. It runs before the digest is taken, so removals are
not misreported as modifications.

### ⚠️ Open decision — directionality was my default, not your spec

`INDIVIDUAL_DIRECTION = "both"` means an instance must have *both* its incoming and outgoing link
unambiguous. That is the conservative reading and was not specified; `"forward"` and `"backward"`
are implemented. Inert on this dataset (nothing is flagged either way).

Also: a **missing** neighbour (movie start/end, or a gap wider than `MAX_FRAME_GAP`) contributes no
evidence and therefore never flags. Absence of a link is not the same as an ambiguous one. The 16
instances in the final frame consequently have no forward margin, which is why the sweep denominator
is 31,984 rather than 32,000 — they are still labelled, via their backward link.

---

## Stage 3d — `Fragment` objects (`build_fragments.py`) — WORKING

Walks the `next_global_index` graph into fragments and constructs **upstream's own `Fragment`
class**, unmodified (`idtrackerai/fragment.py:137`). Output is a real `ListOfFragments`, saved via
upstream's own serializer to `session_fourfly/list_of_fragments.json` (384 kB).

### The graph is built in the same pass as the cost matrix

Per spec, linking is not a second sweep. `assignment_ambiguity` now returns `sigma` alongside the
flags, because σ *is* both things at once: the correspondence being used as the link, and the
assignment whose robustness the ambiguity test measures. One solve, two outputs.

σ only indexes `0..n-1` within a frame and is globally meaningless on its own, so it is resolved
through the frame it came from: instance `i` of frame `t` → instance `sigma[i]` of frame `t+1`, and
that `(frame, instance)` key maps to a unique `global_index`. Centroids are stored alongside so the
pairing can be replayed in video coordinates.

Per spec, a link is kept **only when both endpoints agree on `is_overlapping`**, so a fragment can
never span a change in the flag. This mirrors upstream's chain condition (`fragmentation.py:68`).
It is applied at write time rather than inside the pass, because `is_overlapping` is not final until
`resolve()` has combined the forward and backward directions.

### Our graph cannot branch; upstream's can

Upstream's `blob.next` is *every* next-frame blob whose pixels overlap, so a blob may have several
successors and `compute_fragment_identifier` must stop wherever the chain is not 1-to-1. Ours comes
from a Hungarian assignment, which is a **permutation** — at most one successor and at most one
predecessor per instance. Fragment extraction is therefore "start at every node with no predecessor
and walk forward", with no ambiguity to resolve. `build_chains` *verifies* this rather than assuming
it: a second predecessor or a cycle raises.

### Field conventions that were checked against upstream, not guessed

| field | value | why |
|---|---|---|
| `end_frame` | last frame **+ 1** | `list_of_fragments.py:213` — `end + 1, # it is not inclusive` |
| `images` | **episode-local** row index | `load_id_images` (`py_utils.py:468`) indexes the episode's dataset directly with it; a global index would silently fetch the wrong animal |
| `episodes` | one per image | a fragment may span episode boundaries — fragment 0 spans all four |
| `is_an_individual` | `not is_overlapping` | the single polarity flip |
| `exclusive_roi` | `-1` | `Blob`'s own default (`blob.py:86`); no ROIs in this port |

### Result: the movie partitions into 16 fragments of 2000 frames

Nothing is ever confusable in this clip, so every fly yields one unbroken chain spanning the whole
movie — 16 individual fragments, 0 crossing fragments, 32,000 images. Fragment 0 spans episodes
[0,1,2,3] and travels 900 px.

### Validated, including a round-trip through upstream

Internal checks: every instance in exactly one fragment; lengths sum to 32,000; identifiers are
`0..N-1` in order (which `ListOfFragments.__init__` asserts); `end_frame - start_frame == n_images`;
no fragment repeats a frame; every frame fully covered.

Independent re-check from disk, plus two that matter more:

1. **Round-trip.** `ListOfFragments.load()` reads our JSON back and reconstructs 16 fragments,
   `n_animals=16`, 32,000 images.
2. **Upstream can actually fetch the images.** `load_id_images(files, fragment.image_locations)`
   returns `(10, 83, 57)` uint8 and matches a direct per-episode read byte for byte. This is the
   check that proves the local-vs-global index convention is right — the failure mode it rules out
   is silent and returns real-looking images of the wrong animal.
3. **Physical sanity.** Max within-fragment centroid step between frames is **4.98 px**; the closest
   two flies ever get is **68.34 px**. A wrong link would have to jump ~14× further than any fly
   actually moves.

### ⚠️ Note: 83×57 images still flow through unchanged

`load_id_images` returns them fine, but the IdCNN input-shape problem from Stage 3b is untouched and
will surface at training. Still the first genuine surgical edit this port needs.

---

## Stage 3e — Training batches from upstream code alone — WORKING, 2 DEPENDENCIES FLAGGED

Verification only. **No new code was written** — a throwaway driver calls existing idtracker.ai
functions on our HDF5 files. No model was constructed and nothing was passed to a network.

```python
lof = ListOfFragments.load("session_fourfly/list_of_fragments.json")
cl  = ContrastiveLearning(lof)                      # only required arg is our ListOfFragments
cl.train_loader.batch_sampler.n_batches = 1         # upstream's own idiom, contrastive.py:538
batch = next(iter(cl.train_loader))
```

| loader | batch shape | dtype |
|---|---|---|
| contrastive train | `(800, 1, 83, 57)` ×2 + `(800,)` | float32 / int64 |
| contrastive val | `(400, 1, 83, 57)` + `(400,)` | float32 / int8 |
| `get_onthefly_dataloader` | `(500, 1, 83, 57)` + `(500,)` | float32 / int64 |

800 = 400 negative + 400 positive pairs (`conf.CONTRASTIVE_BATCHSIZE = 400`). Pair generation
found **120 negative pairs** = C(16,2), since all 16 fragments span the whole movie and therefore
all coexist, plus 16 positive pairs. Images preloaded to RAM in 0.1 s (151 MB).

`ContrastiveLearning` needs **no `Session` object** — just the `ListOfFragments`, which carries
`id_images_file_paths` and `n_animals`. `fragment.coexisting_individual_fragments` is populated by
`ListOfFragments.__init__` → `connect_coexisting_fragments()`, which our constructor already calls.

### ✅ DEPENDENCY 1 — no single conda env can run the pipeline end to end — **RESOLVED**

| env | has | lacks |
|---|---|---|
| `sleap_id` | sleap-io, cv2, h5py, scipy | ~~torch~~ |
| `idtrackerai` | torch 2.12.0 (MPS), h5py, sklearn | **sleap-io** |

Stages 1–3d ran in `sleap_id`; the batch check had to run in `idtrackerai`. Fixed by
`pip install torch torchvision scikit-learn psutil` into `sleap_id` → **torch 2.13.0,
torchvision 0.28.0, MPS available**, alongside sleap-io 0.7.1. One env now holds both halves.

pip warns that idtrackerai's `pyqt6`, `qtpy`, `superqt` and `opencv-python-headless` are absent.
Left absent deliberately — they are GUI dependencies (`segmentation_app/`, `GUI_tools/`) and the
headless pipeline never imports them. Revisit only if the idtracker GUI is ever needed.

### ✅ DEPENDENCY 2 — `ListOfGlobalFragments` was never built — **RESOLVED in Stage 3f**

Upstream builds it in `fragmentation_API` (`fragmentation.py:41`) right after `ListOfFragments`.
We skipped it. Consequences, in order of severity:

- Batch creation is **unaffected** — `first_gfrag=None` is handled, `gfrag_loader = None`, and
  clustering falls back to k-means++ (`contrastive.py:434`).
- `tracker_API` treats it as a degraded mode: with no Global Fragments it logs *"We will not run
  the accumulation protocol"* and returns the contrastive identifier early (`tracker.py:110-116`).

So tracking would run contrastive-only. Whether that is acceptable or whether Stage 3f should build
`ListOfGlobalFragments` is an open decision.

### Corrected: IdCNN is not a blocker

The Stage 3b 🔴 on `IdCNN`'s square-input assumption was overstated. The contrastive identity model
is `ResNet18`, which adaptive-pools to 1×1 before `fc` and accepts 83×57 fine — confirmed by the
batch shapes above. See the revised Stage 3b entry.

### Non-issues that looked like issues

- `BatchSampler has n_batches set to None` — not a missing dependency. Upstream sets it per epoch
  in `train_step` (`contrastive.py:538`); an epoch is defined by batch count, not dataset size.
- macOS `freeze_support()` / spawn error — an artefact of the throwaway driver lacking a
  `__main__` guard, which upstream's CLI has. Not an upstream problem.

---

## Stage 3f — `ListOfGlobalFragments` (`build_global_fragments.py`) — WORKING

A *global fragment* is a stretch where every animal is simultaneously in its own individual
fragment — the only stretches where identity can be learned without ambiguity, and what the
accumulation protocol trains on.

### The user-configurable number keeps idtracker.ai's own name

```python
MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION = 4   # confparams.py:26, same name and default
```

It is applied through **upstream's own setter** —
`conf.set_parameters(MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION=...)` — so the code that
consumes it is idtracker.ai's untouched filter in `ListOfGlobalFragments.__init__`
(`list_of_global_fragments.py:72`), which splits `global_fragments` from
`non_accumulable_global_fragments`. Verified to be genuinely wired, not decorative:

| value | accumulable | non-accumulable |
|---|---|---|
| 4 / 500 / 2000 | 1 | 0 |
| 2001 / 5000 | 0 | 1 |

Flips exactly at 2001, our fragments being 2000 images long.

### Only the core detection is ours; the classes are upstream's

`ListOfGlobalFragments.from_fragments` takes `blobs_in_video: list[list[Blob]]` — and this port
has no blobs, which is its whole premise. So that one classmethod is unusable. Everything else is
called unmodified: `GlobalFragment(fragments)`, `ListOfGlobalFragments(gfs)`, and `.save()` in
upstream's own JSON format. Only `get_global_fragment_core`
(`list_of_global_fragments.py:221`) is re-expressed against our arrays, condition for condition.

### "Just count the individuals" is most of it, but not all of it

`is_clean_frame` really is only *"does this frame hold `n_animals` individuals"*. Two extra pieces
turn clean frames into global fragments, and measurement shows they are not equally important here:

- **The seed is load-bearing.** Upstream forces index 0 to `False` (`[False] + [...]`) so a core
  already running at frame 0 still produces a `[False, True]` transition. Without it this movie —
  clean in every one of its 2000 frames — yields **0 global fragments** instead of 1. Caught by
  running both versions, not by reading.
- **The `same_fragment_identifiers` clause is inert here.** It exists to split a clean run at a
  *fragment turnover* (one animal's individual fragment ending and another beginning while every
  frame stays clean), which would otherwise collapse into one global fragment holding a stale set
  of `Fragment` references. No turnover happens in this clip, so with the seed applied to both
  sides the clause changes nothing: 1 core either way. It will matter on footage with crossings.

### Result and validation

**1 global fragment**, accumulable, holding all 16 fragments, min 2000 images each, min distance
travelled 73 px. Every frame is clean (2000/2000), 1999 core frames, 1 distinct core.

Checks: every global fragment holds exactly `n_animals` fragments with no duplicates; all members
are individuals; the accumulable/non-accumulable split matches the threshold on both sides; every
member fragment actually spans the core frame; round-trips through
`ListOfGlobalFragments.load(path, fragments)`.

---

## Stage 3g — Non-square input end to end — WORKING (first surgical edit to `idtrackerai/`)

Question asked: is the integration bound to square images, and can it just be x-median × y-median?
Answered by running tensors through both models rather than by reading the source.

| model | 83×57 | 57×83 | 200×37 | square | used by |
|---|---|---|---|---|---|
| `ResNet18` | ✅ | ✅ | ✅ | ✅ | contrastive identity (the path we use) |
| `IdCNN` *(before)* | ❌ | ❌ | — | ✅ | crossing detector, accumulation cascade |
| `IdCNN` *(after)* | ✅ | ✅ | ✅ | ✅ | — |

**`ResNet18` was never square-bound.** It inherits torchvision's `ResNet`, whose
`nn.AdaptiveAvgPool2d((1,1))` collapses spatial dimensions before `fc`, so `fc` sees 512 channels
regardless of H×W. Verified working from 8×8 upward. The identity path has therefore been running
at x-median × y-median all along — id-images are 57 × 83 and batches flow as `(B, 1, 83, 57)`.

Also checked: the training augmentation `np.rot90(images, 2, axes=(1,2))` (`train.py:268`) is a
180° rotation and preserves non-square shape. `k=1` would have transposed it and broken everything;
upstream happens to use `k=2`.

### The one real blocker, and why it stopped being hypothetical

`IdCNN` squared the flatten (`models.py:68`). Generalised to `H//4 * W//4` — see
`idtrackerai/PROVENANCE.md` for the diff and rationale. Square behaviour is bit-identical
(80×80 → 40000, 52×52 → 16900, 100×100 → 62500 all match the original formula), so pretrained
square models still load.

⚠️ **Stage 3f made this live.** `IdCNN` is only reached by the supervised accumulation cascade
(`tracker.py:176/192`), which `tracker_API` skips entirely when there are no Global Fragments
(`tracker.py:116` returns the contrastive identifier early). Building Global Fragments removed that
escape: the cascade now runs whenever contrastive accumulates less than
`conf.CONTRASTIVE_MIN_ACCUMULATION` (0.5). So a latent crash became a reachable one, and fixing it
became necessary rather than tidy.

### `Session` was not edited

`set_id_image_size` derives `[max_size, max_size, 1]` (`session.py:543`) — but only on the auto
path, guarded by `if not self.id_image_size:` (line 539). An explicitly supplied `[83, 57, 1]` is
respected, so nothing needs patching. This also corrects the Stage 3b note, which cited that line
as reinforcing a hard square constraint; it is a default, not a constraint.

---

## Stage 3h — Forward-pass validation on real pairs — WORKING

Real id-image pairs pushed through both models. Mirrors `ContrastiveLearning.train_step`
(`contrastive.py:551-562`) exactly, but with `eval()` + `no_grad` and **no optimizer step** — this
validates plumbing and output shape, it does not train.

### Contrastive path (`ResNet18`, 11,174,336 params, embedding dim 8)

| tensor | shape | finite |
|---|---|---|
| `images_A` / `images_B` | `(800, 1, 83, 57)` float32, range [0, 0.714] | ✅ |
| `embedded_A` / `embedded_B` | `(800, 8)` float32 | ✅ |
| `criterion(...)` | `(800,)`, mean 49.64 | ✅ |

800 = 400 negative + 400 positive pairs. Eval-mode forward is deterministic (same input twice →
bit-identical output).

**The loss decomposes exactly as theory predicts at initialisation.** `criterion` is
`relu(concat(margin − d_neg, d_pos − 1)).square()` with `margin=10`:

- negatives: `(10 − 0.0362)² = 99.28`
- positives: `relu(0.0194 − 1) = 0`
- mean = `(99.28·400 + 0·400)/800 = 49.64` — matching the measured 49.6389 to 4 decimals.

So at init *all* loss comes from negative pairs sitting too close (a random net collapses
everything near the origin) and positive pairs already satisfy their target. That is the correct
starting state for contrastive learning, and the arithmetic agreeing is a stronger check than the
tensors merely being finite.

### The pair semantics are verifiably right

Untrained, positive pairs embed **1.9× closer** than negatives (0.0194 vs 0.0362, sd ≈ 0.010).
This is *not* evidence the model learned anything — it is a random projection. It is evidence the
**pairing is semantically correct**: `PairsOfFragments.__getitem__` draws a uniformly random frame
from each fragment (`contrastive.py:90`), so a positive pair is the same fly at two arbitrary times
out of 2000, not two adjacent near-duplicate frames. Had positives been mislabelled or mixed with
negatives, the two distributions would coincide. They do not.

### Patched `IdCNN` on the same real images

2,987,896 params, `flatten = 28000` = `100 · (83//4) · (57//4)` = `100 · 20 · 14` ✅.
Output `(64, 16)` logits, softmax rows sum to 1, predictions spread over classes 2–12 of 0–15.
This is the surgical edit from Stage 3g exercised on **real** non-square data rather than zeros.

---

## Stage 3i — Shared config + the no-global-fragment path — WORKING

Two changes, one prompted by the other.

### `config.py` — every user-set knob in one file

Each stage carried its own constants, and two of them were being imported *between* stages
(`build_overlaps` pulled `BODY_NODES` from `build_id_images` and `CENTROID_NODE` from
`rotate_boxes`). All knobs now live in `src/sleap_idtracker/config.py`; each stage binds its old
names from it, so no other line in any stage changed. `config.py` imports nothing from the
pipeline, so there is no cycle. Derived quantities (id-image size, episode boundaries, fragment
count) are deliberately *not* in it — they are computed and reported by the stage that owns them.

Verified: every stage constant is identical to its `config.py` source, and `SLP` / `OUT_DIR` are
now shared objects rather than three copies of the same string.

### `N_ANIMALS` is the identity count, and nothing else

It is `k`. It flows `config.N_ANIMALS` → `list_of_fragments.json` → `ListOfFragments.n_animals` →
`ContrastiveLearning.n_animals` (`contrastive.py:261`) → `MiniBatchKMeans(self.n_animals, …)`
(`contrastive.py:744`, `:797`). Set to **16** for this clip.

It is **not** validated against how many instances the tracker detected. Detecting fewer is
ordinary occlusion; detecting more is spurious instances; neither changes how many real animals
there are. The two counts are printed side by side so a disagreement is visible, never enforced.

### The no-global-fragment case

**Finding: no downstream edit was needed.** idtracker.ai already does exactly the right thing —
`first_gfrag=None` drops the global-fragment K-Means seeding (`contrastive.py:435`), `kmeans_init`
returns `{"n_init": 20, "init": "k-means++"}` (`contrastive.py:854`), and `contrastive_step`
returns a ratio of `inf` (`tracker.py:276`) which makes the caller skip the accumulation protocol
outright (`tracker.py:110`). Nothing in `idtrackerai/` or `vendor/` was touched.

What *was* wrong was the reporting. Every structural check in `build_global_fragments.py` has the
form `all(… for g in allgf)`, and `all()` of an empty iterable is `True` — so a video with no
global fragments printed a full row of `PASS` and the `ALL GLOBAL FRAGMENT CHECKS PASS` headline
having examined nothing, over an empty JSON. Now:

- the empty case branches out and is validated on its own terms (`report_kmeans_fallback`);
- so does the "global fragments exist but none is long enough to accumulate on" case, which is
  indistinguishable downstream because `tracker.py:69` tests the *accumulable* list;
- both end on `… IDENTIFICATION WILL USE K-MEANS++ WITH k=N` instead of the success headline.

Exercised by forcing `N_ANIMALS = 17` so no frame can be clean: 0 cores, 0 accumulable, empty set
round-trips, fallback preconditions all pass, correct headline. Restored to 16 afterwards.

### Two hard requirements, on different axes

`N_ANIMALS` identities get assigned over the course of the video. That yields two conditions,
easy to conflate and not the same:

**1. All `N_ANIMALS` must APPEAR — enforced.** They do not have to coexist, but each must turn up
somewhere. Checked via individual fragments: a fragment is one animal's continuous track, so
distinct animals cannot share one, and N animals therefore produce at least N individual
fragments. Fewer than `N_ANIMALS` in the whole video is positive proof some declared animal is
never seen, and `build_fragments.py` raises. The converse does not hold — one repeatedly occluded
animal yields many fragments — so this is a floor, not a count of animals.

**2. Some two must COEXIST — enforced.** Contrastive learns "different" only from animals seen at
the same time, and raises outright with no co-occurring pair (`contrastive.py:291`). So the
maximum number of individuals alive in any single frame must be `>= 2`. Deliberately a maximum
rather than a total or a fraction: brief stretches of two animals together are enough.

**Not required: all of them at once.** That is the global-fragment condition, and its absence is
handled by the k-means++ path above rather than being an error.

Both checks run **before** `write_identifiers`, so a rejected run leaves the id-image files exactly
as `build_overlaps.py` left them instead of half-stamped with identifiers from a rejected
configuration. Found by observing the first version raise *after* the HDF5 writes.

This clip: 16 individual fragments for `N_ANIMALS=16`, 2000 frames hold 2+ individuals, max 16
together, connectivity 0.94.

Verified in both directions: `N_ANIMALS=14` passes (14 ≤ 16 animals appear); `N_ANIMALS=17` raises
before writing, naming the 1 animal never seen. Because that no longer reaches the
empty-global-fragment branch, that branch was re-verified the way it would actually arise — all 16
animals appearing but one always mid-overlap, so no frame is ever clean: 0 cores, connectivity
1.00, correct `k=16` k-means++ headline.

---

## Stage 4 — identities for every instance (`build_identities.py`) — WORKING

First end-to-end MVP: every instance in the video carries an identity, and trajectories are
written in idtracker.ai's own format. Scoped deliberately: **no** accumulation protocol, **no** P2
exclusion cascade, **no** impossible-velocity correction, **no** crossing interpolation.

### Getting a crossing to exist at all

The clip has **no overlapping instances** at `SIMILARITY_THRESHOLD = 0.3` — 0/32000, because the
smallest per-edge gap anywhere is 0.7856. So the crossing machinery could not be tested.

Raised the threshold to **0.80** (and `REPORT_CEILING` 0.5 → 1.0, since the invariant is
`REPORT_CEILING >= SIMILARITY_THRESHOLD`). That flags **4 instances**: animals 4 and 9, mutually
ambiguous across frames 571→572 — a genuine 2-cycle.

**Caveat, recorded honestly:** those two flies are ~1000 px apart. This is *assignment-cost*
ambiguity, not physical occlusion — the least-confident link in a clip that contains no real
crossings. The code path is exercised, not stress-tested. A clip with real occlusions is still
needed before trusting the crossing behaviour.

### Two populations, two identity mechanisms

| | training | identity decided |
| --- | --- | --- |
| individual fragments | yes | fragment-level, pooled over all images (P1 argmax) |
| crossing fragments | **excluded** | per-ROW, nearest cluster centre, one forward pass each |

`ContrastiveLearning.predict` already restricts itself to `frag.is_an_individual`
(contrastive.py:767), so crossing rows are skipped from training for free — no filter of ours
needed. Crossing rows are then pushed through `IdentifierContrastive.forward` (models.py:260)
one at a time, with preprocessing copied exactly from `collate_fun` (`dtype=np.float32`, then
`unsqueeze(1) / 255`) — a mismatch there would silently feed differently-scaled images than the
model trained on.

### Crossing fragments now merge (structural change to `build_fragments.py`)

`merge_crossing_chains`: crossing chains whose frame spans intersect fuse **transitively** into one
node group — the "one big fragment" holding every animal in the crossing. 20 chains → 19 groups.

This breaks two invariants *by design*, since a merged group holds several rows per frame:
`end_frame - start_frame == n_images` and one-row-per-frame now apply to **individual fragments
only**. Two new checks replace them: every row of a crossing fragment is flagged overlapping, and
a merged group absorbs *every* flagged row in its span (otherwise two mutually-ambiguous animals
ended up in different fragments, which is the thing merging exists to prevent).

Side effect: the crossing splits the movie into **2 global fragments** (0–570, 573–1999).

### DECISION: skip the cluster reorder (`reference_gfrag=None`)

`predict()` optionally renumbers the k-means clusters onto a "canonical" identity ordering by
Hungarian-matching them against a reference global fragment (contrastive.py:811). **We do not do
this.**

Why: k-means labels are arbitrary, and the reorder pins them to an external anchor. But that
anchor only means something when it comes from somewhere real — a knowledge-transfer folder
carrying identities from a *previously tracked session*, or exclusive ROIs. With neither, upstream
falls back to `identities = np.arange(n_animals)` (identity_transfer.py:42), i.e. *"identity k =
the k-th fragment in the first global fragment"* — which recovers no information and merely picks
a convention.

It also requires `temporary_id` on every member of that global fragment, set by the
accumulation-protocol step this port skips, so passing it raises `assert frag.temporary_id is not
None`. That assertion is what surfaced the choice.

**Consequence:** identities are arbitrary but internally consistent — fly "3" is the same fly
throughout the video, just not tied to anything outside it. That is the correct guarantee for an
unsupervised MVP. Revisit when identities must be stable ACROSS videos; the reorder is the hook.

### Results

Trained from scratch to **silhouette 0.9632** (above idtracker's own 0.91 target) in ~15 min on
MPS. Then:

- 18/18 individual fragments identified
- 4 crossing rows predicted individually → identities **[5, 10, 5, 10]** — instance 4 → id 5 and
  instance 9 → id 10, consistently in both frames, no collision
- **32000/32000** instances assigned, **16/16** identities used
- **exactly 2000 rows per identity** (32000/16), a perfectly even split

Written into the existing `identities` dataset, which build_id_images.py created zero-filled
precisely so upstream's `require_dataset` finds a matching shape. 0 remains "unassigned".

---

## Stage 5 — trajectories (`build_trajectories.py`) — RETIRED, WAS WORKING

**Out of the pipeline.** The readout is the SLEAP GUI: Stage 6 writes identities into a
`.idtracker_predictions.slp` as Tracks, which is the same information in the format the GUI already
opens. idtracker.ai's `trajectories/` folder was *its* output format and has no consumer here.

Retired in place rather than deleted, per the repo convention — the `__main__` entry point is
commented out with a `# SLEAP-PORT:` marker so it cannot be run by accident, and the module carries
a RETIRED banner. `produce_output_dict` is kept because it is the only written-down mapping from
this port's HDF5 columns onto upstream's output dict, which is where to start if trajectories are
ever needed to feed an idtracker.ai-native tool. Its collision rule outlived it: Stage 6 reuses that
decision for tracks rather than making a second one.

The already-generated `session_fourfly/trajectories/` was left on disk; retiring the stage does not
require deleting its output.

What it did, for the record:

Port of `produce_output_dict` (trajectories_creation.py:102) with the data source swapped. Upstream
walks `blobs_in_video` and does `trajectories[blob.frame_number, identity - 1] = centroid`; every
quantity that loop reads is already an HDF5 column here, so the same array is filled straight from
columns. The output dict has upstream's keys and shapes and is handed to upstream's own
`save_trajectories` unmodified.

**Collision policy (decision): colliding cells are left NaN.** Crossing rows are predicted
independently, so two rows in one frame can claim one identity — upstream never faces this because
its greedy P2 cascade enforces mutual exclusion. Writing one of the two would invent a resolution
the evidence does not support. Collisions are counted and reported. Unit-tested on synthetic
columns: a frame where two rows both claim identity 1 leaves both NaN, 1 collision / 2 rows dropped.

**Not available in this port:** `areas` (upstream reads `blob.area`, a segmentation-mask pixel
count; there is no mask here — written as NaN, hull area would be a reasonable substitute) and
`id_probabilities` (no per-row confidence persisted yet).

### Results

`trajectories (2000, 16, 2)`, **32000/32000 cells assigned, 0 collisions, fraction_identified
1.0000**. Wrote `trajectories.npy`, `trajectories.h5`, and `trajectories_csv/`
(trajectories.csv, id_probabilities.csv, areas.csv, attributes.json) — the same folder layout
idtracker.ai produces. Both binary formats round-trip and match the in-memory array.

**Continuity check** (the real test of whether identities are right, not just plumbed): per-frame
displacement median **0.06 px**, p99 2.66, max **4.98** — and **zero** jumps over 50 px. Across the
crossing frames, max displacement is 2.2–2.5 px. No identity teleports.

---

## Bugs found and fixed while building Stage 4-5

- **`snapshot()` digested its own namespace** (`build_overlaps.py`). It hashed every attribute
  including the `overlap_*` ones this stage owns and rewrites, so it passed only while the config
  never changed. The first edit to `SIMILARITY_THRESHOLD` turned a correct rewrite into a spurious
  "pre-existing data was modified" failure. Now skips `ATTR_PREFIX`, matching how it already
  skipped its own datasets.
- **DataLoader deadlock on macOS.** The stage hung indefinitely at ~4% CPU. macOS **spawns**
  workers instead of forking, so each must pickle `collate_fn` — which holds the whole 151 MB
  preloaded image array — across a pipe, twice (train + val, both `persistent_workers`). Fixed
  port-side with `num_workers=0`, reusing the same `batch_sampler` object because `train_step`
  mutates its `n_batches` (contrastive.py:538). No `idtrackerai/` edit.
- **`SHORT_TRAINING` bounded nothing.** Both stop conditions in `train()` need *consecutive*
  non-improvements, and `steps_without_improvement` resets on every improvement — so a
  monotonically improving model never stops. `target_silhouette_score=0.0` only changes a log
  message. The run took ~15 min rather than the intended ~1. Left as-is since it converged to
  0.9632; `REUSE_CHECKPOINT` now skips retraining on rerun.

---

## Stage 3j — instances with no alignment keypoint — WORKING

An instance whose `TOP_NODE` or `CENTROID_NODE` is NaN cannot be rotated into the canonical pose,
so it gets **no row** in the id-image HDF5. That was already the behaviour; what was missing was
everything that has to hold around it.

**The requirement.** Drop the instance, and let every *other* instance keep the index SLEAP gave
it. `instance_idx` stays the position in `lf.instances` — the column goes **sparse in `i`**, it is
never renumbered. The `None` placeholder in `collect_geometry` (`build_id_images.py:134`) is what
makes this true and is load-bearing; the write loop's `continue` now says so.

**What actually broke.** Two things, both fatal on the *first* dropped instance:

- **`gmap[(fu, j)]` raised `KeyError`** (`build_overlaps.py`). The frame-to-frame link is resolved
  from the .slp, so `succ` can name a successor that has no row. `gmap` is sparse in `i`, so the
  lookup missed and took down the whole stage over one instance. Now `gmap.get(...)`; a successor
  with no row is treated exactly like "no partner" (`j < 0`) and the chain **ends there**.
  Counted and reported as `N of them because the successor has no row`.
- **`total != n_inst` assertion** (`build_overlaps.py`). It compared h5 rows against SLEAP
  instances, which cannot match once anything is dropped. Now `total + n_skipped == n_inst`, so
  the invariant is "nothing is lost *silently*" rather than "nothing is ever dropped".

**The drop is now recorded, not just printed.** `build_id_images.py` writes `n_skipped` and
`n_instances` per episode as HDF5 attributes. Previously the only trace was a `WARNING` line, which
made an absent row indistinguishable from an instance SLEAP never predicted.

**Not affected.** `build_fragments.py` indexes by `global_index`, which is assigned from the write
cursor and stays dense regardless of skips. `frame_crossings` already handled `None` entries, and
`similarity()` already degraded all-NaN instances to zero IoU.

**Validated** by blanking `head` on 4 instances — `(800,5)`, `(801,5)`, `(1200,0)`, `(0,15)`, chosen
so a surviving row's successor is one of the dropped ones — then running stages 3 → 3h → 3i on a
scratch session. 31996 rows, `n_skipped` sums to 4, exactly 2 links dropped for a missing successor,
fragments 19 → 21 as the chains split, all 10 fragment checks pass, and every touched frame holds
`[0..15]` minus the dropped index rather than a shifted `[0..14]`.

**Decision deliberately NOT made:** a dropped instance mid-chain *breaks* the fragment in two rather
than bridging frame `t` to `t+2`. Breaking is consistent with the existing `MAX_FRAME_GAP` semantics
and with `j < 0`; bridging would be a new policy and is left open.

**Still open:** the guard catches *missing*, not *degenerate*. If `TOP_NODE` and `CENTROID_NODE`
landed on the same pixel, `arctan2(0, 0)` returns `0.0` — a valid 0° matrix, so the crop silently
comes out unrotated with no `None`, no warning and no `n_skipped` increment. Head–thorax distance in
the current clip runs 19.3–32.2 px, so it does not arise here.

**Not re-run:** `session_fourfly/` predates these attributes. Regenerating it means `build_id_images`
deleting and recreating the HDF5 files, which wipes the `identities` column — so it was left alone.

---

## Stage 3-2 — id-image area clamp (`config.MAX_CROP_AREA`) — WORKING

The canonical frame is `SIZE_STAT(x_spread) x SIZE_STAT(y_extent)`, set by how big the animals are on
screen, and nothing bounded it. A close-up recording or a larger species produces crops that blow up
the id-image HDF5 and the training batches with it — cost is linear, so at 5.76M instances every
1000 px of crop area is ~5.8 GB on disk.

`MAX_CROP_AREA = 6400` (80x80 equivalent). If `W*H` exceeds it, **both** dimensions are scaled by the
same `sqrt(MAX_CROP_AREA / (W*H))`, so the animals keep their shape. `None` disables the clamp.

**The scale folds into the affine `canonical_crop` already applies** rather than being a second pass.
`A` maps `p -> A @ [p,1]`, so `s * (A @ [p,1]) == (s*A) @ [p,1]` — multiplying every entry,
translation included, scales the output. `warpAffine` then resamples straight into the reduced frame:
one interpolation, not a warp followed by a resize, and no extra pass over the data.

**Rounding.** Round first (tracks the aspect ratio more closely than truncation), but rounding *up*
can push the product back over the bound, so fall back to floor in that case —
`floor(W*s) * floor(H*s) <= W*H*s*s == MAX_CROP_AREA` exactly, so the second attempt always fits.

**One interaction this would have broken silently.** Two places read the `height` attribute *as the
animal's body length*: `body_length_from_h5` (live — it is the scale for the centroid similarity
added in Stage 3c-2) and the retired `build_trajectories.py`. Once the clamp engages, `height` is the
reduced crop's own dimension while the centroids being compared are still in video pixels, so the
centroid metric would have been scaled by the wrong number with no error. Fixed by writing a separate
`body_length` attribute that stays in video pixels; both readers now prefer it and fall back to
`height` for files written before the clamp existed, where the two were the same number.

New attributes: `body_length`, `crop_width_raw`, `crop_scale`, `max_crop_area`.

### Results

**Does not engage on this clip** — 57x83 = 4731 px, under 6400, `scale = 1.0`, existing sessions
unaffected.

**Validated** three ways. A sweep of bounds (`None, 100000, 6400, 4731, 4000, 2000, 900, 100`): area
never exceeds the cap and the aspect ratio stays within 5% of 0.687 at every one, including the two
rounding edge cases — 900 lands exactly on 900 (25x36), and 100 falls back to floor for 96 (8x12).
Content: a clamped crop compared against `cv2.resize` of the unclamped one differs by a mean of
**4.82/255**, which is the INTER_LINEAR-vs-INTER_AREA difference — a misalignment would be far
larger. And `build_id_images.py` run end to end at `MAX_CROP_AREA=2000`.

---

## Stage 3c-2 — configurable similarity method — WORKING

`config.SIMILARITY_METHOD` now selects which similarity builds the frame-to-frame cost matrix.
Three options, all ports of the SLEAP tracker's own scoring, all living in `assignment_margin.py`
beside the existing ones:

| method | feature | similarity | upstream |
|---|---|---|---|
| `bounding_box` | `poses_to_bboxes` | `iou_matrix` | `compute_iou` / `get_bbox`, `+1` convention |
| `keypoint` | raw poses | `oks_matrix` | `compute_oks`, cocoeval normalization |
| `centroid` | `poses_to_centroids` | `centroid_matrix` | `get_centroid`; similarity is **not** upstream's |

`oks_matrix` already existed and was unused — only the wiring was missing.

**The centroid similarity could not be ported as-is.** Upstream's `compute_euclidean_distance`
returns a raw NEGATIVE distance: a score where higher is better, but unbounded below. The solver
needs `S` in [0, 1], because the cost is `1 - S` and has to stay finite and non-negative against
`UNMATCHED_COST` padding. **Decision (asked, not assumed):** map through `exp(-d / body_length)`,
so the unit is body lengths travelled — `d=0 -> 1.0`, one body length `-> 0.368`, two `-> 0.135`.
Never reaches 0, so no pair is hard-excluded and the solver sees no discontinuity. Scaling by a
measured quantity rather than a pixel constant means it transfers to other animals without retuning.
The scale is read from the id-image files' `height` attr, so it is the *same* number the crops were
sized with rather than a subtly different recomputation.

Two centroids are now in play and must not be conflated: the *method* uses upstream's `get_centroid`
(nanmedian over `OVERLAP_NODES`), while the h5 `centroid` column and the crop centring still use
`CENTROID_NODE` (thorax) regardless of method.

**`SIMILARITY_THRESHOLD` is now a dict keyed by method**, because the three do not share a scale.
Measured minimum per-edge gap on this clip — the value a threshold must exceed before anything is
flagged at all:

    bounding_box   min 0.7856   median 0.9535
    keypoint       min 0.5820   median 0.9110
    centroid       min 0.4908   median 0.8482

Roughly 0.3 apart, which is the concrete argument for per-method values. At threshold 0.9 the
methods diverge hard: bbox flags 9.11% of instances, centroid flags 100%.

**Thresholds are set to 0.2 for all three, by explicit instruction.** At 0.2 every method flags
nothing on this clip — 0/32000 for all three — so the crossing machinery downstream does not run.
Deliberate and pending calibration, not an oversight. The sessions built so far used
`bounding_box` at 0.80; re-running `build_overlaps.py` at 0.2 would collapse 19 fragments to 16 and
delete the crossing fragment the crossing-identity path is tested on, so **it was not re-run** and
`session_fourfly/` still holds the 0.80 results.

**Validated** by running all three over the whole clip: each stays inside [0, 1] (asserted), each is
accepted by the solver, mean same-instance similarity 0.9769 / 0.9570 / 0.9885 respectively. `main()`
then run end to end under `bounding_box` and `centroid` on the Stage 3j scratch session — attrs record
the method and the scale (`nan` for the scaleless methods, 83.0 px for centroid), and the
instance-accounting invariant still holds with 4 dropped instances.

**Bug found and fixed while wiring it:** the new scale variable collided with an existing local
`scale` in `main()` holding a report string, so the attr write got a string. Renamed to
`centroid_scale`; the report string is now `units` and names the active method instead of hardcoding
"IoU".

---

## Stage 4b — the P2 exclusion cascade — WORKING

Previously omitted; now ported. `config.USE_P2_ASSIGNMENT = True`.

**What P2 is.** P1 is a fragment's own vote — how its images distributed over the identities. P2
weights that by every fragment which *coexists* with it:

    P2 = P1 * prod(1 - P1_coexisting)      (fragment.py:455)

so an identity a simultaneous fragment already claims confidently is driven toward zero. This is the
mechanism that stops two fragments in one frame taking the same identity. Fragments are then
assigned in descending order of P2 certainty (`get_fragments_to_identify`, list_of_fragments.py:344).

**The port is assigner.py:157-167 verbatim**, using upstream's own methods rather than a
reimplementation of the arithmetic — one step dropped: upstream re-runs the network over every
non-accumulated fragment first, because in its flow they have not been predicted on yet. Ours have,
so the P1 from `contrastive.predict` is reused.

**It runs without accumulation.** I expected coupling — `assign_remaining_fragments` calls
`reset(roll_back_to="accumulation")` and filters on `not frag.used_for_training`. In practice every
default lines up for a port with no accumulation pass: `identity=None`, `used_for_training=False`,
`identity_is_fixed=False`, `exclusive_roi=-1` against an `id_to_exclusive_roi` of all `-1`, and
`coexisting_individual_fragments` is populated at load. So the assignment half is separable from the
accumulation half after all.

**One gap had to be closed** (`seed_missing_P1`). `contrastive.predict` only sets P1 on fragments
with at least `MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION` images (contrastive.py:769);
`P1_vector` has no default, so `compute_P2_vector` would raise on a shorter one — and on every
fragment coexisting with it, since it reads their vectors too. Upstream never hits this because its
own predict pass has no length filter. Zeros are the right filler: they contribute `(1-0)=1` to
every coexisting product (exclude nothing, correct for a fragment with no evidence), and the
fragment's own P2 collapses to a zero vector, which `assign_identity` reads as an n-way tie and
answers with identity 0. No evidence in, explicit refusal out.

**Behaviour change:** identity 0 is now a legitimate outcome. When two identities tie on max P2 the
fragment is refused rather than guessed at (fragment.py:431). So "every instance has an identity"
stops being a hard check under P2 and becomes a reported count — failing the stage on it would
forbid the behaviour that was asked for. The check is still hard on the `USE_P2_ASSIGNMENT=False`
path.

### Results

**P2 changed nothing on this clip.** 18 fragments assigned, **0 ambiguous**, 18 fixed above
`FIXED_IDENTITY_THRESHOLD`, and it **differs from `argmax(P1)` on 0/18 fragments**. The clip is well
separated (silhouette 0.9632), so the coexistence term never had anything to overturn. The cascade
is running; it is not yet doing work here.

18 individual fragments fit into 16 identities because fragments 4 (frames 0-570) and 17 (573-1999)
do **not** coexist, and likewise 9 and 18 — the pre- and post-crossing halves of the same two
animals. Verified directly.

**So the branches were tested by fabricating P1 vectors** rather than trusting the clip:

- two *coexisting* fragments both voting identity 1 with confidence 0.90 and 0.60: the confident one
  takes it, the other is refused. Without P2 both would have taken it.
- 18 fragments given deliberately noisy overlapping votes: **0 coexisting pairs share an identity**.
- a fragment stripped of its P1 comes out identity 0, and the rest are unaffected.

**Still open:** with 18 fragments and 16 identities, forcing distinct assignments makes exactly 2
refusals unavoidable — correct behaviour, but a reminder that P2 refuses rather than reaches. Nothing
downstream re-tries a refused fragment; upstream would, via `correct_impossible_velocity_jumps` and
`close_trajectories_gaps`, both still unported.

---

## Stage 6 — identities back into a .slp (`build_sleap_tracks.py`) — WORKING

The .slp SLEAP produced has no identities at all: `labels.tracks == []` and every
`instance.track is None`. It holds poses. This stage fills that in from the port's output and writes
a **new** file, leaving the original canonical:

    <name>.predictions.slp   ->   <name>.idtracker_predictions.slp

Same frames, same instances, same keypoints, same skeleton, same video — the only difference is that
each instance now carries a `Track`.

**The mapping is two HDF5 columns.** `instance_idx` is the instance's position in `lf.instances`,
i.e. the index SLEAP itself gave it, and `identities` is what `build_identities.py` assigned. So row
`(frame f, instance_idx i, identity k)` means `labels[f].instances[i].track = tracks[k - 1]`. This
only works because `instance_idx` is sparse-not-renumbered — see Stage 3j, which is what makes the
column usable as a key.

`n_animals` tracks are created (the number of clusters the contrastive step sorted into, per the
request) and named for the identity they carry, so `track.name == "7"` is idtracker identity 7. An
unclaimed track is still created and stays empty — the count is a property of the clustering, not of
what happened to be used.

**Three ways an instance ends up with `track=None`**, none of them guessed at:

1. no HDF5 row — dropped by `build_id_images.py` for want of an alignment keypoint. Keypoints are
   still written; only the identity is absent.
2. identity 0, upstream's unassigned sentinel.
3. a collision — two instances in one frame given the same identity. Crossing rows are predicted
   independently with no mutual-exclusion pass, so this is reachable. **Both** are left untracked,
   reusing the policy already decided for `build_trajectories.py` rather than inventing a second
   one; writing either would invent a resolution the evidence does not support.

**Result on `session_fourfly`:** 32000/32000 instances tracked, 16 tracks, exactly 2000 instances
each, 0 collisions. Track "3" walks 600 frames with a median thorax step of 0.10 px and a max of
3.24 px — a real trajectory, not a label that jumps between animals.

**Validated** two ways. On the real session: keypoints bit-identical to the source, frame and
instance counts unchanged, file declares 16 tracks, every tracked instance's `track.name` equals its
HDF5 identity after a save/reload round trip, and no frame has two instances on one track. On the
Stage 3j scratch session (4 instances with no row) with identities fabricated as `instance_idx + 1`
and one collision injected: 31994/32000 tracked = 32000 − 4 − 2, the 4 dropped instances come back
with `track=None` and their keypoints intact, and both sides of the collision are untracked while
the rest of that frame is not.

**Still open:** identity numbering is arbitrary per run (see the Stage 4 decision on skipping the
cluster reorder), so track "3" in this file is not track "3" in a rerun. The tracks are internally
consistent, not stable across sessions.

---

## Stage 3c-3 — `GAP_SCALE = "total"` (second-best within threshold) — WORKING

**Set by explicit instruction:** a link is a crossing when the SECOND-BEST assignment costs within
`SIMILARITY_THRESHOLD` of the best one. That is the raw cost gap, so `config.GAP_SCALE` moved from
`"per_edge"` to `"total"`. The `"total"` branch already existed in `assignment_ambiguity`; only the
default and the documentation changed. Under it, the minimum over all off-assignment cells IS the
second-best assignment, so the code now reads as the definition states.

**Why the change was needed — diagnosed on `Pletcher_10fly`** (18000 frames, 10 flies, 150472
instances, 149342 id-image rows; run of 2026-08-22 on della, `/scratch/gpfs/SHAEVITZ/`). Of 3001
untracked instances the causes decompose exactly:

    1130  no h5 row       (head NaN -> no alignment -> never cropped)    37.7%
    1205  identity 0      (fragment below the 4-image gate)              40.2%
     666  collision       (two instances claiming one identity)          22.2%

Chasing one case — track 3, 1-based frames 139-140 — showed `per_edge` flagging an animal with **no
competitor anywhere near it**. Its bbox IoU with every other instance in the next frame was exactly
0.0000; nearest neighbour 118-179 px away; steady 11 px/frame drift; self-IoU 0.4334.

**The mechanism.** When detections drop out, the padded matrix gains a spare column at
`UNMATCHED_COST = 1.0` — the same cost as any zero-IoU pair. The orphaned row is then a row of
constants and can be moved for free. The cheapest alternative is "the moving animal goes unmatched,
the orphan takes its column", whose entire cost is the IoU the moving animal forfeits:

    committed    r0 -> pad  (1.0)  +  r3 -> col2 (1 - 0.4334)   = 1.5666
    alternative  r0 -> col2 (1.0)  +  r3 -> pad  (1.0)          = 2.0000
    increase 0.4334,  n_changed = 2,  per_edge = 0.2167  <= 0.3  -> FLAGGED

`n_changed` is 2 but **only one edge pays** — deltas `[0.0000, 0.4334]`. So `per_edge` halves the
margin and the effective rule degenerates to *flag any instance whose successor IoU < 2 x threshold*
(i.e. < 0.6). 15.52% of frame transitions on this clip have unequal instance counts and open such a
slot. Raising `UNMATCHED_COST` does **not** help: it appears on both sides of the subtraction and
cancels, so the gap is `IoU / 2` regardless of its value.

**Measured effect of the switch** (bbox, threshold 0.3, whole video, links rebuilt from the
`.slp` and chained under the same agreement rule):

| | `per_edge` | `total` |
|---|---|---|
| `is_overlapping` flagged | 6374 (4.27%) | 2702 (1.81%) |
| links severed | 6980 | 3888 |
| fragments | 6980 | 3888 |
| fragments < 4 images | 4468 | 2716 |
| instances stranded in those | 6752 | 3144 |
| max fragment length | 3459 | 4869 |

**Fragments drop 44%, stranded instances 53%.** On the specimen fly the five fragments in frames
134-150 collapse to one unbroken 17-frame chain, and all three severances in that window disappear.

**Replication validated** against the stored session: severed links came out at exactly 6980,
matching the recorded `next_global_index == -1` count. The flag count differs (6374 vs the stored
6159) because the replication counts flags on the 1130 rowless instances that never reach the h5.

**Caveats, recorded rather than resolved.** `"total"` is the blunt instrument: it is uniformly more
permissive and gives up the long-cycle protection `per_edge` existed to provide (16 animals each
shifting one place has a large total but a small per-animal cost). On footage where animals genuinely
cross, confirm real encounters still flag. A targeted alternative — divide by the number of edges
whose cost actually *changed* rather than the number of animals whose partner changed — would give
`0.4334 / 1` here and keep the long-cycle case at `total / 16`. Not implemented.

**Hull makes no difference to this failure.** Re-run with `SIMILARITY_METHOD = "hull"`: identical
flag pattern, identical cuts, identical 2-image stub. Hull is the tighter region (2397 px^2 hull vs
3922 px^2 box, 39% smaller), so self-IoU falls and the gap goes *further* below threshold
(0.1766 vs 0.2167). No similarity function can alter a row of constants.

**Known efficiency regression, NOT fixed.** Under `"total"` the prune limit becomes
`max(threshold, REPORT_CEILING)`, and `REPORT_CEILING = 1.0` exceeds every achievable bound, so the
prune rejects nothing. Measured over 4000 frame pairs (mean k = 8.30, 61.7 candidate cells/pair):

| configuration | solves/pair | forced/pair | pruned | sec |
|---|---|---|---|---|
| `per_edge` | 12.1 | 2.84 | 95.4% | 0.6 |
| `total`, ceiling 1.0 | 71.0 | 61.66 | **0.0%** | 2.7 |
| `total`, ceiling 0.5 | 12.6 | 3.32 | 94.6% | 0.6 |
| `total`, ceiling 0.4 | 11.2 | 1.88 | 97.0% | 0.6 |
| `total`, ceiling 0.3 | 10.9 | 1.58 | 97.4% | 0.6 |

Lowering `REPORT_CEILING` to ~0.4 restores the pruning (flags stay correct as long as
`REPORT_CEILING >= SIMILARITY_THRESHOLD`, already asserted in `main()`); only the reported gap sweep
narrows. **Left at 1.0** pending a decision. Absolute cost is ~3 s vs ~12 s for the whole video.

**Also unfixed:** `assignment_ambiguity` computes the lower bound `L` **twice** (`build_overlaps.py`
:227-232 against `P`, then :261-267 against `Q`); the first result is discarded before use. Harmless
to correctness, `k` wasted Hungarian solves per frame pair, ~180000 across this video.

**Documentation note:** the measured per-method minima recorded in Stage 3c-2 were taken under
`per_edge`. Under `total` the same alternatives score `n_changed >= 2` times higher, so those figures
are now lower bounds. A note to that effect was added beside `SIMILARITY_THRESHOLD`.

---

## Stage 3c-4 — neighbour gate on the cost matrix (`NEIGHBOUR_RADIUS`) — WORKING

**Requested optimisation.** Only score instance pairs whose `CENTROID_NODE` positions lie within
`NEIGHBOUR_RADIUS * body_length`; everything else is left at similarity 0, i.e. cost 1.0 ==
`UNMATCHED_COST`. `config.NEIGHBOUR_RADIUS = 1.0`, `None` disables. The radius is in body lengths,
read from the id-image `body_length` attribute — the same measured quantity the crops were sized
with — so it transfers between datasets without retuning.

**It removes work, not answers.** `similarity()` now routes gated calls through new row-wise twins in
`assignment_margin.py` — `iou_pairs`, `oks_pairs`, `centroid_pairs` — which score the P candidate
pairs directly rather than computing an all-pairs matrix and masking it afterwards. `hull_iou_matrix`
takes the mask itself, so gated pairs never reach `cv2.intersectConvexConvex` — the expensive metric
benefits most. Ungated behaviour is untouched: with `radius=None` the original matrix path runs.

**A missing centroid keeps the pair.** The test is written `~(d > radius)`, not `d <= radius`, so a
NaN distance from a missing `CENTROID_NODE` falls on the KEEP side. Absence of a position is not
evidence of distance, and excluding on it would silently drop links the ungated code would make.

**Validated** on synthetic poses with 10% missing keypoints, all four methods:

| method | open gate == ungated | inside gate == ungated | outside == 0 | nonzero cells dropped |
|---|---|---|---|---|
| `bounding_box` | yes | yes | yes | **0** |
| `keypoint` | yes | yes | yes | 3 |
| `centroid` | yes | yes | yes | **48** |
| `hull` | yes | yes | yes | **0** |

**Exact for `bounding_box` and `hull`** — two shapes whose centroids are further apart than their own
extent cannot overlap, so the skipped cells were 0 anyway. **Not exact for `centroid`**:
`exp(-d / body_length)` is 0.368 at exactly one body length and never reaches 0, so a gate at 1.0
truncates a similarity that metric considers meaningful. Raise to ~3.0 (truncation at 0.050) or set
`None` if matching on centroids. OKS is ~0 by one body length, so the gate is effectively exact there.

Attrs `overlap_neighbour_radius` and `overlap_gate_px` are recorded; `main()` prints the skipped
fraction. **Not measured:** the actual speedup on real data. `build_overlaps.py` has not been re-run
end to end since this change.

---

## Stage 6b — collision tiebreak by length then max(P1) — WORKING

**Set by explicit instruction.** When two fragments claim the same identity in one frame,
`drop_collisions` (`build_sleap_tracks.py`) previously dropped **both** — "a cell claimed twice is
claimed by nobody", inherited from the retired `build_trajectories.py`. It now arbitrates:

0. **Both in an overlapping fragment -> both `[none]`.** If every claimant is an `is_overlapping`
   row, all of them are left untracked. Set by explicit instruction and kept as its own rule rather
   than left to fall out of step 2. It currently *would* fall out — those rows carry NaN `p1_max` and
   step 2 cannot pick a winner among NaNs — but that is a consequence of how `p1_max` happens to be
   filled, and would vanish silently if crossing rows ever gained a per-row confidence. The rule does
   not depend on it: it reads the `is_overlapping` column directly.
1. **Length, but only against a ONE-IMAGE claimant.** A claimant carrying a single image of evidence
   loses to any competitor carrying more, with no score compared. **Two things count as one image:**
   a row of a CROSSING fragment *whatever that fragment's length* — its rows are different animals
   predicted one at a time with no pooling, so the evidence behind any one row is exactly one network
   call — and a row of an INDIVIDUAL fragment holding one image. Skipped when every claimant is a
   one-image claimant, since then there is no longer competitor to prefer.
2. **max(P1).** Among survivors, the highest `max(P1)` wins. P1 is normalised to sum to 1, so the
   value is comparable across fragments of different lengths.

Crossing rows have no pooled vote and so no P1: NaN, which loses to any competitor that has one. All
claimants are left untracked when no survivor has a finite `max(P1)` — two crossing rows contesting
each other — or when the maximum is exactly tied.

**One wrong reading was implemented and corrected first:** it ranked on raw fragment length, so a
5-row crossing fragment counted as 5 images of evidence. It is not — each of its rows is one
prediction. Effective length is `np.where(is_overlapping, 1.0, fragment_size)`.

**`is_overlapping` is the authoritative crossing indicator, not NaN `p1_max`.** Verified on the whole
session: across all 149342 rows there is not one where `is_overlapping == is_an_individual`, i.e. the
two are exact negations, so the column can be read directly instead of inferring the condition from a
missing score.

**Why it was needed.** On `Pletcher_10fly`, 1-based frame 16834: fragment #4817 holds **1 image** and
fragment #4279 holds **3459** — the longest individual fragment in the session — and both claim
identity 1. Under the old rule both were blanked, so a stray singleton 681 px away cost the real
track a frame. 263 of the 333 collisions have this shape: short crossing fragment against a long
individual one, the short side 1-16 images and `is_an_individual = False` in every one of the top 12.

**A trap in the implementation.** `assign_identity` collapses `P1_vector` to a one-hot
(`fragment.py:449-450`), so after the P2 cascade *every* assigned fragment reports `max(P1) == 1.0`
and the comparison would be vacuous. `p1_max_snapshot()` therefore captures it **before** the
cascade, next to the existing `p1_only` snapshot taken there for the same reason, and
`write_identities` writes it to a new per-row `p1_max` dataset.

**Crossing rows have no P1** — their rows are different animals predicted one at a time with no
pooling, so there is no vote to take a maximum of. They are written NaN and **lose to any finite
P1**. This is a judgement call, recorded as such: a row with no pooled evidence should not outrank a
fragment that has some. If every claimant is NaN, nobody wins.

**Validated** by 10 unit cases, all passing. Step 0: two overlapping rows at sizes 1 vs 5 both go to
`[none]`; two overlapping rows **carrying finite p1_max** still both go to `[none]`, which is the case
that proves the guarantee is independent of the NaN convention; a three-way all-overlapping contest
leaves all three untracked. Step 1: a row of a **5-row** crossing fragment loses to individual(3),
confirming effective length is 1 rather than 5; individual(1) loses to individual(4). Step 2:
individual(3) vs individual(3) goes to the higher max(P1); an exact tie drops both; overlapping vs
individual(1) goes to the individual, which has a P1. Plus uncontested rows untouched, and a call with
no `p1_max` column reproducing the old drop-both exactly so older sessions still load.

**Measured on the 333 real collisions** (proxy replay — that session predates the `p1_max` column, so
"is an individual fragment" stood in for "has a finite P1"; step 1 is exact, and there were no
individual-vs-individual contests):

    step 0  all claimants overlapping -> all [none]  :  69
    step 1  settled on length (one-image loses)      : 264
    step 2  settled on max(P1)                       :   0
            otherwise untracked (tie / no P1)        :   0

    INSTANCES RECOVERED : 264      (collision loss 666 -> 402, about 40%)

**Every contest in this session falls to step 0 or step 1.** All 264 resolved cases are an overlapping
row against a long individual fragment, settled on length; the other 69 are all-overlapping and go to
`[none]` by rule. **There is not a single individual-vs-individual contest in the video**, so step 2
is unit-tested but never exercised on real data here, and the residual "tie / no P1" bucket is empty.

**Not done:** `build_identities_silhouette.py` has its own `write_identities` (line 343) and does not
yet write `p1_max`, so silhouette-sourced sessions fall back to the length rule only.
`build_trajectories.py` keeps the old unconditional drop-both, so the two outputs now disagree on
contested frames by design.

---

## Stage 3-3 — hull crossing detection removed — DONE

`CROSSING_NODES`, `CROSSING_METRIC` and `CROSSING_THRESHOLD` are gone from `config.py`, along with
`frame_crossings()` in `build_id_images.py`, the `crossings` HDF5 column, its three attrs, and the
diagnostic print. Deleted on request, after checking that nothing downstream reads any of it.

**Why it was safe.** The block was already documented as diagnostic. Verified rather than trusted:

- Nothing in `src/sleap_idtracker/` reads the `crossings` dataset — only `build_id_images.py` wrote
  it. `build_fragments.py:402` filters on `Fragment.is_an_individual`, which comes from
  `is_overlapping` (build_overlaps.py), not from this column.
- Upstream's `crossing_detector.py` writes a `crossings` dataset of its own, but it belongs to the
  blob-segmentation path this port skips entirely and is never imported by our code.
- `build_overlaps.py`'s `snapshot()` / `drop_stale()` enumerate `fh.keys()` generically, so a file
  with one fewer dataset needs no change, and older sessions that still carry the column still load.
- `frame_crossings()`'s return values fed only that column and the print — nothing load-bearing.

`BODY_NODES` lost its last use inside `build_id_images.py` (it was resolving the crossing hull's node
subset) and the binding was dropped there; `config.BODY_NODES` stays, since `OVERLAP_NODES = "body"`
still selects it in `build_overlaps.py`.

**Validated end to end** on a 40-frame / 80-instance mice subset: `build_id_images.py` then
`build_overlaps.py` both run clean, 80/80 hulls from `SURROUNDING_KEYPOINTS`, 78/80 links,
"every one of 80 instances labelled exactly once". The written file now holds

    centroid, frame_numbers, global_index, hull_fallback, id_images, identities,
    instance_idx, is_overlapping, local_index, next_global_index

with `crossings` absent and no `crossing_*` attrs. All ten modules byte-compile and import.

**What this does not touch.** "Crossing fragments" — the merged `is_an_individual = False` fragments
that hold several animals — are unaffected. They come from `is_overlapping`, are still built, still
predicted per-row, and are still what step 0 of the collision tiebreak keys on. Only the unused
within-frame geometric flag is gone.

---

## Stage 3k / 6c — `TOP_NODE_FALLBACK`, pre-existing tracks, and the oline mice run — WORKING

First run on `oline_multimouse/camera0_topb` (5 mice, 108000 frames, 1024x1280 grayscale, 60 fps).
Ran on della as two Slurm jobs. Three code changes were needed to get there.

### 1. `TOP_NODE_FALLBACK` (new, requested)

`TOP_NODE` was a single node: NaN there meant no rotation, no crop, no h5 row, no identity. On this
dataset `Nose` is NaN on 5.85% of instances, which would have discarded them outright. Config now
takes `TOP_NODE_FALLBACK`, consulted ONLY when `TOP_NODE` is NaN. Implemented in
`rotate_boxes.top_node_xy` and threaded through `build_id_images.collect_geometry` /
`rotation_matrix`; per-episode `n_top_fallback` attr records how often it fired. Both NaN still skips
the instance, exactly as before.

MEASURED on 529048 instances: 498344 aligned on `Nose`, **24801 (4.69%) on the `Head` fallback**,
5903 (1.12%) skipped (5889 with both top nodes NaN, 14 with no `Trunk`). Without the fallback 30704
instances (5.80%) would have had no id-image row.

### 2. Pre-existing SLEAP tracks are now cleared (`build_sleap_tracks.apply_tracks`)

A real bug, exposed by the first pre-tracked input. `apply_tracks` replaced `labels.tracks` with the
n_animals Tracks it creates, but `Instance.track` holds a POINTER to a Track object, not a name or an
index. Every instance this stage does not re-point kept pointing at one of the old objects --
reachable from the frames, absent from `labels.tracks` -- and sleap-io resolves each instance's track
as `labels.tracks.index(inst.track)` (`slp.py:1784`), so the writer dies with

    ValueError: Track(name='...') is not in list

MEASURED: this input carries **581 SLEAP tracks on all 529048 instances**. Every earlier input was
untracked (`track=None` throughout) and the Pletcher run additionally stripped tracks up front in
`prepare_clean_slp.py`, which is why it never surfaced before. Fixed by clearing every
`instance.track` at the top of `apply_tracks`, so the guarantee holds for ANY input rather than
depending on a per-dataset preparation script. This is what `main()`'s validation already asserted
("file declares n_animals tracks"); the contract is now enforced rather than assumed. **User decision
(asked, not assumed): overwrite the tracks.**

Note the deliberate loop variable `pi`, not `inst`: `apply_tracks` unpacks `inst = cols["instance_idx"]`
at the top, and a loop named `inst` leaves that name bound to a PredictedInstance, so the later
`inst[row]` indexes an instance by node instead of an array by position and fails with
`IndexError: Invalid indexing argument for skeleton: 0`. That was the first version of this patch and
the preflight caught it before any GPU time was spent.

### 3. Per-batch loss logging (`contrastive.py`, `# SLEAP-PORT:`)

Upstream's `batch_counter` is `logging.debug` inside a `rich` Console and renders NOTHING to a
non-TTY Slurm log. Added an explicit recorder writing `epoch,batch,global_step,loss` to
`loss_per_batch.csv` in the session dir, plus a plain `print` every 50 batches that survives
redirection.

### The run

Input corrected first: on frames holding n > 5 instances, the n-5 lowest `tracking_score` instances
were removed (**user decision**; `tracking_score` median 0.422, distinct from `score` median 0.790).
532071 -> **529048** instances, **3023 removed over 3001 frames** (2979 at n=6, 22 at n=7), written as
`corrected_mice_camera0_topb.slp`. The .slp referenced `/mnt/cup/...`, not mounted on della; repointed
to the copy on scratch.

| job | partition | wall | notes |
|---|---|---|---|
| A (stages 1-4) | cpu | 12:52 | build_id_images 642s, build_overlaps 104s, peak RSS 3.5 GB |
| B (stages 5-6) | gputest | 19:14 | 1:00:00 limit, MAX_TRAIN_S=1500 |

523145 id-images at **57x111** -- the `MAX_CROP_AREA = 6400` clamp engages hard here, scaling native
97x187 (18139 px) by 0.594. Flagged to the user, who chose to keep 6400. 11.96% of hulls rebuilt by
Graham scan. **816 instances (0.15%) flagged overlapping** at the 0.3 hull threshold: unlike the
Pletcher clip, the threshold sits INSIDE the real gap distribution here, so the crossing path
actually engages. 5277 fragments (2930 individual / 2347 crossing), 1641 accumulable global
fragments, so KMeans seeded from a real global fragment rather than k-means++.

Training stopped on **patience at 985s**, inside the 1500s budget -- the wall-clock cap never fired.
**Best silhouette 0.8216** (final 0.8024) over 219 validations / 10950 batches. The 0.9 target was NOT
reached; per instruction ("prioritize working for an hour rather than working to achieve a score")
the run continued and wrote its output. Loss 42.88 -> 0.30, min 0.0412 at step 10901.

Output `corrected_mice_camera0_topb.idtracker_predictions.slp` (240.4 MB):

    tracked   472425 / 529048  (89.30%)
    tracks    5 of 5 used, 86418 / 96822 / 96128 / 97127 / 95930
    untracked 5903 no h5 row, 49665 identity 0, 1055 collided

All five identities are evenly occupied (80-90% of frames each) -- no collapsed or starved track.
**identity 0 is the dominant loss at 49665**, the same short-fragment gate seen on the fly data.

**della partition note:** `gpu-test` is not a partition name -- it is `gputest`, and della *refuses*
an explicit `--partition=gputest` ("This is not allowed"). Its submit plugin routes the job there
automatically; the line must be omitted.

---

## Stage 4c / 6d — the P1-saturation veto, the `p1_max` omission, and a test suite — PARTLY FIXED

Investigating a 1729-frame hole in the oline mice output (identity 5 absent from 1-based frames
30288-32016 while all five mice were plainly detected) turned up two independent bugs and one
structural weakness. One is fixed, one is half fixed, and the port now has its first tests.

### BUG A — `p1_max` was never passed to `write_identities` (FIXED, 3 drivers)

`write_identities(lof, per_fragment, per_row, p1_max=None)` was being called with **three**
positional arguments by every driver, and `p1_max_snapshot` was never called at all. So `p1_max`
took its `None` default, `p1_max or {}` made it empty, every `.get()` returned NaN, and the column
was written **all-NaN: 523145 rows, 0 finite**. No error, no warning.

Downstream, step 2 of `build_sleap_tracks.drop_collisions` — the max(P1) tiebreak — had nothing to
rank on. The oline log reads `819 settled on length, 0 on max(P1)`. It cost nothing in that run
because step 1 caught every contest, but it is latent for any run where two multi-image individual
fragments contest one identity: both are dropped instead of the better-supported one winning.

**Root cause is duplication, not a typo.** Stage 5 exists three times — `build_identities.main()`
plus a re-implementation inside each `track_*.py`. `main()` passes `p1_max` correctly at line 480
and **never runs**; the drivers do. The module gained a parameter, the drivers did not, and nothing
checked. Fixed in `track_oline_mice.py`, `track_mice_new.py` and `track_blotted_flies.py`.

`track_oline_mice.py` was also only ever on della — the driver that produced the results was not
version controlled. It is now in the repo.

### BUG B — P1 saturates to exactly 1.0, turning coexistence into a hard veto (HALF FIXED)

`set_P1_from_frequencies` (`fragment.py:483-491`) is a base-2 softmax over vote counts, so
algebraically P1 < 1 always. In float64 it is not: once the winner leads by **53 votes**, `2**53`
exceeds the mantissa and the losing terms round away. MEASURED: **1116 of 2156 predicted fragments
(51.8%) saturate to exactly 1.0.**

`compute_P2_vector` multiplies by `prod(1 - coexisting_P1)`. A saturated neighbour contributes an
EXACT `0.0`, so a soft discount becomes an irrevocable veto. With N animals the other N-1
legitimately claim N-1 identities and one is always free — the all-vetoed state is UNREACHABLE
unless the network puts two COEXISTING fragments on the same cluster. When it does, the last slot
closes, the numerator is all zeros, and `assign_identity` (`fragment.py:428-433`) reads the zero
vector as an N-way tie and returns identity 0. Both fragments annihilate and the identity they
fought over ends up used by nobody.

That is exactly the reported hole: `#1284` (1582 images) and `#1282` (314 images) both voted
cluster 5, coexist on frames 30433-30608, both got identity 0, and identity 5 has **zero rows**
across the whole window.

Accounting for the 49665 identity-0 instances:

| cause | fragments | instances | share |
|---|---|---|---|
| P2 zeroed by full coexistence veto | 95 | **48154** | **97.0%** |
| seeded zero P1 (`n_images < 4`) | 774 | 1318 | 2.7% |
| genuine tie on a nonzero P2 | 16 | 193 | 0.4% |

The 774 too-short fragments are 30% of the ambiguous fragments but only **2.7%** of the lost
instances. Earlier notes had this weighted the other way round.

**Saturation breaks things in TWO ways**, which matters for the fix:

1. a neighbour's `1 - P1 == 0` makes its veto absolute, so every slot can be blocked;
2. a fragment's OWN saturated one-hot has `P1 == 0.0` on every other identity, so
   `numerator[i] = P1[i] * prod[i] = 0` **even for a completely free slot** — it has no fallback.

### `enforce_p1_uniqueness` (NEW, `build_identities.py`) — fixes form 1

Ported from what upstream does and this port skips. Upstream never reaches the degenerate state
because every route to a one-hot P1 goes through the accumulation protocol first:
`P1_array[:, temporary_id] = 0.0` after each assignment (`accumulation_manager.py:545`), an explicit
`is_inconsistent_with_coexistent_fragments` check (`:537`), and a final
`global_fragment.is_unique(n_animals)` rollback (`globalfragment.py:72`). This port feeds raw network
argmaxes straight to P2, so the guarantee is absent.

The new pass restores the guarantee alone, at the P1 stage: for each set of COEXISTING fragments
whose saturated P1 claims the SAME identity, the one with the most images keeps the claim and the
rest are desaturated so `1 - P1` is `1e-12` rather than `0.0`. Argmax is preserved, no identity is
assigned or forbidden, the cascade still decides, and **no vendored code is touched** —
`compute_P2_vector` simply stops receiving an exact 1.0 on a contested slot.

Verified against real `Fragment` objects (not a numpy model):

    n_saturated 1116 · n_conflicting_pairs 92 · n_demoted 81 (21533 images)
    fully vetoed 108 -> 50    RESCUED 58 (42315 images)    newly zeroed 0
    #1284 rescued -> identity 5

### Still open: clipping, for form 2

The 50 that remain are mostly (42/50) demoted losers whose own saturated one-hot leaves them zero
mass on every free slot. Simulated:

| | fully vetoed |
|---|---|
| baseline | 108 fragments / 58841 images |
| uniqueness only | 50 / 16526 |
| **clipping only** | **0 / 0** |
| uniqueness + clipping | 0 / 0 |

Clipping the exponent in `set_P1_from_frequencies` so `1 - P1` never reaches 0 fixes BOTH forms and
should be the primary fix; uniqueness remains worth keeping because it makes duplicate resolution
principled (most evidence wins) rather than dependent on cascade order, which is driven by
`certainty_P2` — `inf` for 1560 fragments, hence effectively identifier order. **NOT implemented**;
it changes P1 everywhere and needs a measured re-run. Note the simulations above are STATIC and
pre-cascade: they predict that no fragment hits the degenerate state, NOT that the identities
assigned are correct.

### `tests/` — the port's first tests

No test infrastructure existed for `src/sleap_idtracker/` (only the vendored forks have any), and
pytest is not installed in `sleap_id`. Both files run as plain `python tests/x.py` and drop into
pytest unchanged.

`tests/test_numerics.py` — property tests on the numerical core, swept across
`N_ANIMALS = [2,5,10,16]` x `FRAG_LEN = [4,20,53,60,500,5000]`. **The sweep is the design**: the
saturation invariant PASSES at lengths 4/20/53 and FAILS at 60+, and every fixture this port was
validated on was below that line. Currently 2/5 pass — the three failures are real bugs:
`test_p1_never_reaches_exactly_one`, `test_p2_preserves_veto_ordering`, and
`test_unmatched_cost_is_distinguishable_from_a_real_pair` (the `UNMATCHED_COST = 1.0` sentinel
collides with the cost of a zero-similarity real pair — the same bug shape as the saturation, and
the root of the earlier `GAP_SCALE` problem).

`tests/test_contracts.py` — AST checks that every `track_*.py` passes the load-bearing arguments to
the stage-5 functions and snapshots P1 before the cascade. It found Bug A in two drivers nobody had
looked at. Now 2/2 pass.

Still missing: an end-to-end fixture asserting that tracked + no-row + identity-0 + collided sums to
the instance total, which is the invariant that would have surfaced "9% of the data is silently
identity 0" the day it appeared.

---

### Re-run on the oline mice, prediction only (no retraining) — MEASURED

Stages 5-6 re-run in place against the Sep 5 checkpoint (md5 `a764506c...` verified unchanged
before and after; `MAX_TRAIN_S=0` is NOT sufficient, because `train_with_budget` trains 50 batches
and validates before its elapsed check and would `torch.save` over the checkpoint on the first
validation -- a `SLEAP_IDTRACKER_NO_TRAIN=1` branch was added instead, off by default).

| | baseline (Sep 5) | with the fixes | delta |
|---|---|---|---|
| tracked instances | 472425 | **503595** | +31170 |
| % of 529048 instances | 89.30% | **95.19%** | +5.89 pt |
| % of 540000 animal-frames | 87.49% | **93.26%** | +5.77 pt |
| untracked: identity 0 | 49665 | **18377** | -31288 |
| untracked: no h5 row | 5903 | 5903 | 0 |
| untracked: collided | 1055 | 1173 | +118 |

`P1 uniqueness: 1115 saturated, 91 coexisting duplicate pair(s), 80 demoted (21102 images)` --
within kmeans jitter of the 1116/92/81 predicted offline. `P2 cascade: 2080 assigned, 850 ambiguous`
against the baseline's 2045/885.

**The reported 1729-frame hole is closed**: identity 5 now covers **1582 of those frames**, exactly
fragment #1284 recovered by the demotion. #1282 (314 images) stays unassigned, which is the intended
outcome -- it lost the duplicate contest.

**The `p1_max` fix contributed nothing measurable to this result**, and that should be recorded
honestly: stage 6 reports `1051 contested cells -> 114 all-overlapping, 937 settled on length,
0 on max(P1)`. Every contest was already resolved at step 0 or step 1, exactly as in the baseline.
The column is now populated (518243 finite / 4902 NaN, where the NaN are 3584 crossing rows by
design plus 1318 rows of fragments too short for `predict()`), so step 2 is armed rather than
silently dead -- but it changed no assignment here. All +31170 instances came from
`enforce_p1_uniqueness`.

The +118 collisions are the expected cost of assigning more fragments (1051 contested cells vs 933).
All seven stage-6 export validations pass, keypoints bit-identical to source.

Baseline preserved at `corrected_mice_camera0_topb.idtracker_predictions.slp.baseline` and
`session_oline_mice/baseline_sep5/` (stage 5 truncates its logs and `findings.json` on every run).

**18377 instances remain identity 0**, mostly the 50 fragments needing the clipping fix -- form 2 of
the saturation bug, where a demoted fragment's own one-hot P1 leaves it zero mass on every free slot.

---

## Things that did not work

- **`per_edge` gap via plain constrained Hungarian solves** — under-flagged, because cost-per-edge is
  a ratio and the Hungarian minimises sums. Replaced by the λ-shift above. See Stage 3c.
- **Strict `>` on the gap threshold** — silently discarded exactly-tied optima, the single most
  ambiguous case. Replaced by a `TIE_TOL` comparison. See Stage 3c.
- **File-path fallback import of `Episode`** — worked, but was the wrong solution to a problem better
  fixed by `pip install -e ./idtrackerai --no-deps`. Now dead code pending deletion.
- **Convex hull as a silhouette** — geometrically correct, perceptually poor. See above. Not a bug;
  a modelling-choice limitation, and the most important open question right now.
- **`GAP_SCALE = "per_edge"` as the ambiguity scale** — divides by the number of animals whose partner
  changed, which counts animals whose change cost nothing. With a padding slot present the margin
  becomes `IoU / 2`, flagging isolated cleanly-tracked animals. Replaced by `"total"`. See Stage 3c-3.
- **Dropping both sides of an identity collision** — let a 1-image fragment blank a 3459-image one.
  Replaced by the length / max(P1) tiebreak. See Stage 6b.
- **Switching to hull similarity to fix fragmentation** — measured, no effect: identical flags and
  identical cuts, and marginally worse margins. The failure is in the cost matrix's padding column,
  which no similarity function participates in. See Stage 3c-3.
