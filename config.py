"""Every knob the user sets, in one place.

Each stage module used to carry its own constants. They are gathered here so the
whole pipeline can be configured from a single file, and so a value shared by two
stages (BODY_NODES, CENTROID_NODE) has exactly one definition rather than an
import chain between stages.

This module imports nothing from the rest of the pipeline, so it can be imported
from anywhere without a cycle. Stage modules bind their old names to these
values, so nothing else in them had to change:

    from config import CROSSING_THRESHOLD    # build_id_images.py

Values that are DERIVED rather than chosen -- the id-image size, the episode
boundaries, the fragment count -- are not here. They are computed by the stage
that owns them and reported at run time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file sits at the top of the repository, so the root is its own directory.
# It used to be `parents[2]` -- the port lived inside a larger checkout and
# reached up into it for both code and data. Nothing reaches upward now.
REPO = Path(__file__).resolve().parent

# idtracker.ai ships *inside* this repo rather than being cloned alongside it.
# It carries surgical `# SLEAP-PORT:` edits (base/network/models.py, the
# non-square input change from Stage 3g), so unlike the vendor/ forks it is NOT
# reproducible from upstream and has to travel with the port.
#
# Putting its `src/` on sys.path here is what lets a fresh clone run without
# `pip install -e idtrackerai/` first. It is the one deliberate side effect in
# this module, and it is why stage modules must `import config` BEFORE they
# `from idtrackerai... import ...`. If idtracker.ai is already installed
# (editable or otherwise), that install is left alone -- an existing entry wins
# because we only insert when the path is absent.
IDTRACKERAI_SRC = REPO / "idtrackerai" / "src"
if IDTRACKERAI_SRC.is_dir() and str(IDTRACKERAI_SRC) not in sys.path:
    sys.path.insert(0, str(IDTRACKERAI_SRC))

# Videos, .slp files and session output are NOT tracked in this repo -- they are
# gigabytes of experiment data, and which experiment you are running changes per
# user. `data/` is the default location; override it to point at a checkout that
# keeps its recordings elsewhere:
#
#     export SLEAP_IDTRACKER_DATA=/scratch/matthew/fourfly
#
# SLP and SESSION_DIR can each be overridden on their own if the layout under
# the data root does not match.
DATA_ROOT = Path(
    os.environ.get("SLEAP_IDTRACKER_DATA") or REPO / "data"
).expanduser()

SLP = Path(
    os.environ.get("SLEAP_IDTRACKER_SLP")
    or DATA_ROOT
    / "predictions"
    / "fourfly_clip.mp4_labels.v001.slp.260721_104536.predictions.slp"
).expanduser()

SESSION_DIR = Path(
    os.environ.get("SLEAP_IDTRACKER_SESSION") or DATA_ROOT / "session_fourfly"
).expanduser()
OUT_DIR = SESSION_DIR / "id_images"
FRAGMENTS_JSON = SESSION_DIR / "list_of_fragments.json"
GLOBAL_FRAGMENTS_JSON = SESSION_DIR / "list_of_global_fragments.json"

# build_sleap_tracks.py writes the identities back into a .slp as SLEAP Tracks.
# It never touches SLP itself: the output is a sibling file with
# `.predictions.slp` swapped for the suffix below, so the file SLEAP produced
# stays canonical and the port's output is distinguishable at a glance.
PREDICTIONS_SUFFIX = ".predictions.slp"
IDTRACKER_SLP_SUFFIX = ".idtracker_predictions.slp"
IDTRACKER_SLP = SLP.parent / (
    SLP.name[: -len(PREDICTIONS_SUFFIX)] + IDTRACKER_SLP_SUFFIX
    if SLP.name.endswith(PREDICTIONS_SUFFIX)
    else SLP.stem + IDTRACKER_SLP_SUFFIX
)


# ---------------------------------------------------------------------------
# The population
# ---------------------------------------------------------------------------

# How many distinct animals the video can be expected to contain.
#
# This is the number of identities to sort the animals into, and nothing else.
# It is stored in list_of_fragments.json, read back as ListOfFragments.n_animals,
# and becomes ContrastiveLearning.n_animals (contrastive.py:261) -- which is
# literally the `k` handed to MiniBatchKMeans(self.n_animals, ...)
# (contrastive.py:744, :797). Set it to the number of animals you put in the
# arena.
#
# It is deliberately NOT validated against how many instances the tracker
# actually detected. Detecting fewer is the ordinary consequence of occlusion,
# and detecting more means spurious instances -- neither changes how many real
# animals there are, so neither changes k. The counts are reported side by side
# at run time so a large disagreement is visible, but it is not an error.
#
# One place does compare them, and must: idtracker.ai defines a global fragment
# as a frame holding exactly n_animals separate individuals
# (list_of_global_fragments.py:221). If N_ANIMALS exceeds what is ever detected
# at once, no frame qualifies and there are no global fragments -- which is a
# supported outcome, handled by the k-means++ path in build_global_fragments.py.
N_ANIMALS = 16

# How an individual fragment's identity is read off after prediction.
#
# True  -- upstream's P2 cascade (assigner.py:157-167). P2 weights a fragment's
#          own P1 vote by the votes of every fragment that COEXISTS with it:
#              P2 = P1 * prod(1 - P1_coexisting)   (fragment.py:455)
#          so an identity a simultaneous fragment already claims confidently is
#          driven toward zero. Fragments are then assigned in descending order
#          of P2 certainty. This is what stops two fragments in the same frame
#          from taking the same identity.
#
#          Costs something: when two identities tie on max P2 the fragment is
#          assigned identity 0 (ambiguous) rather than guessed at, so "every
#          instance gets an identity" becomes "every instance gets an identity
#          or an explicit refusal". That is upstream's behaviour, not an
#          addition of ours.
#
# False -- the earlier MVP behaviour: argmax(P1) per fragment, no coexistence
#          term. Always assigns, and can collide. Kept so the two are
#          comparable, since the collision-handling policy downstream exists
#          only because of this path.
USE_P2_ASSIGNMENT = True


# ---------------------------------------------------------------------------
# Episodes  (episodes.py)
# ---------------------------------------------------------------------------

# How many frames go in one id-image HDF5 file. Upstream's default,
# idtrackerai/src/idtrackerai/session.py:101.
FRAMES_PER_EPISODE = 500


# ---------------------------------------------------------------------------
# Skeleton: which keypoints mean what  (rotate_boxes.py, canonical_hulls.py)
# ---------------------------------------------------------------------------

# The two alignment keypoints. Every id-image is rotated so TOP_NODE ends up
# directly above CENTROID_NODE, and CENTROID_NODE is the point rotated about --
# so it is also the fixed point of the transform and the animal's origin.
TOP_NODE = "head"
CENTROID_NODE = "thorax"

# The "bottom" of the animal, used only to decide how high the centroid sits in
# the canonical frame. NOTE: wingR is the lowest node in just 9/16 flies on frame
# 0 -- the rest are bottomed by the left wing or a hindleg -- and being the RIGHT
# wing makes it asymmetric under wing pose. Kept as specified.
BOTTOM_NODE = "wingR"

# The animal's body outline, excluding limbs and wings. Legs and wings splay far
# past the body, so a hull over every node reports contact long before the bodies
# are anywhere near each other.
BODY_NODES = ["head", "eyeL", "eyeR", "thorax", "abdomen"]

# Padding around the rotated keypoint extent, in pixels (rotate_boxes.py).
PAD = 0


# ---------------------------------------------------------------------------
# Id-image geometry  (build_id_images.py)
# ---------------------------------------------------------------------------

# Population statistic setting the canonical frame size, applied to the x spread
# and the y extent alike. Computed over EVERY instance in the movie, not frame 0.
SIZE_STAT = "median"

# Upper bound on the AREA of one id-image, in pixels. The canonical frame is
# SIZE_STAT(x_spread) x SIZE_STAT(y_extent), which is set by how big the animals
# are on screen and can be arbitrarily large -- a close-up recording, or a bigger
# species, produces crops that blow up the id-image HDF5 and the training batches
# with it. Cost is linear in this number: at 5.76M instances every 1000 px of
# crop area is ~5.8 GB on disk.
#
# If W*H exceeds this, BOTH dimensions are scaled by the same factor
# sqrt(MAX_CROP_AREA / (W*H)) so the aspect ratio is preserved -- the animals
# stay the same shape, just smaller. The scale folds into the affine that
# canonical_crop already applies, so it costs no extra interpolation and no
# extra pass. Set to None to disable the clamp entirely.
#
# 6400 = 80x80 equivalent. This clip's crops are 57x83 = 4731, comfortably
# under, so the clamp does not engage here and the existing sessions are
# unaffected.
MAX_CROP_AREA = 6400


# ---------------------------------------------------------------------------
# Hull crossing detection  (build_id_images.py)  -- DIAGNOSTIC ONLY
# ---------------------------------------------------------------------------
# NOTE ON STATUS: this is the within-frame geometric test, and it does NOT decide
# what the pipeline treats as a crossing. The `crossings` dataset it writes is
# read by nothing downstream. The load-bearing determination is `is_overlapping`,
# computed from the assignment cost matrix in build_overlaps.py -- see below.
# These knobs are kept because the hull overlap is still a useful thing to
# measure and report.
#
# An instance is flagged when its hull overlaps another instance's hull in the
# same frame by more than CROSSING_THRESHOLD, measured by CROSSING_METRIC over
# the nodes selected by CROSSING_NODES.
#
# CROSSING_NODES: "body" -- hull of BODY_NODES only.  "all" -- hull of every node.
#
# CROSSING_METRIC:
#   "iou"   -- intersection / union. Symmetric; the conventional choice.
#   "iomin" -- intersection / area of the SMALLER hull. Use when one animal is
#              much smaller, where IoU under-reports full occlusion of the small
#              one. Not needed for same-species same-size subjects.
#
# CROSSING_THRESHOLD: measured on this dataset, full-hull IoU never exceeds
# 0.0382 anywhere in 2000 frames and body-hull IoU is 0.0000 everywhere. See
# UPDATES.md for the sweep. 0.05 sits above the full-hull noise floor with
# headroom and is inert for body hulls.
CROSSING_NODES = "body"
CROSSING_METRIC = "iou"
CROSSING_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Frame-to-frame linking and overlap  (build_overlaps.py)  -- LOAD-BEARING
# ---------------------------------------------------------------------------
# This is where the pipeline actually decides whether an instance is a confident
# individual or an overlap. An instance is flagged as overlapping when the
# Hungarian assignment that links it to the neighbouring frame is not clearly
# better than the alternatives -- i.e. when identity cannot be trusted across
# that link. `is_an_individual` is `not is_overlapping`.
#
# SIMILARITY_THRESHOLD: the cutoff. An alternative assignment counts as "nearly
#   as good" when it costs within this of the optimum, measured on the scale
#   GAP_SCALE selects. Cost is 1 - IoU, so under the default "per_edge" scale
#   the units are IoU's own: 0.3 means "if some other pairing gives up less than
#   0.3 of IoU per re-assigned animal, do not trust this link".
#
# OVERLAP_NODES: which keypoints define the bounding box.
#   "all"  -- every node, which is what SLEAP's own tracker uses.
#   "body" -- BODY_NODES only, matching CROSSING_NODES. Legs and wings splay, so
#             full-pose boxes are larger and overlap sooner.
#
# OVERLAP_DIRECTION: which links have to be unambiguous for an instance to
#   count as non-overlapping.
#   "both"     -- both the link into it and the link out of it (conservative).
#   "forward"  -- only t -> t+1.
#   "backward" -- only t-1 -> t.
#
# MAX_FRAME_GAP: only pair frames at most this far apart. A larger separation is
#   treated as "no neighbour", which yields no evidence and therefore no flag.
#
# GAP_SCALE: what the threshold is measured against. This matters more than it
#   looks, because the two options are on different scales.
#   "per_edge" -- divide the total cost increase by the number of animals whose
#                 partner changed. Units are then "mean IoU given up per
#                 re-assigned animal", i.e. the [0, 1] scale IoU itself lives on,
#                 which is the scale SIMILARITY_THRESHOLD is quoted in.
#   "total"    -- the raw increase in total assignment cost. Because a
#                 permutation can never change just one pairing (the difference
#                 between two permutations is a cycle of length >= 2), the
#                 smallest possible change touches two edges and this quantity
#                 sits on roughly [0, 2]. A threshold of 0.3 here is ~6x
#                 stricter than the same number under "per_edge".
#   "per_edge" also handles long cycles correctly: 16 animals each shifting one
#   place is genuinely ambiguous, and per-animal cost catches it where a large
#   total would not.
#
# SET TO 0.80 DELIBERATELY, NOT 0.3. On this clip the smallest per-edge gap
# found anywhere is 0.7856 (p1 = 0.8743), so the original 0.3 flags nothing at
# all -- 0/32000 instances -- and the crossing machinery downstream never runs.
# 0.80 sits just above that minimum, so only the genuinely tightest encounters
# are flagged. This is a real detection on real data, not a synthetic one, and
# it is what makes the crossing-identity path testable. Lower it back to 0.3 for
# a production run on a clip where animals actually cross.
# Which similarity the frame-to-frame cost matrix is built from. These are the
# three the SLEAP tracker offers, ported to match its own definitions
# (sleap_nn/tracking/utils.py):
#
#   "bounding_box"  bbox IoU over OVERLAP_NODES        -> assignment_margin.iou_matrix
#   "keypoint"      OKS over OVERLAP_NODES             -> assignment_margin.oks_matrix
#   "centroid"      exp(-d / body_length), where the centroid is the nanmedian
#                   of OVERLAP_NODES (upstream's get_centroid, NOT CENTROID_NODE)
#                                                      -> assignment_margin.centroid_matrix
#
# All three return [0, 1], which the solver requires: the cost is `1 - S` and
# has to stay finite and non-negative against UNMATCHED_COST padding. Upstream's
# euclidean scorer returns a raw negative distance and could not be used as-is;
# see centroid_matrix for the mapping and why the scale is the body length.
SIMILARITY_METHOD = "bounding_box"

# One threshold per method, because the three do NOT share a scale -- the same
# number means a different thing under each, and a single shared value would be
# silently wrong the moment SIMILARITY_METHOD changed.
#
# NOTE (user-set, deliberate): all three are 0.2 for now, pending calibration.
#
# MEASURED on this clip -- the smallest per-edge gap found anywhere, i.e. the
# value a threshold must EXCEED before anything is flagged at all:
#
#     bounding_box   min gap 0.7856   median 0.9535
#     keypoint       min gap 0.5820   median 0.9110
#     centroid       min gap 0.4908   median 0.8482
#
# So at 0.2 every method flags nothing on this clip -- 0/32000 for all three --
# and the crossing machinery downstream never runs. That is a deliberate choice
# to leave them low for now, not an oversight. Note the three minima differ by
# ~0.3, which is the concrete reason these are per-method rather than shared.
#
# The sessions built so far used bounding_box at 0.80, which sits just above its
# minimum and flags only the genuinely tightest encounters (4 instances, 1
# crossing fragment). Set the bbox entry back to 0.80 to reproduce them;
# re-running build_overlaps.py at 0.2 would collapse 19 fragments to 16 and
# remove the crossing fragment the crossing-identity path is tested on.
SIMILARITY_THRESHOLD = {
    "bounding_box": 0.2,
    "keypoint": 0.2,
    "centroid": 0.2,
}
OVERLAP_NODES = "all"
OVERLAP_DIRECTION = "both"
MAX_FRAME_GAP = 1
GAP_SCALE = "per_edge"

# Cost charged for leaving an instance unpaired. 1.0 == the cost of a real pair
# at IoU 0, so padding is never preferred over a genuine (if poor) match.
UNMATCHED_COST = 1.0

# Cells whose lower-bound gap exceeds this are never solved exactly. Must be
# >= SIMILARITY_THRESHOLD or flagging would be wrong; the headroom only buys a
# wider exact range for the reported sweep.
# Raised from 0.5 alongside SIMILARITY_THRESHOLD=0.80 -- the invariant above is
# REPORT_CEILING >= SIMILARITY_THRESHOLD, and violating it would silently prune
# the very cells that ought to be flagged.
REPORT_CEILING = 1.0

# Slack on the two threshold comparisons. A gap of exactly zero -- two genuinely
# tied optimal assignments, which is the most ambiguous case there is -- comes
# out of floating-point summation as ~1e-16 rather than 0.0, and a bare ``>``
# would then prune it as "worse than optimal" and miss the flag entirely.
TIE_TOL = 1e-9

# Thresholds reported in the diagnostic sweep. Does not affect the output.
SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5]


# ---------------------------------------------------------------------------
# Global fragments  (build_global_fragments.py)
# ---------------------------------------------------------------------------

# idtracker.ai's own parameter, kept under its own name and default
# (confparams.py:26). A global fragment is only a candidate for accumulation when
# EVERY one of its fragments is at least this many frames long -- short ones carry
# too few images to learn an identity from. Pushed into upstream's config
# singleton via upstream's own setter, so the filter that consumes it is
# idtracker.ai's, untouched.
MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION = 4
