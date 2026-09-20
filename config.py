"""Every knob the user sets, in one place.

Each stage module used to carry its own constants. They are gathered here so the
whole pipeline can be configured from a single file, and so a value shared by two
stages (BODY_NODES, CENTROID_NODE) has exactly one definition rather than an
import chain between stages.

This module imports nothing from the rest of the pipeline, so it can be imported
from anywhere without a cycle. Stage modules bind their old names to these
values, so nothing else in them had to change:

    from config import MAX_CROP_AREA         # build_id_images.py

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
N_ANIMALS = 2

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
CENTROID_NODE = "torso"

# Second choice for TOP_NODE, used ONLY when TOP_NODE is NaN on that instance.
#
# The alignment needs a point in FRONT of the centroid to measure a heading
# from, and the pipeline's rule until now was all-or-nothing: if TOP_NODE was
# missing the instance produced no id-image row at all (build_id_images
# .collect_geometry -> rotation_matrix returns None -> n_skipped). On a skeleton
# where the nose is the tip of the animal that is a real cost -- the nose is the
# first thing SLEAP loses to occlusion, and the node just behind it is usually
# still there and points the same way.
#
# Set to a node name to enable the fallback, or None to keep the old
# all-or-nothing behaviour exactly. The fallback node must lie on the SAME body
# axis as TOP_NODE (both in front of CENTROID_NODE), because a heading measured
# from it has to mean the same thing as one measured from TOP_NODE -- the two
# populate one id-image set and the network sees no marker saying which was
# used. Substituting a node on a different axis would silently rotate part of
# the population by a different amount.
#
# If BOTH are NaN the instance is skipped as before: no row, counted in
# `n_no_alignment`. How many instances came in through the fallback is recorded
# per episode as the `n_top_fallback` HDF5 attribute (and `n_top_fallback_total`
# for the whole session), so a session is never silently part-aligned on one
# node and part on another.
TOP_NODE_FALLBACK = None

# The "bottom" of the animal, used only to decide how high the centroid sits in
# the canonical frame. NOTE: wingR is the lowest node in just 9/16 flies on frame
# 0 -- the rest are bottomed by the left wing or a hindleg -- and being the RIGHT
# wing makes it asymmetric under wing pose. Kept as specified.
BOTTOM_NODE = "tail_base"

# The animal's body outline, excluding limbs and wings. Legs and wings splay far
# past the body, so a hull over every node reports contact long before the bodies
# are anywhere near each other.
BODY_NODES = ["head", "left_head", "right_head", "torso", "left_hip", "right_hip",
              "tail_base"]

# Padding around the rotated keypoint extent, in pixels (rotate_boxes.py).
PAD = 0


# ---------------------------------------------------------------------------
# Colour mode  (build_id_images.py)
# ---------------------------------------------------------------------------

# Whether the source video is single-channel or three-channel colour. This is a
# DECLARATION, not a request for a conversion: nothing here converts RGB to grey
# or grey to RGB. It states what the frames are, and build_id_images.py checks
# every frame against it and refuses to run if they disagree (see
# rotate_boxes.check_frame). Getting this wrong is otherwise silent -- a colour
# video read as grey would have two thirds of its pixels dropped by the squeeze
# that used to be unconditional, and the identity network would train on the
# result without complaint.
#
# "GRAYSCALE"  frames are (H, W) or (H, W, 1); id-images are (n, H, W) uint8.
# "RGB"        frames are (H, W, 3);           id-images are (n, H, W, 3) uint8.
#
# The crop GEOMETRY is identical either way -- same H, same W, same rows, same
# affine. RGB only widens each pixel from one uint8 slot to three, so an RGB
# session's HDF5 files are 3x the bytes of the grayscale ones at exactly the
# same image dimensions.
#
# NOTE ON SLEAP: sleap-io applies its own conversion before we ever see a frame.
# If the .slp was built with `Video.grayscale = True` -- which is what SLEAP
# does by default, and what every .slp in this repo has -- then `lf.image` comes
# back (H, W, 1) no matter how colourful the underlying mp4 is. Selecting "RGB"
# against such a file is a real misconfiguration and is reported as one: the
# colour is not recoverable without rebuilding the .slp with grayscale off.
#
# Override without editing this file:
#
#     export SLEAP_IDTRACKER_COLOR_MODE=RGB
#
COLOR_MODE = (os.environ.get("SLEAP_IDTRACKER_COLOR_MODE") or "GRAYSCALE").upper()

if COLOR_MODE not in ("GRAYSCALE", "RGB"):
    raise ValueError(
        f"COLOR_MODE must be 'GRAYSCALE' or 'RGB', got {COLOR_MODE!r}. "
        "Set it in config.py or via SLEAP_IDTRACKER_COLOR_MODE."
    )

# Derived, not chosen -- kept here only because it is a pure restatement of the
# line above and every stage that allocates an array needs it.
N_CHANNELS = 3 if COLOR_MODE == "RGB" else 1


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


# SIMILARITY_METHOD choices:
#   "bounding_box"  IoU of axis-aligned boxes over the keypoints. Fast, and what
#                   SLEAP's own tracker uses -- but a box around a diagonal
#                   animal is mostly background (39% animal on this data), so
#                   two boxes can overlap heavily while the animals do not touch.
#   "keypoint"      OKS. Robust to missing keypoints; averages over visible ones.
#   "centroid"      exp(-distance / body_length). Position only, no shape.
#   "hull"          TRUE IoU of the convex silhouettes: intersection area over
#                   union area of the two hulls, computed with
#                   cv2.intersectConvexConvex and inclusion-exclusion. Uses the
#                   SAME hull the id-image was masked with -- surrounding
#                   keypoints when all present, Graham scan over the survivors
#                   otherwise -- so the polygon matched frame-to-frame is the
#                   polygon the identity network was shown. Slower than the box
#                   metric (a per-pair polygon clip rather than a vectorised
#                   rectangle overlap), and worth it when animals pass close
#                   enough that box overlap reports contact that never happens.


# ---------------------------------------------------------------------------
# Silhouette hull  (build_id_images.py)
# ---------------------------------------------------------------------------

# Which keypoints define the id-image silhouette, split by the role they play.
#
# SURROUNDING_KEYPOINTS are the outline: the nodes that, in a well-predicted
# instance, are the ones actually ON the convex hull. They are the default and
# the fast path -- when all of them are present the hull is exactly these, and
# no other node is even looked at.
#
# BOUNDED_KEYPOINTS are the nodes that normally sit strictly INSIDE that
# outline, so including them changes nothing while the outline is intact. They
# exist for the case where it is not: if any surrounding keypoint is missing the
# outline has a hole in it, and the hull is rebuilt by Graham scan over every
# keypoint still available, bounded and surrounding alike, so that the
# silhouette stays as close to the animal's true extent as the surviving
# evidence allows.
#
# This replaces the previous behaviour of hulling every finite node
# unconditionally, and it is NOT a no-op on complete instances. A node is only
# "bounded" as often as the animal's pose keeps it inside; MEASURED on
# mice_new.mp4, the torso lies ON the hull in 434/5404 complete instances (8%),
# and excluding it there costs a median 25.7% of hull area (max 58.6%) -- a
# bending mouse pushes its torso outside the line from its head nodes to its
# hips. So changing these lists changes the silhouettes the identity network
# trains on, and an existing session is not comparable to a new one without
# rebuilding its id-images.
#
# To reproduce the old behaviour exactly, put EVERY node in
# SURROUNDING_KEYPOINTS and leave BOUNDED_KEYPOINTS empty: the primary path then
# hulls all nodes, and any missing one sends it to the fallback, which also
# hulls all surviving nodes.
#
# Defaults below are the mouse skeleton with the torso treated as interior.
# That is a choice about this animal, not a fact about it -- see the measurement
# above before keeping it. Every name must exist in the .slp's skeleton.
SURROUNDING_KEYPOINTS = ["head", "left_head", "right_head",
                         "left_hip", "right_hip", "tail_base"]

BOUNDED_KEYPOINTS = ["torso"]




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
#   "body" -- BODY_NODES only. Legs and wings splay, so
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
#   "total"    -- (DEFAULT, user-specified) the raw increase in total assignment
#                 cost, i.e. an instance is in a crossing when the SECOND-BEST
#                 assignment costs within SIMILARITY_THRESHOLD of the best one.
#                 Because a permutation can never change just one pairing (the
#                 difference between two permutations is a cycle of length >= 2),
#                 the smallest possible change touches two edges and this
#                 quantity sits on roughly [0, 2].
#   "per_edge" -- divide that increase by the number of animals whose partner
#                 changed. Units are then "mean IoU given up per re-assigned
#                 animal". It catches long cycles that "total" misses: 16 animals
#                 each shifting one place is genuinely ambiguous, and has a large
#                 total but a small per-animal cost.
#
# WHY "total" IS THE DEFAULT -- MEASURED on Pletcher_10fly (18000 frames, 10
# flies, 149342 id-image rows). "per_edge" divides by the number of animals whose
# partner changed, which counts animals whose change cost NOTHING. When a
# detection drops out, the matrix gains a padding column at UNMATCHED_COST and
# the orphaned row's costs are then all 1.0 -- identical to any zero-IoU pair --
# so moving it is free. The cheapest alternative becomes "the moving animal goes
# unmatched, the orphan takes its column", whose entire cost is the IoU the
# moving animal forfeits. n_changed is 2 but only ONE edge pays, so per_edge
# halves the margin and reports IoU/2. The effective rule degenerates to "flag
# any instance whose successor IoU < 2 * threshold", which fires on isolated,
# cleanly tracked animals with no competitor anywhere near them. 15.52% of frame
# transitions on that clip have unequal counts and so open such a slot.
#
# Switching to "total" there, at the same threshold:
#     fragments                            6980 -> 3888   (-44%)
#     instances stranded in fragments < 4  6752 -> 3144   (-53%)
#     is_overlapping flagged              4.27% -> 1.81%
#
# The trade is that "total" is uniformly more permissive and gives up the
# long-cycle protection above. On a clip where animals genuinely cross, confirm
# real encounters are still flagged before trusting it.
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
SIMILARITY_METHOD = "bounding_box"   # bbox IoU

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
# NOTE: those minima were measured under GAP_SCALE = "per_edge". The default is
# now "total", on which the same alternatives score n_changed times higher
# (n_changed >= 2), so the figures above are lower bounds -- re-run the sweep
# build_overlaps.py prints before tuning any of these.
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
    "bounding_box": 0.3,
    "keypoint": 0.2,
    "centroid": 0.2,
    # "hull" lives on a DIFFERENT scale from "bounding_box" even though both are
    # IoU: a hull is ~40-60% of its own bounding box on this data (measured on
    # mice_new: 23652 px box vs 9202 px hull), so the same pair of animals scores
    # markedly lower here than on boxes. Start from the bbox value and re-tune
    # against the per-edge gap sweep build_overlaps prints; a threshold below the
    # method's minimum observed gap flags nothing at all.
    "hull": 0.3,
}
# NEIGHBOUR_RADIUS: spatial gate on the frame-to-frame cost matrix. Only pairs
# whose CENTROID_NODE positions lie within NEIGHBOUR_RADIUS * body_length of one
# another are scored; every other cell is set to similarity 0, i.e. cost 1.0 ==
# UNMATCHED_COST. Set to None to score all pairs.
#
# For the IoU-style metrics this is exact, not an approximation: two shapes
# whose centroids are further apart than their own extent cannot overlap, so the
# skipped cells were going to be 0 anyway. It removes the work, not the answer.
#
# For "centroid" it is NOT exact. exp(-d / body_length) is 0.368 at exactly one
# body length and never reaches 0, so a gate at 1.0 truncates a similarity the
# metric considers meaningful. Raise the radius (3.0 puts the truncation at
# 0.050) or set it to None if you are matching on centroids over long jumps.
# "keypoint" (OKS) decays on the scale of the animal itself and is ~0 by one
# body length, so the gate is effectively exact there too.
#
# The radius is in BODY LENGTHS, read from the id-image `body_length` attribute
# -- the same measured quantity the crops were sized with -- so it transfers
# between datasets without retuning.
NEIGHBOUR_RADIUS = 1.0

OVERLAP_NODES = "all"
OVERLAP_DIRECTION = "both"
MAX_FRAME_GAP = 1
GAP_SCALE = "total"

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
