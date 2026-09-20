# Introduction

One of SLEAP's biggest strengths is that, given extensive training data, it is able to learn to accurately predict animal poses. While SLEAP excels at generating accurate pose predictions, it struggles in tracking identities of individuals. While it allows users to track identity with different algorithms  (in particular, "simple" and "flow" tracking), these methods struggle in situations where animal body parts might be occluded from one frame to the next. These errors are particularly prevalent in social situations or situations where the animals become occluded, and they require a great amount of human correction to fix them.

idTracker.ai is a video-tracking system that attempts to overcome the difficulties associated with identity tracking. It uses representation learning to learn how to distinguish two different individuals from each other. However, the software *does not track pose*, and also requires users to manually threshold where animals on the screen might be. The framework relies on this thresholding in order to find "Blobs" on the screen (a crucial object that idTracker.ai gives its own class), which it then crops as examples to use in contrastive learning. The software is able to determine when these blobs are made out of more than one individual, but it requires an erosion process to actually get the blob down to the size of an individual. Crucially, SLEAP gives us enough information to be able to make these crops, and it inherently allows us to locate where an individual exists on a given page. While it doesn't give us enough information to create a mask that *only* contains information about the individual and none of the background, we can use the information to create bounding boxes or convex hulls which we could put into the contrastive learning model. With enough examples, the model should be able to "learn" what is background vs. what is animal, enabling us to separate animal identity without any of the thresholding or erosion normally required with idTracker.ai.

Thus, this project aims to integrate SLEAP and idTracker.ai together to produce high quality pose estimation alongside high quality identity tracking. The remainder of this document will describe the architecture used by the model, as well as provide an overview of how each of the individual modules works.

# Software

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

Then configure `config.py` for your dataset — see
[Configuration](#configuration) below. At minimum you must set `N_ANIMALS` and the skeleton node
names, because the defaults describe a different animal and the pipeline will not run against your
skeleton without them.

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

# Architecture

## Integration Architecture Overview

In order to run the program, we assume that the individual already has a file that contains SLEAP predictions. We go through the following steps:

1. Create episodes, similarly to how idTracker.ai already does
2. Create crops that could be used as training examples using SLEAP keypoint information and create rectangular images of uniform size that contain individuals oriented in the exact same way
3. Create an HDF5 file that indexes all the training examples according to how they are indexed in SLEAP
4. Detect crossings using information about how the instances overlap
5. Generate graphs by using information from the crossings where we can say an individual has certainly kept the same identity
6. Creation of different fragments where we can clearly identity individuals
7. Generate batches using the fragments that we have created. It only uses fragments where it is sure that individuals are separate from one another
8. Contrastive learning model training
9. Prediction
10. Identity clustering using k-means algorithms.
11. Store the identities on a parallel dataset in the HDF5 file.
12. Import the identities into the original SLEAP file as `filename.idtracker_predictions.slp` that we can open in the SLEAP GUI.

### Required User Inputs

Everything is in `config.py`. The shipped defaults describe a 7-node mouse skeleton — **they will
not match your data.** Each entry below says what you need to know about *your* animals and video in
order to fill it in.

#### 1. Must set — what you are tracking

| setting | what to provide |
|---|---|
| `N_ANIMALS` | How many animals are in the arena. Must be exact: the run stops if any frame contains more detections than this. |
| `TOP_NODE` | The node at the front of the animal — the one that should end up pointing upward once every crop is rotated to a common heading. Usually the head, nose or snout. Choose a node your detector rarely misses: an instance without it is dropped entirely and gets no identity. |
| `TOP_NODE_FALLBACK` | A second front-of-animal node to use when `TOP_NODE` is missing on an instance. Leave `None` if you have no sensible alternative, but setting one recovers a lot of otherwise-discarded animals. |
| `CENTROID_NODE` | The node at the animal's centre of mass — the point each crop is rotated around and the position used to track it through the arena. Usually the thorax, torso or trunk. Should be the most reliably detected node you have. |
| `BOTTOM_NODE` | The node at the rear of the animal, opposite `TOP_NODE`. The distance between the two defines the animal's body length, which sets crop size and the search radius. Usually the tail base or abdomen. |
| `SURROUNDING_KEYPOINTS` | The nodes that trace the animal's **outline** — those on its silhouette when seen from the camera. Ears, shoulders, hips, tail base. These are joined into the shape that is cut out of the frame as the animal's image. |
| `BOUNDED_KEYPOINTS` | The nodes that normally sit **inside** that outline, such as the neck or trunk. Listing them here says "expected to be interior"; they are only brought into the outline when one of the surrounding nodes is missing. |
| `BODY_NODES` | The nodes making up the animal's core body, excluding extremities that splay out unpredictably — legs, wings, tail tip. Only used if you set `OVERLAP_NODES = "body"`, but must name real nodes either way. |
| `COLOR_MODE` | Whether your video is `"GRAYSCALE"` or `"RGB"`. State it correctly: this is declared, not detected, and a mismatch stops the run rather than guessing. |

Node names must match your SLEAP skeleton exactly, including capitalisation.

#### 2. Should review — how animals are matched between frames

| setting | what to provide |
|---|---|
| `SIMILARITY_METHOD` | How to judge that a detection in one frame is the same animal as one in the next. `"bounding_box"` compares the boxes around them; `"hull"` compares their actual outlines; `"keypoint"` compares pose similarity; `"centroid"` compares distance moved. Outlines are the strictest, boxes the most forgiving. |
| `SIMILARITY_THRESHOLD` | How close a competing match has to be before the pipeline treats the two animals as confusable and refuses to carry identity across. One value per method, since the four are not on the same scale. If nothing is ever flagged, your value is below anything your data produces — `build_overlaps.py` prints the range it actually saw. |
| `NEIGHBOUR_RADIUS` | How far an animal can plausibly travel between consecutive frames, in body lengths. Anything further apart is not considered a possible match. `1.0` suits animals that move less than their own length per frame; raise it for fast animals or low frame rates, or set `None` to compare everything. |
| `MAX_CROP_AREA` | The largest image, in pixels, that the identity network should see. Crops bigger than this are scaled down, so a small value costs visual detail — check the size the run reports and whether it says the clamp engaged. |
| `MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION` | The shortest run of consecutive frames worth trying to identify. Anything shorter is left unidentified rather than guessed at. |
| `OVERLAP_NODES` | Which nodes to use when comparing animals between frames: `"all"` of them, or just `"body"` (your `BODY_NODES`). Use `"body"` if your animal has limbs or wings that move independently of the body. |
| `MAX_FRAME_GAP` | How many frames apart two detections can be and still be linked. `1` means consecutive frames only. |
| `OVERLAP_DIRECTION` | Whether to judge each detection by the link into it, out of it, or `"both"`. `"both"` is the cautious choice. |
| `GAP_SCALE` | How the margin between the best and second-best match is measured. Leave at `"total"`; `"per_edge"` divides that margin by the number of animals involved and will flag isolated, cleanly-tracked animals whenever a detection drops out. |
| `FRAMES_PER_EPISODE` | How many frames go into each intermediate file. Affects file count and memory use, not results. |
| `USE_P2_ASSIGNMENT` | Whether to let animals visible at the same time rule each other out when assigning identities. Normally `True`. |
| `SIZE_STAT`, `PAD` | How crop dimensions are chosen across the population, and how much margin to leave around each animal. |

`SIMILARITY_THRESHOLD` ships as:

```python
SIMILARITY_THRESHOLD = {
    "bounding_box": 0.3,
    "keypoint":     0.2,
    "centroid":     0.2,
    "hull":         0.3,
}
```

Only the entry matching your `SIMILARITY_METHOD` is used. These numbers were measured on other
datasets — treat them as starting points, not defaults that will suit your animals.

#### 3. Leave alone unless you know why

`N_CHANNELS` is derived from `COLOR_MODE`. `UNMATCHED_COST`, `REPORT_CEILING`, `TIE_TOL` and `SWEEP`
are solver internals; `REPORT_CEILING` only widens the range of exactly-reported gaps and, under
`GAP_SCALE = "total"`, a high value disables the pruning that keeps the assignment search fast.

#### 4. Paths — environment variables, not `config.py`

```bash
export SLEAP_IDTRACKER_DATA=/path/to/experiment     # data root
export SLEAP_IDTRACKER_SLP=/path/to/input.predictions.slp
export SLEAP_IDTRACKER_SESSION=/path/to/session_dir # pipeline output
export SLEAP_IDTRACKER_COLOR_MODE=RGB               # optional
```

#### Before a long run

`build_id_images.py` runs a preflight that aborts on more instances than `N_ANIMALS` in a frame, or
too few keypoints to build any crop. It then prints the chosen crop size, whether the area clamp
engaged, and how many instances were skipped for a missing alignment node. Those four numbers tell
you whether the configuration is right before you spend GPU time on it.
### 1. Episode segmentation

Handled by `episodes.py`

This step gives us chunks of 500 frames called "episodes" that we later use to complete some of these processes in parallel. We just split the SLEAP frames into several different parts and end up with Episode objects that idTracker.ai is already familiar with.

### 2. Image Normalization

Handled by `build_id_images.py` with variables and methods imported from `rotate_boxes.py`

In order to extract information from the video to generate examples that could be used by the training model, we go through the following algorithm:

1. For each fly, rotate the whole frame according to the centroid of the fly.
2. Measure information about distance between the bottom, centroid, and top nodes that were configured by the user, as well as information about the height and width of each instance.
3. Rotate all the images so that whatever the user chooses as top node is at the top. Create convex hulls for each individual.
4. Create screenshots by placing the instances in black bounding boxes whose size we determined using size statistics that we derived by looking at all the individuals that appear in the video.

 *Please note: this step will crop out some information from the instance's bodies, but this should still allow us to get valid training examples. Unless we add excessive padding which will increase the size of the inputs needed for training, this is inevitable.*

In alignment with idTracker.ai, if the size of the black bounding box from step 4 exceeds 6400 pixels, all of the images will be uniformly downscaled until they fit in a box of at most that size.

### 3. HDF5 Creation

Handled by `build_id_images.py`

Creates an HDF5 that stores all of the screenshots that we created in the previous step. There are as many files as there are episodes, and each channel contains the index of the keypoints relative to the episode stored as `instance_idx` (which we derive from the index that SLEAP gives us) and the actual images themselves.

### 4. Detect Crossings

Handled by `build_overlaps.py`, `assignment_margin.py`

We developed a novel way to detect crossings (or rather, instances of overlap) by using information from the cost matrix. Say that we have a simple cost matrix, where the rows represent all the instances on frame $I$, and the columns represent the instances on frame $I+1$.
We say that a crossing occurs if given some permutation $\sigma_0$ that maximizes the score, we can create another permutation $\sigma_1$ that swaps two of the choices and gives a score that is within some amount of the maximum (a threshold configured by the user).

```
          i_1     i_2     i_3
         _______ _______ _______
i_1     |       |       |       |
        |  0.6  |  0.5  |  0.0  |
        |_______|_______|_______|
        |       |       |       |
i_2     |  0.5  |  0.5  |  0.3  |
        |_______|_______|_______|
        |       |       |       |
i_3     |  0.1  |  0.2  |  1.0  |
        |_______|_______|_______|

Here, we could pick the permutation {i_{1,1}, i_{2,2}, i_{3,3}} to maximize the total similarity score, but we can also pick {i_{2,1}, i_{1,2}, i_{3,3}} to get an amount that is very close to the maximum. The user is able to configure this maximum.

```

We choose this method rather than any kind of other overlap method so that we could use information that is automatically generated by SLEAP during the prediction process. However, it is important to note that as of now, we compute the cost matrices ad-hoc.

### 5. Graph tracking individuals

Handled by `build_overlaps.py` at the same time that the cost matrix is being computed.

We need this graph so that we could find stretches where we are certain that one instance is the same identity to create pairs for the contrastive learning model.

### 6. Fragmentation

Handled by `build_fragments.py`, `build_global_fragments.py`

When we try to find consecutive sequences of frames that we could attribute to the same individual animal or group of animals, we create fragments. This process relies on the graphs built in `build_overlap.py`. We need this fragmenting stage so that we could create training examples.

We handle fragmentation a little bit differently to take advantage of the fact that SLEAP already helps us separate individuals from each other:

1. Individual fragments
        - If we have stretches of frames that we can surely attribute to the same animal, we put them into an individual fragment to be sent into the model
2. Overlapping fragments.
        - If we find that there is a high probability that we could have swapped the identity of a specific individual based on the crossing detection method, we add the individual to a fragment that stores the identity of all the individuals that might be overlapping. These are not used to train the model, but we extract the individual identities from this fragment during the prediction stage.

There is another important fragment created called a *global fragment*. A frame is a global fragment if it contains $n$ individual fragments, where $n$ is the number of animals that the user initially configured.

### 7. Batch Creation, 8. Training, 9. Prediction, 10. Identity Clustering

All of these steps are handled by modules that come from idTracker.ai, so we will not discuss them in depth. The biggest change that we have made is that the ResNet-18 architecture used by idTracker.ai no longer requires square inputs; so, it is now able to rectangular inputs of the size that we determined in the image normalization stage.

In the prediction stage, we make a slight optimization that takes advantage of the fact that we have multiple SLEAP identities already. However, we assign identities to the instances in overlapping fragments by simply iterating over the instances in the fragments.

Identity clustering simply uses the k-means clustering algorithms that were already used in idTracker.ai

### 11. Identity assignment

Handled by  `build_identities.py`

Using the model that we trained in step 8, we assign identities to all of the individuals that appear in the h5 file.

### 12. Importing into SLEAP

Handled by `build_sleap_tracks.py`

Since we used the indexes that were defined in the original SLEAP module, we can hand each instance identity from the implementation to the instance that it corresponds to in SLEAP. We store all of the information in a `.idtracker_predictions.slp` so that they could be opened in the SLEAP GUI.

# Future Improvements

1. Make this code a downloadable package so that a user can run it from the terminal
2. Optimizing idTracker.ai's feature-tracking so it is compatible with individuals of different sizes, not just small flies.
3. Improving the way that we generate the convex hulls
4. Making it so that the process runs at the same time as the SLEAP prediction. As things stand, we first run SLEAP prediction, receive a .prediction.slp file, and *then* run the idTracker.ai integration. However, it would be very helpful if these didn't need to be separate steps, and we could configure everything in one go
        1. Want to ensure that idTracker.ai will be cognizant of the `similarity_method` the user configures for SLEAP tracking and use the cost matrix information properly.
        - *this would help our model be more efficient, since we could just hijack the information given in the cost matrices and run the process in one fell swoop. It takes 3 seconds for 2000 frames, so if we wanted to track for an hour long video, for this specific setup, it would take about 10 minutes to compute the matrices for the whole video; we should certainly try to cut down on this time if it's possible!*
5. Make required user inputs something that we could initialize upon running the module (as in, --bottom_node = x, etc)
6. We want to only take a maximum input size
7. Add in impossible velocity thresholding in the same way that idTracker.ai does it so that I could find out whether or not there are impossible identity change. This is something that I chose not to handle in this initial stage, but would be helpful to include a little bit more insight on later
        - would probably be useful considering that we also have individual readings that might not always work
8. Improving memory allocation and setting batches so that we could more efficiently break up the work that we are doing
