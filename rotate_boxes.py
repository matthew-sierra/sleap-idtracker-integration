"""Egocentric rotation, frame 0 — align every animal to a common heading.

The user picks two keypoints:

    TOP_NODE       the body part that should end up pointing "up"   (head)
    CENTROID_NODE  the body part treated as the animal's origin     (thorax)

For each instance we measure the angle the CENTROID->TOP vector makes with the
+y axis, rotate by the negative of that angle so TOP lands directly above
CENTROID, then re-derive the bounding box from the rotated keypoints and crop.

Two implementation notes that matter:

1. IMAGE COORDINATES HAVE +y POINTING DOWN. The note's mental model ("y axis
   lines go vertically") is the y-up maths convention. The conversion is folded
   into `heading_angle_deg` below -- see the derivation there -- and the result
   is checked by assertion rather than trusted.

2. WE ROTATE IN THE FULL FRAME, NOT IN THE TIGHT CROP. Rotating an already-tight
   rectangle throws away whatever swings out past its corners, which would lose
   exactly the leg and wing tips the box was drawn to contain. Rotating about the
   thorax in the full frame and cropping afterwards is the same operation with
   nothing discarded.

Because the box is recomputed from the ROTATED keypoints, these crops are not
the frame-0 boxes re-cut: an animal lying diagonally has a large axis-aligned
box before rotation and a snug one after. That is the point -- it is also why
these crops are more uniform in size than the unrotated ones.

Run:
    /opt/anaconda3/envs/sleap_id/bin/python src/sleap_idtracker/rotate_boxes.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import sleap_io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

REPO = config.REPO
SLP = config.SLP
OUT = REPO / "crops_frame0_rotated"

# The two alignment keypoints, and the crop padding. Set in config.py.
TOP_NODE = config.TOP_NODE          # ends up pointing up
TOP_NODE_FALLBACK = config.TOP_NODE_FALLBACK  # used only when TOP_NODE is NaN
CENTROID_NODE = config.CENTROID_NODE  # the origin we rotate about
PAD = config.PAD
COLOR_MODE = config.COLOR_MODE
N_CHANNELS = config.N_CHANNELS


class ColorModeError(ValueError):
    """config.COLOR_MODE disagrees with the frames actually being read."""


def _where(frame_idx, source) -> str:
    """Message prefix. Only ever built on the error path."""
    if frame_idx is None:
        return ""
    at = f"frame {frame_idx}"
    if source is not None:
        at += f" of {source}"
    return at + ": "


def _n_channels(image) -> int:
    """Channel count, for reporting a mismatch. Error path only."""
    if image.ndim == 2:
        return 1
    if image.ndim == 3:
        return int(image.shape[2])
    return -1


def check_frame_grayscale(image, frame_idx=None, source=None):
    """(H, W) or (H, W, 1) -> (H, W). Bound to `check_frame` under GRAYSCALE.

    sleap-io hands back (H, W, 1) when Video.grayscale is set and (H, W) when it
    is not, so both are accepted and the trailing axis is dropped -- exactly the
    squeeze this replaced.
    """
    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[:, :, 0]
    elif image.ndim == 2:
        return image
    raise ColorModeError(
        f"{_where(frame_idx, source)}COLOR_MODE is 'GRAYSCALE' but this frame "
        f"has {_n_channels(image)} channels (shape {image.shape}) -- RGB video "
        "was provided.\n"
        "  Set COLOR_MODE = \"RGB\" in config.py, or export "
        "SLEAP_IDTRACKER_COLOR_MODE=RGB, to keep the colour."
    )


def check_frame_rgb(image, frame_idx=None, source=None):
    """(H, W, 3) -> unchanged. Bound to `check_frame` under RGB."""
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    found = _n_channels(image)
    raise ColorModeError(
        f"{_where(frame_idx, source)}COLOR_MODE is 'RGB' but this frame has "
        f"{found} channel{'' if found == 1 else 's'} (shape {image.shape}) -- "
        "grayscale video was provided.\n"
        "  Set COLOR_MODE = \"GRAYSCALE\" in config.py, or export "
        "SLEAP_IDTRACKER_COLOR_MODE=GRAYSCALE.\n"
        "  If the source mp4 really is colour, the .slp is what flattened it: "
        "sleap-io honours Video.grayscale, which SLEAP sets True by default. "
        "Re-export the predictions with grayscale off to recover the channels."
    )


# Bound ONCE, at import, from the declared mode -- not re-derived per frame. A
# video is entirely one thing or entirely the other, so which branch applies is
# known before the first frame is read; only *whether the frames agree with it*
# is a per-frame question, and that is all the chosen function asks. Callers say
# `check_frame(...)` and get the block for their mode with no dispatch in it.
check_frame = check_frame_grayscale if COLOR_MODE == "GRAYSCALE" else check_frame_rgb


def node_xy(inst, name):
    """(x, y) of a named node in full-frame coords, or None if missing/NaN.

    sleap-io 0.7.1's PredictedInstance has no mapping interface, so we index
    `inst.numpy()` -- which is (n_nodes, 2) in skeleton node order -- by the
    node's position in the skeleton.
    """
    names = [n.name for n in inst.skeleton.nodes]
    if name not in names:
        raise KeyError(f"node {name!r} not in skeleton; have {names}")
    xy = np.asarray(inst.numpy()[names.index(name)], dtype=float)
    return xy if np.isfinite(xy).all() else None


def heading_angle_deg(top_xy, centroid_xy):
    """Angle to hand cv2 so that `top` ends up directly above `centroid`.

    Derivation (image coords, +y down). cv2.getRotationMatrix2D(c, a, 1) maps a
    point offset (dx, dy) from the centre to:

        x' =  cos(a)*dx + sin(a)*dy
        y' = -sin(a)*dx + cos(a)*dy

    We want the TOP node to land at (0, -r): directly above, since +y is down.
    Setting x' = 0 and requiring y' < 0 gives a = atan2(dx, -dy).

    Sanity checks folded in: top already above (dx=0, dy=-r) -> 0 deg; top to the
    right (dx=r, dy=0) -> +90 deg; top below (dx=0, dy=r) -> 180 deg.

    This is the same quantity as "the angle the centroid->top line makes with the
    +y axis" in the y-up convention -- the sign flip on dy is the conversion.
    """
    dx, dy = top_xy - centroid_xy
    return float(np.degrees(np.arctan2(dx, -dy)))


def top_node_xy(inst, top_name=TOP_NODE, fallback_name=TOP_NODE_FALLBACK):
    """(x, y) of the alignment node, falling back to `fallback_name` when NaN.

    Returns (xy, used_fallback), or (None, False) when neither node is usable.
    `fallback_name` of None reproduces the old single-node behaviour exactly --
    the second lookup is not even attempted. See config.TOP_NODE_FALLBACK for
    why the fallback must sit on the same body axis as the primary.
    """
    xy = node_xy(inst, top_name)
    if xy is not None:
        return xy, False
    if fallback_name is None:
        return None, False
    xy = node_xy(inst, fallback_name)
    return (xy, True) if xy is not None else (None, False)


def rotate_instance(image, inst, top_name=TOP_NODE, centroid_name=CENTROID_NODE, pad=PAD,
                    fallback_name=TOP_NODE_FALLBACK):
    """Rotate `image` so the animal is upright; return (crop, angle, n_points).

    Returns (None, None, 0) if the centroid keypoint is missing, or if BOTH the
    top node and its configured fallback are.
    """
    top_xy, _used_fallback = top_node_xy(inst, top_name, fallback_name)
    centroid_xy = node_xy(inst, centroid_name)
    if top_xy is None or centroid_xy is None:
        return None, None, 0

    angle = heading_angle_deg(top_xy, centroid_xy)
    M = cv2.getRotationMatrix2D(tuple(centroid_xy), angle, 1.0)

    h, w = image.shape[:2]
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    # Push every keypoint through the SAME matrix, so we know where the body went.
    pts = np.asarray(inst.numpy(), dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts_rot = (M @ np.c_[pts, np.ones(len(pts))].T).T

    # Verify the rotation actually did what we claimed, rather than assuming it.
    top_rot = (M @ np.r_[top_xy, 1.0])
    cen_rot = (M @ np.r_[centroid_xy, 1.0])
    assert top_rot[1] < cen_rot[1], "top node did not end up above the centroid"
    assert abs(top_rot[0] - cen_rot[0]) < 1e-6, "top node not vertically aligned"

    x0, y0 = pts_rot.min(axis=0) - pad
    x1, y1 = pts_rot.max(axis=0) + pad
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        return None, angle, len(pts)
    return rotated[y0:y1, x0:x1], angle, len(pts)


def imwrite(path, img):
    """cv2.imwrite, but with our channel order.

    Frames arrive from sleap-io as RGB; cv2 writes the first channel to the blue
    plane. Writing a colour crop straight out therefore swaps red and blue and
    the diagnostic sheet -- whose only purpose is being looked at -- comes out
    wrong. Grayscale is passed through untouched.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(str(path), img)


def montage(crops, cols=4, gap=6, bg=255):
    """Contact sheet of the crops, padded to a common cell. Viewing aid only.

    Works for (H, W) and (H, W, 3) crops alike: the cell grid is computed from
    the leading two axes and any trailing channel axis is carried through to the
    sheet, so a colour crop tiles exactly as a grey one does.
    """
    ch = max(c.shape[0] for c in crops)
    cw = max(c.shape[1] for c in crops)
    tail = crops[0].shape[2:]          # () grayscale, (3,) RGB
    rows = int(np.ceil(len(crops) / cols))
    sheet = np.full((rows * (ch + gap) + gap, cols * (cw + gap) + gap, *tail),
                    bg, np.uint8)
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        y = gap + r * (ch + gap) + (ch - c.shape[0]) // 2
        x = gap + k * (cw + gap) + (cw - c.shape[1]) // 2
        sheet[y : y + c.shape[0], x : x + c.shape[1]] = c
    return sheet


def main(frame_idx=0):
    labels = sio.load_slp(str(SLP))
    lf = labels[frame_idx]
    image = check_frame(lf.image, lf.frame_idx, Path(SLP).name)

    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()

    print(f"frame {lf.frame_idx}: {len(lf.instances)} instances")
    print(f"aligning {CENTROID_NODE!r} -> {TOP_NODE!r} onto the +y axis"
          + (f" (fallback {TOP_NODE_FALLBACK!r} when {TOP_NODE!r} is NaN)"
             if TOP_NODE_FALLBACK else "") + "\n")
    print(" inst   angle   width  height   file")
    print("-" * 46)

    crops, written = [], 0
    for i, inst in enumerate(lf.instances):
        crop, angle, _ = rotate_instance(image, inst)
        if crop is None:
            reason = "missing alignment keypoint" if angle is None else "degenerate box"
            print(f" {i:>4}      --      --      --   SKIPPED ({reason})")
            continue
        name = f"fly{i:02d}.png"
        imwrite(OUT / name, crop)
        crops.append(crop)
        print(f" {i:>4}  {angle:>6.1f}   {crop.shape[1]:>5}   {crop.shape[0]:>5}   {name}")
        written += 1

    imwrite(OUT / "_montage.png", montage(crops))
    print("-" * 46)
    print(f"wrote {written}/{len(lf.instances)} rotated crops to {OUT}")
    print("contact sheet: _montage.png")
    assert written == len(lf.instances), "not every instance produced a crop"


if __name__ == "__main__":
    main()
