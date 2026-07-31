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
CENTROID_NODE = config.CENTROID_NODE  # the origin we rotate about
PAD = config.PAD


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


def rotate_instance(image, inst, top_name=TOP_NODE, centroid_name=CENTROID_NODE, pad=PAD):
    """Rotate `image` so the animal is upright; return (crop, angle, n_points).

    Returns (None, None, 0) if either alignment keypoint is missing.
    """
    top_xy = node_xy(inst, top_name)
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


def montage(crops, cols=4, gap=6, bg=255):
    """Contact sheet of the crops, padded to a common cell. Viewing aid only."""
    ch = max(c.shape[0] for c in crops)
    cw = max(c.shape[1] for c in crops)
    rows = int(np.ceil(len(crops) / cols))
    sheet = np.full((rows * (ch + gap) + gap, cols * (cw + gap) + gap), bg, np.uint8)
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        y = gap + r * (ch + gap) + (ch - c.shape[0]) // 2
        x = gap + k * (cw + gap) + (cw - c.shape[1]) // 2
        sheet[y : y + c.shape[0], x : x + c.shape[1]] = c
    return sheet


def main(frame_idx=0):
    labels = sio.load_slp(str(SLP))
    lf = labels[frame_idx]
    image = lf.image
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()

    print(f"frame {lf.frame_idx}: {len(lf.instances)} instances")
    print(f"aligning {CENTROID_NODE!r} -> {TOP_NODE!r} onto the +y axis\n")
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
        cv2.imwrite(str(OUT / name), crop)
        crops.append(crop)
        print(f" {i:>4}  {angle:>6.1f}   {crop.shape[1]:>5}   {crop.shape[0]:>5}   {name}")
        written += 1

    cv2.imwrite(str(OUT / "_montage.png"), montage(crops))
    print("-" * 46)
    print(f"wrote {written}/{len(lf.instances)} rotated crops to {OUT}")
    print("contact sheet: _montage.png")
    assert written == len(lf.instances), "not every instance produced a crop"


if __name__ == "__main__":
    main()
