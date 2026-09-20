"""Assign identities from per-image silhouette instead of the P1/P2 cascade.

Alternative to ``build_identities.py`` stage 4. Same inputs, same outputs, a
different rule for turning the contrastive embedding into identities.

WHY THIS EXISTS
---------------
Upstream decides a fragment's identity from *vote counts*. ``predict`` gives
every image a hard nearest-centre label; ``set_identification_statistics``
reduces those to a bincount; ``set_P1_from_frequencies`` turns the bincount into
P1 (a base-2 softmax of the counts); ``compute_P2_vector`` multiplies P1 by
``prod(1 - P1)`` over coexisting fragments to enforce "two animals in one frame
are different animals".

Everything after the bincount is integer arithmetic. How *far* an image sat from
the decision boundary never enters. Two coexisting fragments whose images all
land in the same cluster therefore produce numerically identical P1 vectors, and
P2 degenerates:

    n images   1 - P1[winner]     P2 of each        how the tie breaks
       31        4.657e-10      [0.500, 0.500]   sub-nanoscale float tilt
       80        0.000e+00      [1.000, 0.000]   list order (lower identifier)

The 53-image cliff is float64: ``1 - P1[winner]`` underflows to exactly zero once
2**-n drops below eps/2. So two pairs in an identical epistemic state -- the model
cannot separate them -- resolve to opposite labels depending only on how many
images each fragment happens to hold. Worse, the underflow branch reports
``certainty_P2 = inf``, so those fragments sort to the FRONT of the cascade and
are marked ``identity_is_fixed``: the least-informed decisions are made first and
locked.

THE RULE HERE
-------------
Silhouette, per image, evaluated under each hypothesis.

For image i and candidate identity k, using the k-means partition as the fixed
reference structure:

    a_k(i) = mean distance from i to the images of cluster k
    b_k(i) = min over j != k of the mean distance from i to the images of cluster j
    s_k(i) = (b_k(i) - a_k(i)) / max(a_k(i), b_k(i))

which is exactly ``contrastive.silhouette_scores`` (contrastive.py:892) asking
"how well would image i sit in cluster k?" rather than scoring the label it was
given. A fragment's score for identity k is the mean of s_k over its images.

Coexisting fragments must take distinct identities. That constraint is upstream's
and is kept exactly; only the quantity being maximised has changed. It is applied
as a graph COLOURING, not a matching, because coexistence is not transitive -- a
long fragment can overlap one fragment early and another late without those two
ever sharing a frame, and treating its coexisting set as a clique would invent
constraints no frame imposes. Each connected component of the coexistence graph
is solved on its own; for two animals a bipartite component admits exactly two
proper colourings, so both are scored by total silhouette and the better wins.
Exact, not greedy.

This subsumes the tie-break rather than special-casing it. A pair the embedding
separates cleanly scores overwhelmingly for one colouring. A pair the embedding
collapses still gets decided -- but by which fragment sits deeper in the cluster,
which is evidence, rather than by float64 rounding, which is not.

WHAT THIS DOES NOT FIX
----------------------
Identity is still cluster membership, and the coexistence constraint is local.
Measured on this clip: 138 predicted fragments, 82 edges, 62 connected
components -- 51 of them isolated pairs, the largest spanning 4 fragments over
about 65 frames. So the constraint graph carries nothing across the video. If the
embedding puts one physical animal in different clusters at different points,
this method reproduces that flip exactly as the cascade does. It makes each local
decision on evidence; it cannot manufacture global consistency the embedding
lacks. Fixing that needs a second source of linkage -- motion continuity across
fragment boundaries, or upstream's accumulation protocol -- not a better
tie-break.

The --evaluate pass measures precisely that, using centroid continuity as an
external check. Positions are never an input to the assignment.

CROSSING FRAGMENTS
------------------
Handled as ``build_identities.py`` handles them: one forward pass per row,
nearest centre, no pooling. The rows of a crossing fragment are different
animals, so a fragment-level score would be meaningless. Unchanged here so the
two stage-4 implementations differ in exactly one place.

Run:
    /opt/anaconda3/envs/sleap_id/bin/python \\
        src/sleap_idtracker/build_identities_silhouette.py --evaluate
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from idtrackerai.base.network import ResNet18  # noqa: E402
from idtrackerai.base.network.device import DEVICE  # noqa: E402
from idtrackerai.list_of_fragments import ListOfFragments  # noqa: E402
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments  # noqa: E402
from idtrackerai.utils import conf  # noqa: E402
from idtrackerai.utils.py_utils import (  # noqa: E402
    load_id_images, nchw_for,
)

# The id-image -> NCHW layout for this session, chosen ONCE from the declared
# colour mode rather than re-derived from each batch's shape. Every video is
# wholly one mode or the other, and build_id_images.py has already refused to
# write a file that disagrees, so nothing downstream needs to look again.
TO_NCHW = nchw_for(config.N_CHANNELS)

SESSION_DIR = config.SESSION_DIR
ACCUMULATION_DIR = SESSION_DIR / "accumulation"
FRAGMENTS_JSON = config.FRAGMENTS_JSON
GLOBAL_FRAGMENTS_JSON = config.GLOBAL_FRAGMENTS_JSON

BATCH = 512


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


@torch.inference_mode()
def embed(model, paths, locations) -> np.ndarray:
    """Embed image locations, in the same preprocessing as collate_fun."""
    out = []
    for i in range(0, len(locations), BATCH):
        chunk = list(locations[i: i + BATCH])
        imgs = load_id_images(paths, chunk, verbose=False, dtype=np.float32)
        tensor = TO_NCHW(imgs).to(DEVICE) / 255
        out.append(model(tensor).numpy(force=True))
    return np.concatenate(out)


def kmeans_centres(model, lof, first_gfrag) -> np.ndarray | None:
    """Upstream's kmeans_init: per-animal mean embedding of the first global fragment.

    Returns None when upstream would fall back to k-means++ -- no global fragment,
    or the biggest one is under 30 images (contrastive.py:432-444).
    """
    if first_gfrag is None or first_gfrag.min_n_images_per_fragment < 30:
        return None
    locs, labels = [], []
    rng = np.random.default_rng(0)
    for k, frag in enumerate(first_gfrag):
        il = list(frag.image_locations)
        if len(il) > 500:
            il = [il[i] for i in rng.choice(len(il), 500, replace=False)]
        locs += il
        labels += [k] * len(il)
    emb = embed(model, lof.id_images_file_paths, locs)
    labels = np.asarray(labels)
    return np.asarray([emb[labels == k].mean(0) for k in range(lof.n_animals)])


# ---------------------------------------------------------------------------
# Silhouette under a hypothesis
# ---------------------------------------------------------------------------


def silhouette_matrix(emb: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    """(n_images, n_identities) -- silhouette of each image under each hypothesis.

    ``contrastive.silhouette_scores`` scores an image against the label it was
    given. This scores it against every label it *could* be given, which is what
    a decision needs. Column k of row i is s_k(i) as defined in the module
    docstring; the diagonal case (k == labels[i]) reproduces upstream's number
    exactly, self-distance excluded from the intra-cluster mean the same way.
    """
    members = [np.flatnonzero(labels == k) for k in range(n)]
    mean_d = np.empty((len(emb), n))
    for k, idx in enumerate(members):
        if len(idx) == 0:
            mean_d[:, k] = np.inf
            continue
        d = cdist(emb, emb[idx])
        # For an image already in k, upstream divides by (len - 1): its own zero
        # distance is in the sum but must not be in the count.
        own = labels == k
        totals = d.sum(1)
        counts = np.full(len(emb), len(idx), dtype=float)
        counts[own] -= 1
        mean_d[:, k] = totals / np.maximum(counts, 1)

    out = np.empty((len(emb), n))
    for k in range(n):
        a = mean_d[:, k]
        other = np.delete(mean_d, k, axis=1)
        b = other.min(1)
        out[:, k] = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    return np.nan_to_num(out)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def coexistence_graph(frags, scorable: set[int]) -> dict[int, set[int]]:
    """fragment_identifier -> the identifiers it shares at least one frame with.

    Coexistence is NOT transitive: a long fragment can overlap one fragment early
    and a different one late without those two ever being alive together. Treating
    a fragment's coexisting set as a clique would manufacture constraints that no
    frame actually imposes, so the constraint is kept as an edge list and solved
    as a colouring below.
    """
    adj: dict[int, set[int]] = {f.identifier: set() for f in frags
                               if f.identifier in scorable}
    for frag in frags:
        if frag.identifier not in scorable:
            continue
        for g in frag.coexisting_individual_fragments:
            if g.identifier in scorable and g.identifier != frag.identifier:
                adj[frag.identifier].add(g.identifier)
                adj[g.identifier].add(frag.identifier)
    return adj


def connected_components(adj: dict[int, set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    comps: list[list[int]] = []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            seen.add(u)
            stack += [v for v in adj[u] if v not in comp]
        comps.append(sorted(comp))
    return comps


def two_colouring(comp: list[int], adj: dict[int, set[int]]) -> dict[int, int] | None:
    """Proper 2-colouring of a component, or None if it contains an odd cycle."""
    colour = {comp[0]: 0}
    stack = [comp[0]]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in colour:
                colour[v] = 1 - colour[u]
                stack.append(v)
            elif colour[v] == colour[u]:
                return None
    return colour


def assign(frags, index_of, sil, n_animals, report):
    """fragment_identifier -> identity, maximising silhouette under coexistence.

    The constraint is upstream's -- fragments alive in the same frame are
    different animals -- but it is a graph colouring, not a matching. Each
    connected component of the coexistence graph is solved on its own:

    * n_animals == 2 and the component is bipartite: exactly two proper
      colourings exist (one and its complement). Both are scored by total
      silhouette and the better is taken. This is exact, not greedy.
    * otherwise: fragments are taken in descending order of decisiveness (best
      identity minus runner-up) and given the best identity not already held by
      an assigned neighbour. A fragment with no identity left is dropped to 0
      rather than guessed at.

    A component with an odd cycle has three fragments mutually coexisting with
    only two animals available, which means the segmentation produced a spurious
    individual. Reported, then handled greedily.
    """
    scores = {fid: sil[np.asarray(index_of[fid])].mean(0) for fid in index_of}
    adj = coexistence_graph(frags, set(scores))
    out: dict[int, int] = {}

    for comp in connected_components(adj):
        colour = two_colouring(comp, adj) if n_animals == 2 else None
        if colour is not None:
            straight = sum(scores[f][colour[f]] for f in comp)
            flipped = sum(scores[f][1 - colour[f]] for f in comp)
            pick = colour if straight >= flipped else {f: 1 - c for f, c in colour.items()}
            for f in comp:
                out[f] = pick[f] + 1
            report["margin"][tuple(comp)] = abs(straight - flipped)
        else:
            if n_animals == 2:
                report["odd_cycle"].append(comp)
            order = sorted(comp, key=lambda f: -(np.sort(scores[f])[-1]
                                                 - np.sort(scores[f])[-2]))
            for f in order:
                taken = {out[g] for g in adj[f] if g in out}
                free = [k for k in range(1, n_animals + 1) if k not in taken]
                if not free:
                    report["starved"].append(f)
                    continue
                out[f] = max(free, key=lambda k: scores[f][k - 1])

        # A component is "collapsed" where two fragments that must differ both
        # argmax to the same identity: the embedding supplied no separation and
        # this is exactly where the P1/P2 cascade has nothing to decide on.
        for f in comp:
            for g in adj[f]:
                if g > f and int(np.argmax(scores[f])) == int(np.argmax(scores[g])):
                    report["collapsed"].append({
                        "pair": (f, g),
                        "argmax": int(np.argmax(scores[f])) + 1,
                        "assigned": {f: out.get(f), g: out.get(g)},
                        "scores": {f: scores[f].copy(), g: scores[g].copy()},
                    })
    return out


@torch.inference_mode()
def crossing_identities(model, centres, lof) -> dict[int, list[int]]:
    """fragment_identifier -> per-ROW identity, nearest centre, no pooling."""
    out: dict[int, list[int]] = {}
    for frag in lof.fragments:
        if frag.is_an_individual:
            continue
        locs = list(frag.image_locations)
        emb = embed(model, lof.id_images_file_paths, locs)
        out[frag.identifier] = (cdist(emb, centres).argmin(1) + 1).tolist()
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_identities(lof, per_fragment, per_row, out_dir: Path):
    """Write the `identities` column into id-image HDF5s under out_dir."""
    src = [Path(p) for p in lof.id_images_file_paths]
    if out_dir.resolve() != src[0].parent.resolve():
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in src:
            shutil.copy2(p, out_dir / p.name)
    files = [out_dir / p.name for p in src]

    buf = {}
    for e, path in enumerate(files):
        with h5py.File(path, "r") as fh:
            buf[e] = np.zeros(len(fh["identities"]), dtype=np.int64)

    for frag in lof.fragments:
        locs = list(frag.image_locations)
        if frag.is_an_individual:
            ident = per_fragment.get(frag.identifier)
            if ident is None:
                continue
            for img, ep in locs:
                buf[int(ep)][int(img)] = ident
        else:
            for (img, ep), ident in zip(locs, per_row.get(frag.identifier, [])):
                buf[int(ep)][int(img)] = int(ident)

    for e, path in enumerate(files):
        with h5py.File(path, "r+") as fh:
            fh["identities"][...] = buf[e]
            fh.attrs["identities_source"] = "build_identities_silhouette.py"
    return np.concatenate([buf[e] for e in range(len(files))]), files


# ---------------------------------------------------------------------------
# External evaluation -- NOT an input to the assignment
# ---------------------------------------------------------------------------


def evaluate(files: list[Path], n_animals: int) -> None:
    """Look for identity swaps by centroid continuity. External check only.

    Deliberately independent of everything above: it reads the centroid column
    the id-image builder recorded, which the assignment never sees. For every
    step where an identity reappears, it asks whether that identity landed
    closer to where it was, or closer to where a DIFFERENT identity was. The
    latter is what a swap looks like from the outside, and it is found by
    scanning the whole video rather than by looking anywhere in particular.
    """
    frames, cents, idents = [], [], []
    for path in files:
        with h5py.File(path, "r") as fh:
            frames.append(fh["frame_numbers"][...])
            cents.append(fh["centroid"][...])
            idents.append(fh["identities"][...])
    frame = np.concatenate(frames)
    cent = np.concatenate(cents).astype(float)
    ident = np.concatenate(idents)

    keep = ident > 0
    frame, cent, ident = frame[keep], cent[keep], ident[keep]
    order = np.argsort(frame, kind="stable")
    frame, cent, ident = frame[order], cent[order], ident[order]

    last_pos: dict[int, np.ndarray] = {}
    last_frame: dict[int, int] = {}
    events = []
    for f in np.unique(frame):
        rows = np.flatnonzero(frame == f)
        here = {int(ident[r]): cent[r] for r in rows}
        for k, pos in here.items():
            if k in last_pos:
                own = float(np.linalg.norm(pos - last_pos[k]))
                others = {j: float(np.linalg.norm(pos - last_pos[j]))
                          for j in last_pos if j != k}
                if others:
                    j = min(others, key=others.get)
                    if others[j] < own:
                        events.append((int(f), k, j, own, others[j],
                                       int(f) - last_frame[k]))
        for k, pos in here.items():
            last_pos[k] = pos
            last_frame[k] = int(f)

    print("\n=== external check: centroid continuity ===")
    print("    (positions are used HERE ONLY -- never by the assignment)")
    print(f"  {len(np.unique(frame))} frames carry at least one identity")
    if not events:
        print("  no step where an identity landed nearer another identity's "
              "previous position -- CLEAN")
        return
    print(f"  {len(events)} step(s) where an identity landed nearer ANOTHER "
          f"identity's previous position:\n")
    print(f"    {'frame':>6} {'identity':>9} {'nearer to':>10} "
          f"{'own dist':>9} {'other dist':>11} {'gap':>5}")
    for f, k, j, own, oth, gap in sorted(events, key=lambda e: e[3] - e[4])[:20]:
        print(f"    {f:>6} {k:>9} {j:>10} {own:>9.1f} {oth:>11.1f} {gap:>5}")
    if len(events) > 20:
        print(f"    ... and {len(events) - 20} more")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=SESSION_DIR / "id_images_silhouette",
                    help="where to write id-image copies carrying the new identities. "
                         "Always a COPY -- the session's own id_images, and every "
                         "existing module, are left untouched by design.")
    ap.add_argument("--evaluate", action="store_true",
                    help="run the centroid-continuity check after assigning")
    ap.add_argument("--export", action="store_true",
                    help="also write a .slp, by calling build_sleap_tracks.main() "
                         "with this run's out-dir. That function already takes "
                         "slp_path/out_dir/dest as arguments, so stage 6 is reused "
                         "as-is -- no edit to it, and its validation still runs.")
    ap.add_argument("--no-write", action="store_true", help="analyse only")
    args = ap.parse_args()

    t0 = time.time()
    conf.set_parameters(
        MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION=(
            config.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION),
    )

    lof = ListOfFragments.load(FRAGMENTS_JSON)
    logf = ListOfGlobalFragments.load(GLOBAL_FRAGMENTS_JSON, lof.fragments)
    n = lof.n_animals
    print(f"{len(lof.fragments)} fragments, n_animals={n}, device={DEVICE}")

    first_gfrag = (
        max(logf.global_fragments, key=lambda g: g.minimum_distance_travelled)
        if logf.global_fragments else None
    )

    ckpt = ACCUMULATION_DIR / "contrastive_checkpoint.pt"
    if not ckpt.is_file():
        raise SystemExit(f"no contrastive checkpoint at {ckpt} -- run build_identities.py first")
    model = ResNet18.from_file(ckpt).to(DEVICE)
    model.eval()

    print("\n=== embedding ===")
    frags = [f for f in lof.fragments if f.is_an_individual
             and f.n_images >= conf.MIN_N_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION]
    locs, index_of, cursor = [], {}, 0
    for f in frags:
        il = list(f.image_locations)
        index_of[f.identifier] = list(range(cursor, cursor + len(il)))
        cursor += len(il)
        locs += il
    emb = embed(model, lof.id_images_file_paths, locs)
    print(f"  {len(emb)} images from {len(frags)} individual fragments, dim {emb.shape[1]}")

    init = kmeans_centres(model, lof, first_gfrag)
    if init is None:
        print("  k-means init: k-means++ (no usable global fragment)")
        km = MiniBatchKMeans(n, batch_size=1024, n_init=20, init="k-means++")
    else:
        print(f"  k-means init: global fragment at frame "
              f"{first_gfrag.first_frame_of_the_core}")
        km = MiniBatchKMeans(n, batch_size=1024, n_init=1, init=init)
    labels = km.fit_predict(emb)
    centres = km.cluster_centers_

    print("\n=== silhouette ===")
    sil = silhouette_matrix(emb, labels, n)
    own = sil[np.arange(len(emb)), labels]
    print(f"  mean silhouette of the k-means partition: {own.mean():.4f}")
    print(f"  per-image percentiles: " + "  ".join(
        f"p{q}={np.percentile(own, q):+.3f}" for q in (5, 25, 50, 75, 95)))

    print("\n=== assignment ===")
    report = {"collapsed": [], "odd_cycle": [], "starved": [], "margin": {}}
    per_fragment = assign(frags, index_of, sil, n, report)
    adj = coexistence_graph(frags, set(index_of))
    comps = connected_components(adj)
    print(f"  coexistence graph: {len(adj)} fragments, "
          f"{sum(len(v) for v in adj.values()) // 2} edges, "
          f"{len(comps)} connected components (largest {max(len(c) for c in comps)})")
    print(f"  {len(per_fragment)}/{len(frags)} individual fragments assigned "
          f"by exact 2-colouring maximising total silhouette")
    if report["odd_cycle"]:
        print(f"  {len(report['odd_cycle'])} component(s) NOT 2-colourable "
              f"(>{n} mutually coexisting fragments -> spurious detection); "
              f"handled greedily: {report['odd_cycle']}")
    if report["starved"]:
        print(f"  {len(report['starved'])} fragment(s) left at identity 0 -- no "
              f"identity free: {report['starved']}")

    spans = {f.identifier: (f.start_frame, f.end_frame) for f in frags}
    print(f"\n  {len(report['collapsed'])} coexisting pair(s) the embedding did "
          f"NOT separate -- both argmax to the same identity.")
    print("  This is exactly where the P1/P2 cascade has no evidence and breaks "
          "the tie by float64 rounding.")
    for c in report["collapsed"]:
        f, g = c["pair"]
        print(f"\n    frags {f} & {g}, frames {spans[f][0]}-{spans[f][1]} / "
              f"{spans[g][0]}-{spans[g][1]}   (both argmax to identity {c['argmax']})")
        for x in (f, g):
            s = c["scores"][x]
            print(f"      frag {x:>3}: silhouette per identity "
                  f"[{', '.join(f'{v:+.4f}' for v in s)}]"
                  f"  ->  ASSIGNED {c['assigned'][x]}")

    per_row = crossing_identities(model, centres, lof)
    print(f"\n  {len(per_row)} crossing fragment(s), rows predicted individually")

    written: list[Path] = []
    if not args.no_write:
        ident, written = write_identities(lof, per_fragment, per_row, args.out_dir)
        print(f"\n  wrote identities to {args.out_dir}  (copies; originals untouched)")
        print(f"    {int((ident > 0).sum())}/{len(ident)} rows assigned")
        counts = np.bincount(ident, minlength=n + 1)
        print("    rows per identity: " + ", ".join(
            f"{k}={counts[k]}" for k in range(1, n + 1)))
        print("\n  to export a .slp from these, point stage 6 at them:")
        print(f"    OUT_DIR={args.out_dir}  (config.OUT_DIR)")

    if args.evaluate:
        if not written:
            raise SystemExit("--evaluate needs the written files; drop --no-write")
        evaluate(written, n)

    if args.export:
        if not written:
            raise SystemExit("--export needs the written files; drop --no-write")
        import build_sleap_tracks  # noqa: E402  (imported here: only needed on export)

        slp = config.SLP
        suffix = ".idtracker_predictions.SILHOUETTE.slp"
        dest = slp.parent / (
            slp.name[: -len(config.PREDICTIONS_SUFFIX)] + suffix
            if slp.name.endswith(config.PREDICTIONS_SUFFIX)
            else slp.stem + suffix
        )
        print("\n" + "=" * 70)
        print("=== stage 6: build_sleap_tracks.main(), reused unmodified ===")
        print("=" * 70)
        build_sleap_tracks.main(slp_path=slp, out_dir=args.out_dir, dest=dest)

    print(f"\ndone   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
