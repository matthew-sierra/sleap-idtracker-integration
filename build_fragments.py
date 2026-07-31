"""Assemble idtracker.ai ``Fragment`` objects from the linked id-image files.

Stage 3d. Runs after ``build_overlaps.py``, which wrote the identity graph as
``next_global_index``. This module only has to walk it.

A fragment is a maximal chain of linked instances. Because ``build_overlaps.py``
already refused to link across a change in ``is_overlapping``, every chain is
automatically homogeneous: a run of confidently-tracked frames becomes an
individual fragment, and a run where animals are huddled becomes its own
crossing fragment. No extra condition is needed here -- the constraint was
enforced when the edges were created.

WHY THE GRAPH IS A SET OF SIMPLE CHAINS
---------------------------------------
Upstream's blob graph can branch: ``blob.next`` is *every* next-frame blob whose
pixels overlap, so a blob may have several successors and fragmentation has to
stop wherever the chain is not 1-to-1 (``fragmentation.py:65``).

Ours cannot branch. ``next_global_index`` comes from a Hungarian assignment,
which is a permutation, so every instance has at most one successor AND at most
one predecessor. Fragment extraction is therefore just "start at every node with
no predecessor and walk forward", with no ambiguity to resolve. That is checked
rather than assumed -- see ``build_chains``.

FIELDS
------
Upstream's ``Fragment.__init__`` (``fragment.py:137``) is used unmodified:

    fragment_identifier  sequential, 0..N-1 in start-frame order
                         (ListOfFragments asserts i == fragments[i].identifier)
    start_frame          first frame of the chain
    end_frame            last frame + 1 -- EXCLUSIVE, per list_of_fragments.py:213
    images               episode-LOCAL row index, not global_index. Upstream
                         pairs it with `episodes` in `image_locations` and
                         `load_id_images` (py_utils.py:468) indexes the episode's
                         dataset directly with it.
    centroids            CENTROID_NODE position per frame, video coordinates.
                         Consumed for velocity / start_position / end_position.
    episodes             episode per image, so a fragment may span a boundary.
    is_an_individual     `not is_overlapping` -- the polarity flip lives here, at
                         the boundary, so idtracker.ai keeps its own convention
                         and needs no edit.
    exclusive_roi        -1, matching Blob's default (blob.py:86). No ROIs are
                         defined in this port.

Run:
    /opt/anaconda3/envs/sleap_id/bin/python src/sleap_idtracker/build_fragments.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# config must be imported BEFORE idtrackerai: it is what puts the bundled
# idtrackerai/src on sys.path, so a fresh clone resolves these imports without
# idtracker.ai having been pip-installed first.
import config  # noqa: E402

from idtrackerai.fragment import Fragment  # noqa: E402
from idtrackerai.list_of_fragments import ListOfFragments  # noqa: E402

REPO = config.REPO
OUT_DIR = config.OUT_DIR
FRAGMENTS_JSON = config.FRAGMENTS_JSON

N_ANIMALS = config.N_ANIMALS

NO_FRAGMENT = -1  # sentinel in the per-row fragment_identifier dataset


# ---------------------------------------------------------------------------
# Reading the graph back out
# ---------------------------------------------------------------------------


def load_graph(out_dir: Path) -> dict:
    """Gather every per-row field into arrays indexed by ``global_index``.

    global_index is a bijection onto 0..N-1 across all episodes, so it doubles
    as the node id of the graph and every field can be a flat array.
    """
    files = sorted(out_dir.glob("id_images_*.h5"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"no id_images_*.h5 in {out_dir}")

    rows = []
    for path in files:
        with h5py.File(path, "r") as fh:
            for name in ("is_overlapping", "next_global_index", "centroid"):
                if name not in fh:
                    raise KeyError(f"{path.name} has no {name!r}; "
                                   "run build_overlaps.py first")
            rows.append(dict(
                g=fh["global_index"][...], local=fh["local_index"][...],
                frame=fh["frame_numbers"][...], inst=fh["instance_idx"][...],
                over=fh["is_overlapping"][...], nxt=fh["next_global_index"][...],
                cent=fh["centroid"][...],
                episode=np.full(len(fh["global_index"]), int(fh.attrs["episode"])),
            ))

    n = sum(len(r["g"]) for r in rows)
    out = {k: np.zeros(n, dtype=v.dtype) if v.ndim == 1
              else np.zeros((n, v.shape[1]), dtype=v.dtype)
           for k, v in rows[0].items()}
    seen = np.zeros(n, dtype=bool)
    for r in rows:
        g = r["g"]
        if g.min() < 0 or g.max() >= n:
            raise ValueError(f"global_index out of range [0, {n})")
        if seen[g].any():
            raise ValueError("global_index collision across episodes")
        seen[g] = True
        for k, v in r.items():
            out[k][g] = v
    if not seen.all():
        raise ValueError(f"{int((~seen).sum())} global_index values never written")

    out["files"] = files
    out["n"] = n
    return out


def build_chains(nxt: np.ndarray) -> list[np.ndarray]:
    """Walk the link array into maximal chains, ordered by first node.

    Verifies the permutation property rather than trusting it: a node with two
    predecessors, or a cycle, would silently corrupt every fragment downstream.
    """
    n = len(nxt)
    prev = np.full(n, -1, dtype=np.int64)
    for g in range(n):
        j = int(nxt[g])
        if j < 0:
            continue
        if j >= n:
            raise ValueError(f"node {g} links to {j}, out of range")
        if prev[j] != -1:
            raise ValueError(f"node {j} has two predecessors ({prev[j]} and {g}); "
                             "the link array is not a partial permutation")
        prev[j] = g

    chains: list[np.ndarray] = []
    visited = np.zeros(n, dtype=bool)
    for start in np.flatnonzero(prev == -1):
        chain = [int(start)]
        visited[start] = True
        g = int(nxt[start])
        while g >= 0:
            if visited[g]:
                raise ValueError(f"cycle detected re-entering node {g}")
            visited[g] = True
            chain.append(g)
            g = int(nxt[g])
        chains.append(np.asarray(chain, dtype=np.int64))

    if not visited.all():
        # only reachable if a cycle has no entry point at all
        raise ValueError(f"{int((~visited).sum())} nodes are in a closed cycle "
                         "and belong to no chain")
    return chains


def merge_crossing_chains(graph: dict, chains: list[np.ndarray]) -> list[np.ndarray]:
    """Fuse crossing chains that are up at the same time into single node groups.

    A crossing chain is a run of instances the assignment could not tell apart.
    When two (or more) of them overlap in time, they are by construction the
    animals that were ambiguous *with each other*, so they are merged into one
    group: the "one big fragment" holding every individual involved in the
    crossing. Individual chains are never merged and pass through untouched.

    Merging is transitive -- if A co-occurs with B and B with C, all three become
    one group even when A and C do not themselves overlap. They are links in a
    single chain of ambiguity, and splitting them would mean deciding a boundary
    the evidence does not support.

    The resulting group is NOT a single animal's track. It holds several rows per
    frame, one per animal caught in the crossing, which is why the merged
    fragments are exempted from the one-row-per-frame invariants below.
    """
    is_crossing = [bool(graph["over"][c[0]]) for c in chains]
    spans = [(int(graph["frame"][c].min()), int(graph["frame"][c].max())) for c in chains]

    parent = list(range(len(chains)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    crossing_idx = [i for i, c in enumerate(is_crossing) if c]
    for ai, a in enumerate(crossing_idx):
        for b in crossing_idx[ai + 1:]:
            # closed intervals: touching at a single frame still counts as co-occurring
            if spans[a][0] <= spans[b][1] and spans[b][0] <= spans[a][1]:
                union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(len(chains)):
        groups.setdefault(find(i) if is_crossing[i] else -1 - i, []).append(i)

    out = []
    for members in groups.values():
        nodes = np.concatenate([chains[i] for i in members])
        # frame-major, then instance, so images/centroids/episodes stay aligned
        # with a stable, reproducible ordering
        order = np.lexsort((graph["inst"][nodes], graph["frame"][nodes]))
        out.append(nodes[order])
    return out


def make_fragments(graph: dict, chains: list[np.ndarray]) -> list[Fragment]:
    """One upstream Fragment per node group, numbered in start-frame order."""
    order = sorted(range(len(chains)),
                   key=lambda k: (int(graph["frame"][chains[k]].min()),
                                  int(graph["inst"][chains[k][0]])))
    fragments: list[Fragment] = []
    for identifier, k in enumerate(order):
        nodes = chains[k]
        frames = graph["frame"][nodes]
        over = graph["over"][nodes]

        if over.min() != over.max():
            raise AssertionError(f"fragment {identifier} mixes is_overlapping values; "
                                 "build_overlaps.py should not have linked these")

        crossing = bool(over[0])
        if not crossing:
            # An individual fragment is still one animal's track: one row per
            # frame, consecutive. Merged crossing fragments are exempt by design.
            if not np.array_equal(np.diff(frames),
                                  np.ones(len(frames) - 1, dtype=frames.dtype)):
                raise AssertionError(f"fragment {identifier} has non-consecutive frames")

        fragments.append(Fragment(
            fragment_identifier=identifier,
            start_frame=int(frames.min()),
            end_frame=int(frames.max()) + 1,  # exclusive, per list_of_fragments.py:213
            images=graph["local"][nodes].tolist(),
            centroids=[tuple(map(float, c)) for c in graph["cent"][nodes]],
            episodes=graph["episode"][nodes].tolist(),
            is_an_individual=not crossing,  # polarity flip, once, here
            exclusive_roi=-1,
        ))
    return fragments


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_identifiers(graph: dict, chains: list[np.ndarray],
                      fragments: list[Fragment]) -> np.ndarray:
    """Stamp each row with its fragment, and check the partition is total."""
    n = graph["n"]
    frag_of = np.full(n, NO_FRAGMENT, dtype=np.int64)
    order = {int(f.start_frame): None for f in fragments}  # keep flake quiet
    del order

    by_start = sorted(range(len(chains)),
                      key=lambda k: (int(graph["frame"][chains[k][0]]),
                                     int(graph["inst"][chains[k][0]])))
    for identifier, k in enumerate(by_start):
        nodes = chains[k]
        if (frag_of[nodes] != NO_FRAGMENT).any():
            raise AssertionError(f"fragment {identifier} claims an already-claimed node")
        frag_of[nodes] = identifier

    if (frag_of == NO_FRAGMENT).any():
        raise AssertionError(f"{int((frag_of == NO_FRAGMENT).sum())} instances "
                             "belong to no fragment")

    for path in graph["files"]:
        with h5py.File(path, "r+") as fh:
            g = fh["global_index"][...]
            if "fragment_identifier" in fh:
                del fh["fragment_identifier"]
            fh.create_dataset("fragment_identifier", data=frag_of[g], dtype=np.int64)
            fh.attrs["n_fragments"] = len(fragments)
    return frag_of


def main(out_dir: Path = OUT_DIR, json_path: Path = FRAGMENTS_JSON) -> None:
    t0 = time.time()
    print(f"reading the linked id-image files in {out_dir}")
    graph = load_graph(out_dir)
    n = graph["n"]
    n_animals = int(N_ANIMALS)
    observed_n_animals = int(np.bincount(graph["frame"]).max())
    print(f"  {n} instances, {graph['frame'].max() - graph['frame'].min() + 1} frames, "
          f"{observed_n_animals} instances per frame (max)")
    print(f"  n_animals = {n_animals} (config.N_ANIMALS)")

    # N_ANIMALS is how many identities to assign over the course of the video.
    # It says nothing about how many animals share any single FRAME, so a
    # disagreement with the busiest frame is reported and not enforced: the
    # animals are not required to coexist. Whether they all APPEAR is a separate
    # and stricter question, checked below once the fragments exist.
    if n_animals != observed_n_animals:
        print(f"  NOTE: the busiest frame holds {observed_n_animals} instances, "
              f"not {n_animals}; the animals need not coexist")
        if n_animals > observed_n_animals:
            print("        no frame can hold every animal at once, so expect "
                  "zero global fragments and a k-means++ identification")

    # The one hard requirement, and it is about simultaneity rather than about
    # N_ANIMALS. Contrastive learning is trained on pairs of animals visible AT
    # THE SAME TIME -- that is the only signal telling it two animals are
    # different (contrastive.py:291). A video that never shows two animals at
    # once offers no such pair and cannot be trained on, whatever N_ANIMALS says.
    #
    # Deliberately a maximum, not a total or a fraction: stretches of the video
    # where two animals are up together are enough, and they do not have to
    # coexist for the whole video or even for most of it.
    individual = ~graph["over"].astype(bool)
    if individual.any():
        together = np.bincount(graph["frame"][individual])
        max_together = int(together.max())
        n_frames_together = int((together >= 2).sum())
    else:
        max_together = n_frames_together = 0
    print(f"  {n_frames_together} frames hold 2+ individuals at once "
          f"(max {max_together} together)")
    if max_together < 2:
        raise ValueError(
            f"no frame in this video holds two individuals at the same time "
            f"(max {max_together}). Contrastive training learns identities from "
            "animals seen together, so there is nothing to learn from here. "
            "This is independent of N_ANIMALS.")
    print(f"  {int((graph['nxt'] >= 0).sum())} links, "
          f"{int(graph['over'].sum())} overlapping instances")

    print("walking the link graph into chains")
    raw_chains = build_chains(graph["nxt"])
    chains = merge_crossing_chains(graph, raw_chains)
    n_merged = len(raw_chains) - len(chains)
    lengths = np.array([len(c) for c in chains])
    print(f"  {len(raw_chains)} chains -> {len(chains)} node groups "
          f"({n_merged} crossing chain(s) merged into co-occurring groups)")
    print(f"  group lengths min {lengths.min()} "
          f"median {int(np.median(lengths))} max {lengths.max()}")
    fragments = make_fragments(graph, chains)

    # --- every declared animal has to appear at some point ------------------
    # N_ANIMALS identities get assigned over the course of the video, so all
    # N_ANIMALS animals must actually turn up somewhere in it. They do NOT have
    # to be onscreen together -- that is what makes this different from the
    # global-fragment condition, which wants all of them at once and whose
    # absence is merely handled rather than an error.
    #
    # Individual fragments are the check because a fragment is one animal's
    # continuous track, so distinct animals cannot share one. An animal that
    # appears therefore produces at least one individual fragment, and N animals
    # produce at least N of them. Fewer than N_ANIMALS individual fragments in
    # the whole video is thus positive proof that some declared animal never
    # appears. (The converse does not hold -- one animal repeatedly occluded
    # yields many fragments -- so this is a floor, not a count of animals.)
    n_individual_fragments = sum(f.is_an_individual for f in fragments)
    if n_individual_fragments < n_animals:
        raise ValueError(
            f"N_ANIMALS={n_animals} but the whole video contains only "
            f"{n_individual_fragments} individual fragment(s), so at most "
            f"{n_individual_fragments} distinct animals ever appear. "
            f"{n_animals - n_individual_fragments} declared animal(s) are "
            "never seen, and there is no evidence from which to give them an "
            "identity. Set config.N_ANIMALS to the number of animals that "
            "actually appear.")

    # Only now is it safe to write: everything above is read-only, so a failed
    # check leaves the id-image files exactly as build_overlaps.py left them
    # rather than half-stamped with identifiers from a rejected run.
    frag_of = write_identifiers(graph, chains, fragments)

    n_ind = sum(f.is_an_individual for f in fragments)
    print(f"\n{len(fragments)} fragments: {n_ind} individual, "
          f"{len(fragments) - n_ind} crossing")

    list_of_fragments = ListOfFragments(fragments, graph["files"], n_animals)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    list_of_fragments.save(json_path)
    print(f"saved {json_path}")

    # --- validation: the movie must be partitioned cleanly ----------------
    print("\n=== validation ===")
    total_images = sum(f.n_images for f in fragments)
    individuals = [f for f in fragments if f.is_an_individual]
    crossings = [f for f in fragments if not f.is_an_individual]
    # chains[] is in build order; fragment i was built from chains[order[i]]
    crossing_chains = [c for c in chains if bool(graph["over"][c[0]])]

    checks = {
        "every instance in exactly one fragment": bool((frag_of != NO_FRAGMENT).all()),
        "fragment lengths sum to instance count": total_images == n,
        "identifiers are 0..N-1 in order": [f.identifier for f in fragments]
                                           == list(range(len(fragments))),
        # Both of the next two are one-row-per-frame invariants, so they hold for
        # INDIVIDUAL fragments only. A merged crossing fragment deliberately
        # carries several rows per frame -- one per animal in the crossing -- so
        # applying them there would fail by construction.
        "end_frame - start_frame == n_images (individuals)": all(
            f.end_frame - f.start_frame == f.n_images for f in individuals),
        "no individual fragment repeats a frame": all(
            len({*graph['frame'][c].tolist()}) == len(c)
            for c in chains if not bool(graph["over"][c[0]])),
        "images are episode-local (< episode row count)": all(
            (np.asarray(f.images) >= 0).all() for f in fragments),
        # Merged crossing fragments: every row genuinely flagged, and the frame
        # span is covered densely (n_images == sum of per-frame instance counts).
        "every row of a crossing fragment is flagged overlapping": all(
            bool(graph["over"][c].all()) for c in crossing_chains),
        # A merged group must absorb EVERY overlapping row in the frames it
        # spans -- otherwise two animals ambiguous with each other ended up in
        # different fragments, which is the thing merging exists to prevent.
        "crossing fragment holds every flagged row in its span": all(
            len(c) == int((np.isin(graph["frame"], graph["frame"][c])
                           & graph["over"]).sum())
            for c in crossing_chains),
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    # Per frame, the fragments alive must exactly cover that frame's instances.
    alive = np.bincount(graph["frame"], weights=(frag_of >= 0).astype(float))
    per_frame = np.bincount(graph["frame"])
    covered = bool(np.array_equal(alive, per_frame.astype(float)))
    print(f"  {'PASS' if covered else 'FAIL'}  every frame fully covered by fragments")

    # Cross-check the identifiers written to disk against the objects.
    disk = []
    for path in graph["files"]:
        with h5py.File(path, "r") as fh:
            disk.append(fh["fragment_identifier"][...])
    disk_ids = np.concatenate(disk)
    print(f"  {'PASS' if set(disk_ids.tolist()) == set(range(len(fragments))) else 'FAIL'}"
          f"  on-disk identifiers match the {len(fragments)} objects")

    if not (all(checks.values()) and covered):
        raise AssertionError("fragment validation failed")

    print(f"\nexample: {fragments[0]}")
    print(f"  episodes spanned {sorted(set(fragments[0].episodes.tolist()))}, "
          f"distance travelled {fragments[0].distance_travelled:.0f} px")
    print(f"  first image_location {next(iter(fragments[0].image_locations))}")
    print(f"\nALL FRAGMENT CHECKS PASS   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
