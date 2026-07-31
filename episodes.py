"""Divide a SLEAP ``.slp`` predictions file into idtracker.ai "episodes".

What this mirrors
-----------------
idtracker.ai splits a video into fixed-length *episodes* so that later stages
(segmentation, crop extraction, fragmentation) can be farmed out to parallel
worker processes. The canonical implementation is the static helper
``Session.get_processing_episodes`` in::

    idtrackerai/src/idtrackerai/session.py:868

This module reproduces that function's arithmetic **exactly**, including the
slightly surprising ``frames_per_episode + 1`` divisor and the ``np.linspace``
boundary walk. The goal is byte-for-byte agreement with upstream episode
boundaries, not "better" episodes -- downstream idtracker.ai code indexes into
these boundaries, so any deviation is a bug even if it looks tidier.

The one substantive difference from upstream
--------------------------------------------
Upstream discovers the frame count by opening each video with OpenCV::

    int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))

We do **not** open the video. Episode segmentation is pure interval arithmetic,
so the frame count is read from the ``.slp`` metadata instead (the labeled frame
indices). This keeps the step fast and makes it work when the mp4 is not on this
machine. Everything downstream of the frame count is upstream's logic verbatim.

Reusing the real ``Episode``
---------------------------
We import idtracker.ai's own ``Episode`` dataclass
(``idtrackerai/src/idtrackerai/utils/py_utils.py:176``) rather than redefining
it, so objects produced here are accepted by unmodified idtracker.ai code and
round-trip through its ``json_object_hook`` deserializer (py_utils.py:528).

Import strategy, in order of preference (see ``_import_episode``):

1. Plain ``from idtrackerai.utils.py_utils import Episode`` after prepending
   ``idtrackerai/src`` to ``sys.path``. This is the correct path and the one
   that will be taken once the environment is complete.
2. A fallback that loads ``py_utils.py`` in an *isolated* synthetic package when
   step 1 fails on missing third-party dependencies.

Step 2 is needed today: the ``sleap_id`` conda env lacks ``toml``,
``deprecated`` and ``requests``. Those are imported by
``idtrackerai/__init__.py`` -> ``utils/__init__.py`` -> ``logging_utils`` ->
``telemetry``, none of which episode segmentation touches. Since we may not
pip-install and may not edit ``idtrackerai/``, the fallback stubs the missing
modules and loads ``py_utils.py`` directly by file path. **Zero files under
``idtrackerai/`` are modified.**

Caveat worth knowing: under the fallback the loaded class is a distinct class
object from what a fully-installed ``idtrackerai`` would give you, so
``isinstance`` checks across the two import routes would fail. The fields and
behaviour are identical because it is literally the same source file. Installing
``toml``, ``deprecated`` and ``requests`` makes route 1 succeed and the caveat
disappear. See ``_report_episodes.md``.

Single-video assumption
-----------------------
Only one video is processed at a time for now, so ``local_* == global_*``. That
is asserted, not assumed. Frame indices are **absolute**: idtracker.ai allocates
trajectory arrays to the full video length and writes at the global frame
number, so no offset is ever introduced here.
"""

from __future__ import annotations

import sys
import types
from itertools import pairwise
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Was `Path(__file__).resolve().parents[2] / "idtrackerai" / "src"`, computed
# here. idtracker.ai now lives inside this repo and config owns the one
# definition of where, so this module no longer derives a root of its own.
_IDTRACKERAI_SRC = config.IDTRACKERAI_SRC

DEFAULT_FRAMES_PER_EPISODE = config.FRAMES_PER_EPISODE
"""Upstream default, ``idtrackerai/src/idtrackerai/session.py:101``."""

DEFAULT_SLP_PATH = config.SLP


# --------------------------------------------------------------------------
# Importing idtracker.ai's Episode without editing or installing anything
# --------------------------------------------------------------------------


def _install_dependency_stubs() -> list[str]:
    """Register minimal stand-ins for third-party modules ``py_utils`` imports
    but that episode segmentation never exercises.

    Only fills genuine gaps: anything already importable is left untouched.
    Returns the names that were stubbed, for reporting.
    """
    stubbed: list[str] = []

    # `toml` is used solely by `load_toml()`, which we never call. Back the stub
    # with stdlib `tomllib` so it is at least functional rather than a landmine.
    try:
        import toml  # noqa: F401
    except ImportError:
        import tomllib

        toml_stub = types.ModuleType("toml")

        def _load(f: Any) -> dict:
            # tomllib requires binary mode; py_utils passes a text handle.
            return tomllib.load(getattr(f, "buffer", f))

        toml_stub.load = _load  # type: ignore[attr-defined]
        toml_stub.loads = tomllib.loads  # type: ignore[attr-defined]
        sys.modules["toml"] = toml_stub
        stubbed.append("toml")

    # `deprecated.sphinx.deprecated` is a documentation decorator; a pass-through
    # preserves the decorated functions' behaviour exactly.
    try:
        import deprecated.sphinx  # noqa: F401
    except ImportError:
        dep = types.ModuleType("deprecated")
        sphinx = types.ModuleType("deprecated.sphinx")

        def _deprecated(*_args: Any, **_kwargs: Any):
            return lambda fn: fn

        sphinx.deprecated = _deprecated  # type: ignore[attr-defined]
        dep.sphinx = sphinx  # type: ignore[attr-defined]
        sys.modules["deprecated"] = dep
        sys.modules["deprecated.sphinx"] = sphinx
        stubbed.append("deprecated")

    return stubbed


def _import_episode() -> tuple[type, str]:
    """Return idtracker.ai's real ``Episode`` class plus a note on how it loaded.

    Never modifies anything under ``idtrackerai/``.
    """
    if str(_IDTRACKERAI_SRC) not in sys.path:
        sys.path.insert(0, str(_IDTRACKERAI_SRC))

    # Route 1: the real package import. Preferred; works once the env has
    # toml + deprecated + requests. Equivalent alternative for the future is an
    # editable install: `pip install -e idtrackerai/`, which removes the need
    # for the sys.path line above entirely.
    try:
        from idtrackerai.utils.py_utils import Episode  # type: ignore

        return Episode, "package import (idtrackerai.utils.py_utils)"
    except ImportError as exc:
        package_error = exc

    # Route 2: isolated load of the same source file.
    #
    # py_utils.py does `from .rich_utils import track`, a relative import, so it
    # needs a parent package. We synthesise one pointing at the real utils/
    # directory. Deliberately named `_sleap_idtracker_idt_utils` rather than
    # `idtrackerai.utils` so we do NOT poison sys.modules["idtrackerai"] -- a
    # later, correct `import idtrackerai` must still get the real package.
    import importlib.util

    _install_dependency_stubs()

    utils_dir = _IDTRACKERAI_SRC / "idtrackerai" / "utils"
    py_utils_file = utils_dir / "py_utils.py"
    if not py_utils_file.is_file():
        raise FileNotFoundError(f"{py_utils_file} not found")

    pkg_name = "_sleap_idtracker_idt_utils"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(utils_dir)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg

    mod_name = f"{pkg_name}.py_utils"
    if mod_name in sys.modules:
        module = sys.modules[mod_name]
    else:
        spec = importlib.util.spec_from_file_location(mod_name, py_utils_file)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    return module.Episode, (
        f"isolated file load of {py_utils_file} "
        f"(package import unavailable: {package_error})"
    )


Episode, EPISODE_IMPORT_ROUTE = _import_episode()


def _resolve_path(path: Path | str) -> Path:
    """Mirror of ``idtrackerai.utils.py_utils.resolve_path`` (py_utils.py:538).

    Reimplemented rather than imported so this module works even if the loaded
    ``py_utils`` ever changes shape; the behaviour is identical.
    """
    return Path(path).expanduser().resolve()


# --------------------------------------------------------------------------
# Frame-count discovery from the .slp (no pixel data touched)
# --------------------------------------------------------------------------


def read_slp_frame_info(slp_path: Path | str) -> tuple[int, Path, list[int]]:
    """Read frame count, video path and sorted frame indices from a ``.slp``.

    Returns ``(n_frames, video_path, frame_indices)``.

    ``n_frames`` is ``max(frame_idx) + 1``, i.e. the length of the address space
    the labels live in -- NOT ``len(labeled_frames)``. Those differ if the
    predictions are sparse, and the address-space definition is the correct one
    because idtracker.ai indexes trajectories by absolute frame number. Sparse
    input is flagged with a warning since downstream stages assume dense frames.

    The mp4 is never opened.
    """
    import sleap_io as sio

    labels = sio.load_slp(str(slp_path))

    if len(labels.videos) != 1:
        raise NotImplementedError(
            "Only single-video sessions are supported for now; "
            f"{slp_path} references {len(labels.videos)} videos."
        )

    video_path = _resolve_path(labels.videos[0].filename)

    frame_indices = sorted(lf.frame_idx for lf in labels.labeled_frames)
    if not frame_indices:
        raise ValueError(f"{slp_path} contains no labeled frames")

    n_frames = int(frame_indices[-1]) + 1

    if frame_indices != list(range(n_frames)):
        import warnings

        missing = n_frames - len(frame_indices)
        warnings.warn(
            f"{slp_path} frame indices are not contiguous 0..{n_frames - 1} "
            f"({missing} frame(s) absent). Episodes still tile the full "
            "[0, n_frames) range, but downstream stages expect dense frames.",
            stacklevel=2,
        )

    return n_frames, video_path, [int(i) for i in frame_indices]


# --------------------------------------------------------------------------
# The port of Session.get_processing_episodes
# --------------------------------------------------------------------------


def get_processing_episodes(
    video_paths: Sequence[Path | str],
    video_paths_n_frames: Sequence[int],
    frames_per_episode: float = DEFAULT_FRAMES_PER_EPISODE,
    tracking_intervals: Any = None,
) -> tuple[int, list[int], list[list[int]], list[Any]]:
    """Port of ``Session.get_processing_episodes`` (session.py:868).

    Identical to upstream except that per-path frame counts are *supplied* by
    the caller (read from the ``.slp``) instead of being probed with
    ``cv2.VideoCapture``. The existence check on video paths is also dropped,
    since we never read pixels and the mp4 may legitimately be elsewhere.

    Every line below this point is upstream's logic, kept deliberately verbatim
    -- including ``int((end - start) / (frames_per_episode + 1))``. That ``+ 1``
    is not a typo on our side; do not "fix" it.
    """

    def in_which_interval(frame_number, intervals) -> int | None:
        for i, (start, end) in enumerate(intervals):
            if start <= frame_number < end:
                return i
        return None

    video_paths_n_frames = list(int(n) for n in video_paths_n_frames)
    if len(video_paths_n_frames) != len(video_paths):
        raise ValueError("video_paths and video_paths_n_frames length mismatch")
    for n_frames, video_path in zip(video_paths_n_frames, video_paths):
        if n_frames <= 0:
            raise ValueError(f"Non-positive frame count for {video_path}")

    number_of_frames = sum(video_paths_n_frames)

    # set full tracking interval if not defined
    if tracking_intervals is None:
        tracking_intervals = [[0, number_of_frames]]
    elif isinstance(tracking_intervals[0], int):
        tracking_intervals = [tracking_intervals]

    # find the global frames where the video path changes
    video_paths_changes = [0] + list(np.cumsum(video_paths_n_frames))

    # [[first frame of path 0, last frame of path 0], [path 1 ...], ...]
    video_paths_intervals = list(pairwise(video_paths_changes))

    # frames where a tracking interval starts or ends
    tracking_intervals_changes = list(np.asarray(tracking_intervals).ravel())

    # union of both kinds of boundary
    limits = video_paths_changes + tracking_intervals_changes
    limits = sorted(set(limits))

    # "long episodes": spans between any boundary, kept only if inside a
    # tracking interval
    long_episodes = []
    for start, end in pairwise(limits):
        if (
            in_which_interval(start, tracking_intervals) is not None
        ) and 0 <= start < number_of_frames:
            long_episodes.append((start, end))

    # subdivide long episodes to respect frames_per_episode
    index = 0
    episodes: list[Any] = []
    for start, end in long_episodes:
        video_path_index = in_which_interval(start, video_paths_intervals)
        assert video_path_index is not None
        global_local_offset = video_paths_intervals[video_path_index][0]

        n_subepisodes = int((end - start) / (frames_per_episode + 1))
        new_episode_limits = np.linspace(start, end, n_subepisodes + 2, dtype=int)
        for new_start, new_end in pairwise(new_episode_limits):
            episodes.append(
                Episode(
                    index=index,
                    local_start=new_start - global_local_offset,
                    local_end=new_end - global_local_offset,
                    video_path=_resolve_path(video_paths[video_path_index]),
                    global_start=new_start,
                    global_end=new_end,
                    # bbox_images intentionally left None -- populated by the
                    # crop-extraction stage, which is another agent's territory.
                )
            )
            index += 1

    return number_of_frames, video_paths_n_frames, tracking_intervals, episodes


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def episodes_from_slp(
    slp_path: Path | str = DEFAULT_SLP_PATH,
    frames_per_episode: float = DEFAULT_FRAMES_PER_EPISODE,
    tracking_intervals: Any = None,
) -> tuple[list[Any], int, Path, list[int]]:
    """Build idtracker.ai episodes for a single-video ``.slp`` predictions file.

    Returns ``(episodes, n_frames, video_path, frame_indices)``.
    """
    n_frames, video_path, frame_indices = read_slp_frame_info(slp_path)

    _, _, _, episodes = get_processing_episodes(
        video_paths=[video_path],
        video_paths_n_frames=[n_frames],
        frames_per_episode=frames_per_episode,
        tracking_intervals=tracking_intervals,
    )

    validate_episodes(episodes, n_frames, frame_indices, video_path)
    return episodes, n_frames, video_path, frame_indices


def validate_episodes(
    episodes: Sequence[Any],
    n_frames: int,
    frame_indices: Sequence[int] | None = None,
    video_path: Path | None = None,
) -> None:
    """Assert the episode list is a well-formed tiling of ``[0, n_frames)``.

    Real assertions, deliberately: a silent mis-tiling here would corrupt every
    downstream stage in a way that is very hard to trace back.
    """
    assert episodes, "no episodes produced"

    # Indices are 0..n-1 in order.
    assert [ep.index for ep in episodes] == list(range(len(episodes))), (
        "episode indices are not consecutive from 0"
    )

    # Single video => local and global coordinates coincide. Asserted, per spec,
    # rather than quietly assumed.
    for ep in episodes:
        assert ep.local_start == ep.global_start, (
            f"episode {ep.index}: local_start {ep.local_start} != "
            f"global_start {ep.global_start}"
        )
        assert ep.local_end == ep.global_end, (
            f"episode {ep.index}: local_end {ep.local_end} != "
            f"global_end {ep.global_end}"
        )
        assert ep.global_start < ep.global_end, (
            f"episode {ep.index} is empty or inverted: "
            f"[{ep.global_start}, {ep.global_end})"
        )
        assert ep.length == ep.global_end - ep.global_start
        # bbox_images belongs to the crop agent; must be untouched here.
        assert ep.bbox_images is None, (
            f"episode {ep.index}: bbox_images should be None at this stage"
        )
        if video_path is not None:
            assert Path(ep.video_path) == Path(video_path), (
                f"episode {ep.index}: video_path {ep.video_path} != {video_path}"
            )

    # Exact tiling of [0, n_frames): no gaps, no overlaps, full coverage.
    assert episodes[0].global_start == 0, (
        f"first episode starts at {episodes[0].global_start}, expected 0"
    )
    assert episodes[-1].global_end == n_frames, (
        f"last episode ends at {episodes[-1].global_end}, expected {n_frames}"
    )
    for prev, nxt in pairwise(episodes):
        assert prev.global_end == nxt.global_start, (
            f"gap/overlap between episode {prev.index} "
            f"(ends {prev.global_end}) and {nxt.index} "
            f"(starts {nxt.global_start})"
        )

    total = sum(ep.length for ep in episodes)
    assert total == n_frames, f"sum of episode lengths {total} != {n_frames}"

    # Every labeled frame lands in exactly one episode.
    if frame_indices is not None:
        for frame_idx in frame_indices:
            hits = [
                ep.index
                for ep in episodes
                if ep.global_start <= frame_idx < ep.global_end
            ]
            assert len(hits) == 1, (
                f"frame_idx {frame_idx} landed in {len(hits)} episodes: {hits}"
            )


# --------------------------------------------------------------------------
# CLI / smoke test
# --------------------------------------------------------------------------


def _main() -> None:
    slp_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SLP_PATH
    fpe = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FRAMES_PER_EPISODE

    print(f"Episode class    : {Episode.__module__}.{Episode.__qualname__}")
    print(f"Import route     : {EPISODE_IMPORT_ROUTE}")
    print(f"SLP file         : {slp_path}")

    episodes, n_frames, video_path, frame_indices = episodes_from_slp(slp_path, fpe)

    print(f"Video path       : {video_path}")
    print(f"Video exists     : {video_path.exists()} (irrelevant to this stage)")
    print(f"n_frames         : {n_frames}")
    print(f"labeled frames   : {len(frame_indices)} "
          f"({frame_indices[0]}..{frame_indices[-1]})")
    print(f"frames_per_ep    : {fpe}")
    print(f"n_episodes       : {len(episodes)}")
    print()

    header = f"{'idx':>4}  {'global_start':>12}  {'global_end':>10}  {'length':>6}"
    print(header)
    print("-" * len(header))
    for ep in episodes:
        print(
            f"{ep.index:>4}  {ep.global_start:>12}  "
            f"{ep.global_end:>10}  {ep.length:>6}"
        )
    print("-" * len(header))
    print(f"{'':>4}  {'':>12}  {'total':>10}  {sum(e.length for e in episodes):>6}")
    print()
    print("All assertions passed: episodes tile [0, "
          f"{n_frames}) with no gaps or overlaps, lengths sum to {n_frames}, "
          "and every frame_idx maps to exactly one episode.")


if __name__ == "__main__":
    _main()
