# idtracker.ai — vendored with history absorbed

This tree was cloned from upstream and its own `.git` was **removed** so that our fork is tracked
directly in this project's history. That is deliberate.

| | |
|---|---|
| Upstream | https://gitlab.com/polavieja_lab/idtrackerai.git |
| Branch | `master` |
| Commit | `03c7aeda9be627b9691a9b29bb8acd0065a1d038` |
| `git describe` | `6.0.14-1-g03c7aeda` |
| Absorbed on | 2026-07-21 |
| State at absorption | clean — no local commits, no stashes, no modified files |

## Why absorbed rather than a submodule

We **modify** this code. The entire value of the port is legible as a diff against pristine
upstream, and that only works if our edits live in *our* history. A submodule would push those edits
into idtracker.ai's history instead, requiring a second repo to manage forever.

The cost is that upstream's own commit history is not available here and future upstream updates
must be merged by hand. For a fork frozen at one version, neither cost is real.

## Getting the exact surgical-change report

The **first commit of this repository is pristine upstream**, with no edits of ours mixed in. So:

```bash
git diff <first-commit> -- idtrackerai/       # every change we have made, exactly
git log --oneline -- idtrackerai/             # when and why each was made
```

Find the baseline with `git log --oneline | tail -1`.

## Editing policy

Never delete upstream code. Comment it out in place with a `# SLEAP-PORT:` marker and an
explanation of why it is not needed. The commented-out block is documentation for the next reader —
it shows what idtracker.ai does that we deliberately skip.

## Surgical edits made so far

Run `git diff 189a239 -- idtrackerai/` for the authoritative list.

### 1. `base/network/models.py:68` — `IdCNN` generalised to non-square input

```python
# was
nn.Linear(100 * (input_shape[1] // 4) ** 2, 100)
# now
nn.Linear(100 * (input_shape[0] // 4) * (input_shape[1] // 4), 100)
```

**Why.** The square assumed height == width. That held upstream only because
`Session.set_id_image_size` auto-derives `[max_size, max_size, 1]` (`session.py:543`) from a single
median body length. This port sizes id-images from the median x-spread and median y-extent
*separately*, which for flies gives 57 × 83, so the flatten must be `H//4 * W//4`.

**Risk: none for square models.** The two expressions are identical whenever
`input_shape[0] == input_shape[1]` — verified at 80×80 (40000), 52×52 (16900), 100×100 (62500).
Pretrained square weights therefore still load unchanged.

**Why it is needed at all**, given that the contrastive identity model is `ResNet18` (which
adaptive-pools and never cared about shape): `IdCNN` is reached by the supervised accumulation
cascade (`tracker.py:176/192`), which `tracker_API` runs whenever contrastive accumulates less than
`conf.CONTRASTIVE_MIN_ACCUMULATION` (0.5). Before Global Fragments existed that branch was
unreachable and this was a latent bug; building them (Stage 3f) made it live.

`Session.set_id_image_size` was **not** edited. It only forces square on the auto path — line 539
is `if not self.id_image_size:` — so an explicitly supplied `[83, 57, 1]` is respected.

## Installation

Installed into the `sleap_id` conda env as an editable package so our edits take effect immediately
and imports are clean:

```bash
pip install -e ./idtrackerai --no-deps
```

`--no-deps` is intentional: we need the import path and package metadata, not idtracker.ai's full
(heavy) dependency tree. Three small transitive imports are pulled in by
`idtrackerai/__init__` → `logging_utils` → `telemetry` and must be present: `toml`, `deprecated`,
`requests`.
