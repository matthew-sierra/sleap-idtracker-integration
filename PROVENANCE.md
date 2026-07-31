# Provenance of the third-party code this port targets

Two different relationships with upstream code, handled two different ways.

## Bundled, because we modify it — `idtrackerai/`

idtracker.ai **6.0.14**, from https://gitlab.com/polavieja_lab/idtrackerai, lives inside this
repository at `idtrackerai/`. It is not a submodule and not a clean clone: it carries surgical
edits marked `# SLEAP-PORT:` — the non-square input change in
`src/idtrackerai/base/network/models.py` (Stage 3g) being the substantive one. Because those edits
are ours, a fresh upstream clone will **not** reproduce this directory, so it has to travel with
the port.

Editing policy is unchanged: surgical edits at injection points only, every removal commented out
in place with a `# SLEAP-PORT:` marker explaining why, never deleted. Full provenance in
`idtrackerai/PROVENANCE.md`.

`config.py` puts `idtrackerai/src` on `sys.path` at import time, so a fresh clone runs without a
separate install. If you prefer a real install, `pip install -e idtrackerai/` also works and takes
precedence — `config` only inserts the path when it is not already there.

## Not bundled, because we never touch it — the reference forks

Three upstream trees were kept purely to read the port against the exact source it targets. Nothing
in this repository imports them, and they total ~428 MB, so they are **not** tracked here. Recreate
them whenever you want to read along:

| Package | Version | Upstream |
|---|---|---|
| `sleap` | v1.6.3 | https://github.com/talmolab/sleap |
| `sleap-nn` | v0.3.0 | https://github.com/talmolab/sleap-nn |
| `sleap-io` | v0.7.1 | https://github.com/talmolab/sleap-io |

```bash
mkdir -p vendor && cd vendor
git clone --depth 1 --branch v1.6.3 https://github.com/talmolab/sleap.git    sleap
git clone --depth 1 --branch v0.3.0 https://github.com/talmolab/sleap-nn.git sleap-nn
git clone --depth 1 --branch v0.7.1 https://github.com/talmolab/sleap-io.git sleap-io
```

`vendor/` is in `.gitignore`. Never edit anything inside it; it exists to be read.

### Why these versions

- **sleap v1.6.3** and **sleap-nn v0.3.0** were specified directly.
- **sleap-io v0.7.1** is the default io package shipped with SLEAP: `sleap==1.6.3` pins
  `sleap-io[all]>=0.7.0,<0.8.0` (`vendor/sleap/pyproject.toml:55`), and 0.7.1 is the newest release
  in that range. This is the one reference package that is also a runtime dependency — it is in
  `pyproject.toml`, installed from PyPI.

Note that `sleap` 1.6.3 itself ships **sleap-nn 0.2.0**; v0.3.0 is a newer standalone release and
was requested deliberately. Be explicit about which one you mean when reading sleap-nn code.
