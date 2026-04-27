# CLAUDE.md

Notes for Claude Code agents working in this repository.

## Environment quirks

### `uv sync` fails with "Protocol error (os error 71)" on the default venv path

The repo lives on a VirtualBox shared folder (`vboxsf`), which does not
support the symlinks that `uv` creates inside `.venv/bin/`. A bare
`uv sync` aborts with:

```
error: failed to symlink file from .venv/bin/python to ... : Protocol error (os error 71)
```

Workaround: put the venv on the ext4 root filesystem and tell uv to
copy package files instead of hardlinking them.

```bash
export UV_PROJECT_ENVIRONMENT=/home/vagrant/.venv-maimonedes
export UV_LINK_MODE=copy
uv sync
uv run pytest
```

Both env vars are needed. `UV_PROJECT_ENVIRONMENT` moves the venv
itself off the shared folder; `UV_LINK_MODE=copy` avoids hardlink
attempts when uv installs site-packages.

Set them once per shell (or export in your shell profile) and every
subsequent `uv run …` / `uv sync` will use the same venv.
