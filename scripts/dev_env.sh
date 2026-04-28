# Source this file to set the env vars uv needs on this machine.
#
# The repo lives on a VirtualBox shared folder (`vboxsf`), which doesn't
# support the symlinks `uv` creates inside `.venv/bin/`. Putting the
# venv on the ext4 root filesystem and forcing copy-mode installs is
# the standard workaround. See CLAUDE.md.
#
# Usage:
#   source scripts/dev_env.sh
#   uv run alembic upgrade head
#   uv run pytest
#
# Re-running is harmless; the exports are idempotent.

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/home/vagrant/.venv-maimonedes}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo "uv: UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "uv: UV_LINK_MODE=${UV_LINK_MODE}"
