#!/usr/bin/env bash
#
# Wipe perturbation + compliance score data so the next `maimonedes
# perturb` sweep starts from a clean slate. Anchor baselines
# (probe_role='anchor') are also removed — re-run `maimonedes run-once`
# afterwards if you want fresh baselines too.
#
# What gets deleted:
#   - compliance_scores rows where probe_role='perturbation' (the cloud)
#   - perturbation_probes (every probe row)
#   - compliance_scores rows where probe_role='anchor' (the baselines)
#
# What stays:
#   - llm_calls (audit trail of every LLM round-trip)
#   - schema_version + alembic_version
#
# Usage:
#   scripts/reset_perturbations.sh             # asks for confirmation
#   scripts/reset_perturbations.sh --yes       # skip the prompt
#
# Override the DB path:
#   MAIMONEDES_DB=/path/to/file.db scripts/reset_perturbations.sh

set -euo pipefail

DB="${MAIMONEDES_DB:-/home/vagrant/maimonedes.db}"

if [ ! -f "$DB" ]; then
    echo "no database at $DB" >&2
    echo "set MAIMONEDES_DB to the right path or check your .env" >&2
    exit 1
fi

if [ "${1:-}" != "--yes" ]; then
    echo "About to wipe ALL perturbation + score data from:"
    echo "  $DB"
    read -rp "Type 'yes' to confirm: " confirm
    [ "$confirm" = "yes" ] || { echo "aborted"; exit 1; }
fi

sqlite3 "$DB" <<'EOF'
DELETE FROM compliance_scores WHERE probe_role='perturbation';
DELETE FROM perturbation_probes;
DELETE FROM compliance_scores WHERE probe_role='anchor';
EOF

echo "wiped perturbation_probes + compliance_scores in $DB"
