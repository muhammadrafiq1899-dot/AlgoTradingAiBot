#!/bin/sh
# Check that PROJECT_MAP.md is still in sync with the codebase.
#
# POSIX sh on purpose: Termux has no /usr/bin/env, so an `env` shebang would
# break direct execution on Android. Run with:  sh scripts/check_project_map.sh
# (or `bash scripts/check_project_map.sh` — both work).
#
# Exit 1 (with hints) when the map is missing, incomplete, or stale, so the
# same script can be used as:
#   - a pre-commit hook  (install once via:  bash scripts/install_hooks.sh)
#   - a CI step          (bash scripts/check_project_map.sh)
#
# Checks, in order:
#   1. The map exists and has the expected structure (anchor lines).
#   2. EVERY source file in the git index under algotrading/, scripts/,
#      config/ and tests/ is mentioned in the map by name. A new module that
#      isn't documented fails here no matter how fresh the map's mtime is.
#      (`__init__.py` package markers are exempt; the map documents packages
#      structurally, not per-file.)
#   3. No tracked source file is BOTH newer than the map AND actually differs
#      from HEAD — i.e. a real content change made after the map was last
#      touched. Restoring/checking out a file bumps its mtime but has no diff,
#      so it is not flagged.
set -u

PROJECT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$PROJECT_DIR"

MAP="PROJECT_MAP.md"

if [ ! -f "$MAP" ]; then
    echo "!! $MAP is missing — create it (see the Update rule at the top of the file)." >&2
    exit 1
fi

# --- 1. structure sanity ---------------------------------------------------

for anchor in "algotrading/" "scheduler" "Feature map" "Non-negotiable invariants"; do
    if ! grep -q "$anchor" "$MAP"; then
        echo "!! $MAP looks incomplete (missing anchor '$anchor')." >&2
        exit 1
    fi
done

# Files under check: the git index when this is a repo (index = exactly what a
# commit would contain, including newly staged files), else everything on disk.
GIT=0
if [ -d .git ] && command -v git >/dev/null 2>&1; then
    GIT=1
    FILES=$(git ls-files -- 'algotrading/*' 'scripts/*' 'config/*' 'tests/*')
else
    FILES=$(find algotrading scripts config tests -type f \
        ! -path '*/__pycache__/*' ! -name '*.pyc' 2>/dev/null)
fi

# --- 2. every source file is mentioned in the map --------------------------

missing=""
while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "$f" in
        */__init__.py) continue ;;
    esac
    base=$(basename "$f")
    if ! grep -qF -- "$base" "$MAP"; then
        missing="${missing}  - ${f}
"
    fi
done <<EOF
$FILES
EOF

if [ -n "$missing" ]; then
    echo "!! $MAP does not mention these source files (new modules must be documented):" >&2
    printf '%s' "$missing" >&2
    echo "   Add each file to the module reference or directory map, then re-commit." >&2
    exit 1
fi

# --- 3. no changed file is newer than the map ------------------------------

if command -v stat >/dev/null 2>&1; then
    MAP_MTIME=$(stat -c '%Y' "$MAP")
    stale=""
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        mtime=$(stat -c '%Y' "$f" 2>/dev/null) || continue
        [ "$mtime" -gt "$MAP_MTIME" ] || continue
        # Newer than the map: is it a real change, or just a restore/checkout
        # that bumped the mtime? Without git we assume any newer file is stale.
        if [ "$GIT" -eq 1 ] && git diff --quiet HEAD -- "$f" 2>/dev/null; then
            continue
        fi
        stale="${stale}  - ${f}
"
    done <<EOF
$FILES
EOF

    if [ -n "$stale" ]; then
        echo "!! $MAP is stale: these tracked files changed after the map was last updated:" >&2
        printf '%s' "$stale" >&2
        echo "   Update $MAP in this same change (see the Update rule at its top), then re-commit." >&2
        exit 1
    fi
fi

echo "OK: $MAP exists, mentions every source file, and is newer than all changed files."