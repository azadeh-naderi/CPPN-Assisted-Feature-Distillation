#!/bin/bash
# Copies every method_paper result except model checkpoints from results/
# (gitignored) into results_archive/method_paper/ (tracked), mirroring the
# directory layout, so the run records survive losing the cluster space --
# as happened once with /project/ahoover. Run from anywhere on the cluster:
#
#   bash method_paper/scripts/archive_results.sh
#   git add results_archive/method_paper && git commit -m "Archive method_paper results" && git push
#
# Archived: summary.json, training_log.csv, config_snapshot.json,
# extra_metrics.json, view patterns (.pt/.png), evolution and search logs,
# the fitness-study bank and analysis. Not archived: checkpoint.pt (model
# weights, ~300 MB in total) -- back those up separately if needed.
#
# Everything downstream reads the archive like the live results:
#   python3 method_paper/scripts/summarize.py --results-root results_archive/method_paper

set -e
cd "$(dirname "$0")/../.."

src=results/method_paper/
dst=results_archive/method_paper/
if [ ! -d "$src" ]; then
  echo "no $src here -- run this on the machine that has the results" >&2
  exit 1
fi
mkdir -p "$dst"
# tar pipe rather than rsync, which isn't installed everywhere; re-running
# overwrites archived files with their current versions.
(cd "$src" && tar cf - --exclude=checkpoint.pt .) | (cd "$dst" && tar xf -)

echo "archived: $(find "$dst" -type f | wc -l) files, $(du -sh "$dst" | cut -f1)"
big=$(find "$dst" -type f -size +50M)
if [ -n "$big" ]; then
  echo "WARNING: files over 50 MB (GitHub rejects files over 100 MB):" >&2
  echo "$big" >&2
fi
