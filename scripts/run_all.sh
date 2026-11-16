#!/usr/bin/env bash
# Run the whole pipeline for one trait, in order. Extra arguments go to every step.
#
#   bash scripts/run_all.sh configs/sycophantic_olmo2_1b.yaml
#   bash scripts/run_all.sh configs/sycophantic_olmo2_1b.yaml --smoke --force
#
# With --smoke it uses a tiny random model and a fake judge: a minute on CPU,
# meaningless numbers, and a check that every step runs.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bash scripts/run_all.sh <config.yaml> [--smoke] [--force]" >&2
  exit 1
fi
config="$1"
shift
extra=("$@")
step() { echo; echo "### $*"; uv run python "$@" --config "$config" ${extra[@]+"${extra[@]}"}; }

step scripts/generate.py                     # responses to the extraction and evaluation questions
step scripts/judge.py --input extract eval   # trait and coherence scores for them
step scripts/extract_vector.py               # the persona vector, one per layer
step scripts/reliability.py                  # does it depend on the particular questions?
step scripts/separation.py                   # reading test on the evaluation questions
step scripts/steer.py                        # writing test: generate with the vector added
step scripts/judge.py --input steering       # scores for the steered responses
step scripts/steering_summary.py             # does the trait rise with steering strength?
step scripts/attribute.py                    # which neurons write the vector
