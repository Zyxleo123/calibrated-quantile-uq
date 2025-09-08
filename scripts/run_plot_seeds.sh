#!/bin/bash

# Check if the user provided the -n argument
if [ -z "$1" ]; then
  echo "Usage: $0 <n_value>"
  exit 1
fi

NAME=$1
QUANTILES=(0.5 0.25 0.75)

for Q in "${QUANTILES[@]}"; do
    echo "Running plot_seed.py with -n $NAME -q $Q"
    python3 plots/plot_seed.py -n "$NAME" -q "$Q"
done
