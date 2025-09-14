#!/bin/bash

# Check if the user provided the -n argument.
# Changed the check to look for the second argument, which will be the name.
if [ -z "$2" ] || ([ "$1" != "-n" ] && [ "$1" != "--exp_name" ]); then
  echo "Usage: $0 -n <exp_name>"
  exit 1
fi

NAME=$2
QUANTILES=(0.5 0.25 0.75)

for Q in "${QUANTILES[@]}"; do
    echo "Running plot_seed.py with -n $NAME -q $Q -r"
    python plots/plot_seeds.py -n "$NAME" -q "$Q" -r

    echo "Running plot_seed.py with -n $NAME -q $Q"
    python plots/plot_seeds.py -n "$NAME" -q "$Q"
done

echo "All plotting tasks complete."