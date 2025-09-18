#!/bin/bash

# Check if the user provided the -n argument.
# Changed the check to look for the second argument, which will be the name.
if [ -z "$2" ] || ([ "$1" != "-n" ] && [ "$1" != "--exp_name" ]); then
  echo "Usage: $0 -n <exp_name>"
  exit 1
fi

NAME=$2
WINDOW_SIZES=(0.05 0.1 0.2 0.3)

for W in "${WINDOW_SIZES[@]}"; do
    echo "Running plot_lump.py with -n $NAME -w $W"
    python plots/plot_lump.py -n "$NAME" -w "$W"
done

echo "All plotting tasks complete."