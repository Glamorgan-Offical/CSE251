#!/bin/bash

echo "Running ALL experiments..."

# Baseline
echo "Baseline "
python main.py --selector baseline

# Random Selection
echo "Random Selection"

for n in 10 50 100 200 500 1000
do
    echo "Random: $n prototypes/class"
    python main.py --selector random --num_prototypes $n
done

# Cluster Selection
echo "Cluster Selection "
for n in 10 50 100 200 500 1000
do
    echo "Cluster: $n prototypes/class"
    python main.py --selector cluster --num_prototypes $n
done

echo "Completed!"

# Summary
echo ""
echo "Results summary:"
ls -lh results/