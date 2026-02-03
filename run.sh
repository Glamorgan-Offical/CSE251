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
    for pca in 10 20 50 100 150
    do
        echo "Cluster: $n prototypes/class, PCA=$pca"
        python main.py --selector cluster --num_prototypes $n --pca_components $pca
    done
done

echo "Completed!"

# Summary
echo ""
echo "Results summary:"
ls -lh results/ | wc -l