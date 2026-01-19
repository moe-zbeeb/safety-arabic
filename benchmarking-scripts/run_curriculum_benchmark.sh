#!/bin/bash
# Run curriculum learning benchmarks

cd /home/zbibm/Safety-Arabic

echo "Starting curriculum learning benchmark..."
echo "This will benchmark all 5 models x 6 stages = 30 configurations"
echo ""

python benchmarking-scripts/benchmark_curriculum.py

echo ""
echo "Done!"
