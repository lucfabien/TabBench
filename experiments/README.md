# Running the Benchmark

## Reproducibility and Caching

To ensure reproducibility across different machines and environments, all workflows in Neuralk's Foundry use controlled random seeds, including at the step level. However, complete determinism may still depend on system architecture and library versions.

To mitigate this, the benchmark includes a caching mechanism that stores pre-computed splits in the `cache/` directory. These cached splits are used to avoid re-generating random train-test partitions, ensuring consistent results across runs. If you intend to re-run the benchmark or share results, we strongly recommend preserving this `cache/` directory.

## Running the Full Benchmark

To execute the full benchmark suite:

```bash
python run_bench.py
```

This script re-runs all registered models on all benchmark datasets. Ensure that both Neuralk Foundry and TabBench are properly installed before running.

## Evaluating Existing Models on Your Data

To apply the benchmarked models to your own dataset, you can use the following template:

```bash
python run_on_new_dataset.py
```

This script serves as a starting point for evaluating the pre-defined models on custom data. You may need to adapt data loading and preprocessing logic to match your input format.