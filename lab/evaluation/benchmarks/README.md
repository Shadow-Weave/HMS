# Evaluation adapters

This directory contains source code for benchmark ingestion, recall, answer
generation, and judging. The repository does not include datasets, retained
memory banks, generated answers, logs, or result artifacts.

The public benchmark adapter currently supports LongMemEval:

- [LongMemEval reproduction guide](longmemeval/README.md)
- Entry point: `bash .aaaSCRIPT/run_benchmark.sh`
- Python module: `python -m benchmarks.longmemeval.longmemeval_benchmark`

The launcher performs the complete `Retain -> Recall -> Answer -> Judge`
pipeline by default. Set `HMS_RETRIEVAL_ONLY=1` only when the configured
database already contains every required memory bank.
