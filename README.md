# Scaling on one machine

These are marimo notebooks for four office-hour sessions on processing data that is large relative to the machine, using one computer rather than a cluster.

| Week | Notebook | Topic |
|---|---|---|
| 1 | `week1_measure_before_you_scale.py` | Measuring time and memory; dtypes, `category`, Parquet and chunked reading |
| 2 | `week2_query_engines.py` | DuckDB and Polars on a year of NYC taxi trips; query plans and pushdown |
| 3 | `week3_approximate_answers.py` | Reservoir sampling, HyperLogLog and Count-Min sketches on Gutenberg books |
| 4 | `week4_parallelism_and_its_limits.py` | Worker processes, Amdahl's law, the COST argument, and when a cluster is needed |

Each notebook declares its dependencies inline and downloads its data when it first runs, so no data is stored in this repository. To run one locally:

```
uvx marimo edit --sandbox week1_measure_before_you_scale.py
```

The notebooks also run on [molab](https://molab.marimo.io) when imported from this repository.
