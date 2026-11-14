# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas>=3.0",
#     "pyarrow>=16",
#     "duckdb>=1.1",
#     "polars>=1.20",
#     "psutil",
# ]
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Week 2: Let a query engine do it

    Last week we saved memory by hand: choosing dtypes, reading only the columns
    we needed, and reading in chunks. This week we hand that work to two query
    engines, DuckDB and Polars, which do all three automatically. We try
    them on a year of New York City yellow taxi trips, about 41 million rows,
    which is roughly seven times the size of last week's ratings table.

    The session has three parts. First we answer one question four ways and
    compare the cost. Then we look at the query plans to see why the engines
    are cheaper. Finally we find a question where they are not cheaper, and
    work out how to rewrite it so that they are.
    """)
    return


@app.cell
def _():
    import inspect
    import json
    import subprocess
    import sys
    import textwrap
    from io import StringIO
    from pathlib import Path
    from urllib.request import urlretrieve

    import pandas as pd
    return (
        Path,
        StringIO,
        inspect,
        json,
        pd,
        subprocess,
        sys,
        textwrap,
        urlretrieve,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Get the data

    The Taxi and Limousine Commission publishes one Parquet file per month. We
    download the months you select (about 55 MiB each) and cache them next to
    the notebook. If you are on a small machine, start with a few
    months; every cell downstream reruns when you change the number.
    """)
    return


@app.cell
def _(mo):
    months = mo.ui.slider(1, 12, value=12, label="Months of 2024 to use", show_value=True)
    months
    return (months,)


@app.cell
def _(Path, months, urlretrieve):
    BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    DATA = Path("data_week2")
    DATA.mkdir(exist_ok=True)
    FILES = []
    for _m in range(1, months.value + 1):
        _name = f"yellow_tripdata_2024-{_m:02d}.parquet"
        if not (DATA / _name).exists():
            urlretrieve(BASE + _name, DATA / _name)
        FILES.append(str((DATA / _name).resolve()))
    _total = sum(Path(f).stat().st_size for f in FILES) / 2**20
    f"{len(FILES)} files, {_total:.0f} MiB on disk"
    return BASE, FILES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A fairer measuring tool

    Last week's exercise showed that peak memory measured inside a long-running
    notebook is unreliable, because memory freed by earlier cells gets reused.
    This week every measurement runs in a fresh Python process. `measure_fresh`
    sends the function's source code to a new interpreter, runs it there, and
    reports the time, the peak memory, and the result.

    The function has to be self-contained, which means it must import its own
    libraries. `FILES` (the list of Parquet paths) is defined for it.
    """)
    return


@app.cell
def _(FILES, StringIO, inspect, json, pd, subprocess, sys, textwrap):
    _FOOTER = """
    import json, threading, time, psutil
    _proc = psutil.Process()
    _start = _proc.memory_info().rss
    _peak = [_start]
    _done = threading.Event()
    def _sample():
        while not _done.is_set():
            _peak[0] = max(_peak[0], _proc.memory_info().rss)
            time.sleep(0.005)
    _t = threading.Thread(target=_sample, daemon=True)
    _t.start()
    _t0 = time.perf_counter()
    _result = FN()
    _elapsed = time.perf_counter() - _t0
    _done.set()
    _t.join()
    _peak[0] = max(_peak[0], _proc.memory_info().rss)
    if hasattr(_result, "to_pandas"):
        _result = _result.to_pandas()
    if hasattr(_result, "to_frame"):
        _result = _result.to_frame()
    print(json.dumps({
        "seconds": round(_elapsed, 2),
        "peak_extra_MiB": round((_peak[0] - _start) / 2**20, 1),
        "result": (
            _result if type(_result.index).__name__ == "RangeIndex" else _result.reset_index()
        ).to_json(orient="split", index=False),
    }))
    """

    def measure_fresh(label, fn, allow_failure=False):
        """Run fn in a new Python process; return (result DataFrame, stats dict).

        With allow_failure=True, a crash is reported in the stats instead of
        raising, so that one failure does not stop a comparison.
        """
        header = f"FILES = {FILES!r}\n"
        source = textwrap.dedent(inspect.getsource(fn))
        footer = textwrap.dedent(_FOOTER).replace("FN", fn.__name__)
        proc = subprocess.run(
            [sys.executable, "-c", header + source + footer],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            if not allow_failure:
                raise RuntimeError(proc.stderr[-3000:])
            _last = (proc.stderr.strip().splitlines() or ["no message"])[-1]
            return None, {"step": label, "seconds": None, "peak_extra_MiB": None,
                          "outcome": f"crashed: {_last[:80]}"}
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        result = pd.read_json(StringIO(out.pop("result")), orient="split")
        return result, {"step": label, **out, "outcome": "ok"}
    return (measure_fresh,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. One question, four ways

    *For trips paid by credit card, what is the average tip as a fraction of the
    fare, by hour of pickup?*

    Each version below answers that question. The first is what most people
    write first; the second applies last week's lesson of reading only the
    columns you need; the third and fourth hand the whole thing to an engine.
    Before running them, predict the order from cheapest to most expensive.
    """)
    return


@app.cell
def _():
    def tips_pandas_all_columns():
        import pandas as pd
        df = pd.concat([pd.read_parquet(f) for f in FILES])
        df = df[(df.payment_type == 1) & (df.fare_amount > 0)]
        hour = df.tpep_pickup_datetime.dt.hour.rename("hour")
        return (df.tip_amount / df.fare_amount).groupby(hour).mean().rename("tip_frac")

    def tips_pandas_four_columns():
        import pandas as pd
        cols = ["tpep_pickup_datetime", "payment_type", "fare_amount", "tip_amount"]
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in FILES])
        df = df[(df.payment_type == 1) & (df.fare_amount > 0)]
        hour = df.tpep_pickup_datetime.dt.hour.rename("hour")
        return (df.tip_amount / df.fare_amount).groupby(hour).mean().rename("tip_frac")

    def tips_duckdb():
        import duckdb
        return duckdb.sql(f"""
            SELECT hour(tpep_pickup_datetime) AS hour,
                   avg(tip_amount / fare_amount) AS tip_frac
            FROM read_parquet({FILES!r})
            WHERE payment_type = 1 AND fare_amount > 0
            GROUP BY hour ORDER BY hour
        """).df()

    def tips_polars():
        import polars as pl
        # Some copies of the monthly files store timestamps in different
        # units, so each file's pickup time is cast to one unit before the
        # files are combined.
        trips = pl.concat([
            pl.scan_parquet(f).with_columns(pl.col("tpep_pickup_datetime").cast(pl.Datetime("us")))
            for f in FILES
        ], how="diagonal_relaxed")
        return (
            trips
            .filter((pl.col("payment_type") == 1) & (pl.col("fare_amount") > 0))
            .group_by(pl.col("tpep_pickup_datetime").dt.hour().alias("hour"))
            .agg((pl.col("tip_amount") / pl.col("fare_amount")).mean().alias("tip_frac"))
            .sort("hour")
            .collect(engine="streaming")
        )
    return (
        tips_duckdb,
        tips_pandas_all_columns,
        tips_pandas_four_columns,
        tips_polars,
    )


@app.cell
def _(mo):
    run_tips = mo.ui.run_button(label="Run all four")
    run_tips
    return (run_tips,)


@app.cell
def _(
    measure_fresh,
    mo,
    pd,
    run_tips,
    tips_duckdb,
    tips_pandas_all_columns,
    tips_pandas_four_columns,
    tips_polars,
):
    mo.stop(not run_tips.value, mo.md("*Press the button to run the comparison.*"))
    _runs = [
        measure_fresh("pandas, all columns", tips_pandas_all_columns, allow_failure=True),
        measure_fresh("pandas, four columns", tips_pandas_four_columns, allow_failure=True),
        measure_fresh("DuckDB", tips_duckdb, allow_failure=True),
        measure_fresh("Polars, streaming", tips_polars, allow_failure=True),
    ]
    tips_stats = pd.DataFrame([s for _, s in _runs]).set_index("step")
    tips_result = _runs[2][0]
    mo.hstack([tips_stats, tips_result.round(3)], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Query plans and pushdown

    Both engines build a plan before touching the data, and the plan is
    something you can read. In the DuckDB plan, the `READ_PARQUET` box lists
    the only columns it will read under `Projections` (projection pushdown) and
    the conditions it applies while reading under `Filters` (filter pushdown).
    The Polars plan shows the same two things as `PROJECT 4/19 COLUMNS` and
    `SELECTION`.

    Parquet stores minimum and maximum values for each column in each block of
    rows (a *row group*). When a filter is pushed down to the scan, the reader
    can skip any row group whose range cannot match, without decompressing it.
    """)
    return


@app.cell
def _(FILES, mo):
    import duckdb
    import polars as pl

    _duck_plan = duckdb.sql(f"""
        EXPLAIN SELECT hour(tpep_pickup_datetime) AS hour,
               avg(tip_amount / fare_amount) AS tip_frac
        FROM read_parquet({FILES!r})
        WHERE payment_type = 1 AND fare_amount > 0
        GROUP BY hour
    """).fetchall()[0][1]

    _polars_plan = (
        pl.scan_parquet(FILES)
        .filter((pl.col("payment_type") == 1) & (pl.col("fare_amount") > 0))
        .group_by(pl.col("tpep_pickup_datetime").dt.hour().alias("hour"))
        .agg((pl.col("tip_amount") / pl.col("fare_amount")).mean())
        .explain()
    )
    mo.ui.tabs({
        "DuckDB plan": mo.plain_text(_duck_plan),
        "Polars plan": mo.plain_text(_polars_plan),
    })
    return duckdb, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Reading only what you need, over the network

    Projection pushdown matters even more when the file is not on your disk.
    DuckDB can query a Parquet file at a URL, and because the format records
    where each column starts, it fetches only the byte ranges for the columns
    the query uses. We ask a question of one month without downloading the
    whole file, and then download the whole file for comparison.
    """)
    return


@app.cell
def _(mo):
    run_remote = mo.ui.run_button(label="Query over the network")
    run_remote
    return (run_remote,)


@app.cell
def _(BASE, duckdb, mo, pd, run_remote):
    import time
    from urllib.request import urlopen

    mo.stop(not run_remote.value, mo.md("*Press the button to query the remote file.*"))
    _url = BASE + "yellow_tripdata_2024-03.parquet"
    _con = duckdb.connect()
    _con.sql("INSTALL httpfs; LOAD httpfs;")
    _t0 = time.perf_counter()
    _answer = _con.sql(
        f"SELECT count(*) AS trips, avg(trip_distance) AS mean_miles FROM '{_url}'"
    ).df()
    _remote = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    with urlopen(_url) as _r:
        _nbytes = len(_r.read())
    _download = time.perf_counter() - _t0
    mo.vstack([
        _answer,
        pd.DataFrame([
            {"method": "DuckDB query over HTTP", "seconds": round(_remote, 2)},
            {"method": f"download whole file ({_nbytes / 2**20:.0f} MiB)", "seconds": round(_download, 2)},
        ]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Medians by zone pair

    *Which ten pickup and drop-off zone pairs have the most trips, and what is
    the median fare for each?*

    Recall from last week that a mean can be computed in pieces, because sums
    and counts add up, while a median cannot. An exact median requires every
    fare in every group to be kept until all the data have been read, and with
    tens of thousands of zone pairs that takes a lot of memory.

    We run the obvious version in each tool, and also DuckDB's
    `approx_quantile`, which keeps a small summary (a *t-digest*) per group
    instead of every value. Predict whether the summary will reduce the memory
    used.
    """)
    return


@app.cell
def _():
    def median_pandas():
        import pandas as pd
        cols = ["PULocationID", "DOLocationID", "fare_amount"]
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in FILES])
        g = df.groupby(["PULocationID", "DOLocationID"])["fare_amount"]
        out = g.agg(n="size", median_fare="median")
        return out.sort_values("n", ascending=False).head(10)

    def median_duckdb():
        import duckdb
        return duckdb.sql(f"""
            SELECT PULocationID, DOLocationID, count(*) AS n,
                   median(fare_amount) AS median_fare
            FROM read_parquet({FILES!r})
            GROUP BY ALL ORDER BY n DESC LIMIT 10
        """).df()

    def median_duckdb_approx():
        import duckdb
        return duckdb.sql(f"""
            SELECT PULocationID, DOLocationID, count(*) AS n,
                   approx_quantile(fare_amount, 0.5) AS median_fare
            FROM read_parquet({FILES!r})
            GROUP BY ALL ORDER BY n DESC LIMIT 10
        """).df()

    def median_polars():
        import polars as pl
        trips = pl.concat([pl.scan_parquet(f) for f in FILES], how="diagonal_relaxed")
        return (
            trips
            .group_by("PULocationID", "DOLocationID")
            .agg(pl.len().alias("n"), pl.col("fare_amount").median().alias("median_fare"))
            .sort("n", descending=True)
            .head(10)
            .collect(engine="streaming")
        )
    return median_duckdb, median_duckdb_approx, median_pandas, median_polars


@app.cell
def _(mo):
    run_median = mo.ui.run_button(label="Run the median comparison")
    run_median
    return (run_median,)


@app.cell
def _(
    measure_fresh,
    median_duckdb,
    median_duckdb_approx,
    median_pandas,
    median_polars,
    mo,
    pd,
    run_median,
):
    mo.stop(not run_median.value, mo.md("*Press the button to run the comparison.*"))
    _runs = [
        measure_fresh("pandas, exact", median_pandas, allow_failure=True),
        measure_fresh("DuckDB, exact", median_duckdb, allow_failure=True),
        measure_fresh("DuckDB, approx_quantile", median_duckdb_approx, allow_failure=True),
        measure_fresh("Polars streaming, exact", median_polars, allow_failure=True),
    ]
    median_stats = pd.DataFrame([s for _, s in _runs]).set_index("step")
    median_ref = _runs[1][0]
    mo.hstack([median_stats, median_ref], justify="start", gap=2)
    return median_ref, median_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The approximate version is not cheaper, and on a machine with many CPUs it
    can run out of memory and crash. A t-digest is small, but there is
    one per zone pair per thread, and with tens of thousands of pairs the
    summaries add up to more than the raw fares would. An approximation saves
    memory only when the thing it replaces is larger than the approximation.

    DuckDB also has a `memory_limit` setting, and it is tempting to think that
    setting it low solves the problem. It limits DuckDB's buffer manager, which
    can spill sorts, joins and ordinary aggregates to disk; the per-group state
    of a median is held outside it, so the limit is either exceeded or the query
    fails with an out-of-memory error, depending on the number of threads.

    ## 7. Exercise

    Rewrite the query so that it needs far less memory, while still giving
    the **exact** median for each of the ten pairs. Return the same four
    columns as `median_duckdb`. The target is a peak of at most one tenth of
    the pandas version's.

    You need the medians for only ten pairs, but the obvious query computes
    them for every pair. Which part of the question is decomposable?
    """)
    return


@app.cell
def _():
    def my_median():
        # Replace this with your own version. Remember to import inside the
        # function; FILES is already defined.
        import duckdb
        return duckdb.sql(f"""
            SELECT PULocationID, DOLocationID, count(*) AS n,
                   median(fare_amount) AS median_fare
            FROM read_parquet({FILES!r})
            GROUP BY ALL ORDER BY n DESC LIMIT 10
        """).df()
    return (my_median,)


@app.cell
def _(measure_fresh, median_ref, median_stats, mo, my_median, pd):
    mine, my_stats = measure_fresh("exercise: my_median", my_median)
    _cols = ["PULocationID", "DOLocationID", "n", "median_fare"]
    _a = mine[_cols].reset_index(drop=True).astype(float)
    _b = median_ref[_cols].reset_index(drop=True).astype(float)
    _same = _a.shape == _b.shape and bool(((_a - _b).abs() < 1e-6).all().all())
    _target = median_stats.loc["pandas, exact", "peak_extra_MiB"] / 10
    _ok = _same and my_stats["peak_extra_MiB"] <= _target
    mo.vstack([
        pd.DataFrame([my_stats]).set_index("step"),
        mo.callout(
            mo.md(
                f"Same answer: **{_same}**. Peak target: **≤ {_target:.0f} MiB**. "
                f"Yours: **{my_stats['peak_extra_MiB']} MiB**."
            ),
            kind="success" if _ok else "warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Zone names

    The location IDs are hard to interpret on their own, so we join the
    Commission's lookup table to replace them with zone names, which makes it
    easier to judge whether the answer is plausible.
    """)
    return


@app.cell
def _(Path, duckdb, median_ref, mo, run_median, urlretrieve):
    mo.stop(not run_median.value, mo.md("*Run the median comparison first.*"))
    _zones = Path("data_week2/taxi_zone_lookup.csv")
    if not _zones.exists():
        urlretrieve("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv", _zones)
    _con = duckdb.connect()
    _con.register("median_ref", median_ref)
    _con.sql(f"""
        SELECT pz.Zone AS pickup, dz.Zone AS dropoff, m.n, m.median_fare
        FROM median_ref m
        JOIN '{_zones}' pz ON pz.LocationID = m.PULocationID
        JOIN '{_zones}' dz ON dz.LocationID = m.DOLocationID
        ORDER BY m.n DESC
    """).df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next week we will go the other way: instead of computing exact answers more
    cheaply, we will accept answers that are slightly wrong in exchange for a
    small, fixed amount of memory, whatever the size of the data.
    """)
    return


if __name__ == "__main__":
    app.run()
