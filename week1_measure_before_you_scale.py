# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas>=3.0",
#     "pyarrow>=16",
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
    # Week 1: Measure before you scale

    Before reaching for a cluster, find out where the time and memory go.
    Most "this dataset is too big" problems on a single machine turn out to be
    problems of representation: numbers stored in eight bytes when one would do,
    the same string stored six million times, or a file format that forces you to
    read every column to get one.

    In this session we load 6 million Goodreads ratings, make the table as large
    as a careless analysis would, and then shrink it step by step while measuring
    what each step saves. Every number below is measured on whatever machine is
    running this notebook, so your numbers will differ from your neighbour's.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The machine
    """)
    return


@app.cell
def _():
    import os
    import platform
    import threading
    import time
    from pathlib import Path
    from urllib.request import urlretrieve

    import pandas as pd
    import psutil
    return Path, os, pd, platform, psutil, threading, time, urlretrieve


@app.cell
def _(mo, os, platform, psutil):
    _vm = psutil.virtual_memory()
    mo.md(f"""
    | | |
    |---|---|
    | Platform | {platform.platform()} |
    | Logical CPUs | {os.cpu_count()} |
    | Total RAM | {_vm.total / 2**30:.1f} GiB |
    | Available RAM right now | {_vm.available / 2**30:.1f} GiB |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A measuring tool

    `measure` runs a function and reports two things: wall-clock time, and the
    peak growth in the process's resident memory while the function ran. Peak
    memory is sampled by a background thread every few milliseconds, so it
    catches the temporary copies that a function makes and throws away before
    returning. Those temporary copies are often what exhausts the memory of a
    small machine.

    Resident memory is an imprecise measure. Freed memory is not always returned to
    the operating system straight away, so treat small differences as noise and
    pay attention to differences of 2x or more.
    """)
    return


@app.cell
def _(psutil, threading, time):
    def measure(label, fn, *args, **kwargs):
        """Run fn(*args, **kwargs); return (result, stats dict)."""
        proc = psutil.Process()
        start_rss = proc.memory_info().rss
        peak = start_rss
        done = threading.Event()

        def sample():
            nonlocal peak
            while not done.is_set():
                peak = max(peak, proc.memory_info().rss)
                time.sleep(0.005)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            done.set()
            sampler.join()
        peak = max(peak, proc.memory_info().rss)
        stats = {
            "step": label,
            "seconds": round(elapsed, 2),
            "peak_extra_MiB": round((peak - start_rss) / 2**20, 1),
        }
        return result, stats

    def footprint_mib(df):
        """Memory held by a DataFrame, counting the contents of strings."""
        return round(df.memory_usage(deep=True).sum() / 2**20, 1)
    return footprint_mib, measure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Get the data

    The ratings file is the one used in the course (`goodbooks-10k`). It is
    downloaded once and cached next to this notebook. If the machine runs short
    of memory later on, lower the fraction of ratings used; everything downstream reruns
    automatically.
    """)
    return


@app.cell
def _(Path, urlretrieve):
    BASE = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
    DATA = Path("data_week1")
    DATA.mkdir(exist_ok=True)
    for _name in ["ratings.csv", "books.csv"]:
        if not (DATA / _name).exists():
            urlretrieve(BASE + _name, DATA / _name)
    RATINGS_CSV = DATA / "ratings.csv"
    BOOKS_CSV = DATA / "books.csv"
    {p.name: f"{p.stat().st_size / 2**20:.1f} MiB" for p in [RATINGS_CSV, BOOKS_CSV]}
    return BOOKS_CSV, DATA, RATINGS_CSV


@app.cell
def _(mo):
    fraction = mo.ui.slider(
        0.1, 1.0, step=0.1, value=1.0, label="Fraction of ratings to use", show_value=True
    )
    fraction
    return (fraction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The naive load

    `pd.read_csv` with no arguments infers each column's type, and for whole
    numbers it uses `int64`.
    """)
    return


@app.cell
def _(RATINGS_CSV, footprint_mib, fraction, measure, pd):
    ratings_naive, load_naive = measure("read_csv, default dtypes", pd.read_csv, RATINGS_CSV)
    ratings_naive = ratings_naive.sample(frac=fraction.value, random_state=0) if fraction.value < 1 else ratings_naive
    load_naive["footprint_MiB"] = footprint_mib(ratings_naive)
    load_naive["rows"] = len(ratings_naive)
    ratings_naive.dtypes.to_frame("dtype").assign(
        min=ratings_naive.min(), max=ratings_naive.max()
    )
    return load_naive, ratings_naive


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the `min` and `max` columns. A rating is between 1 and 5, and there
    are only 10,000 books, yet every value occupies eight bytes. The table below
    lists the smallest type that holds each column's range.

    | Type | Bytes | Range |
    |---|---|---|
    | `int8` / `uint8` | 1 | −128..127 / 0..255 |
    | `int16` / `uint16` | 2 | −32,768..32,767 / 0..65,535 |
    | `int32` / `uint32` | 4 | about ±2.1 billion / 0..4.3 billion |
    | `int64` | 8 | about ±9.2 × 10¹⁸ |

    Edit `TIGHT_DTYPES` below, choosing the smallest type that fits
    each column. If you choose one that is too small, pandas raises an error
    rather than wrapping silently, which you should check for yourself.
    """)
    return


@app.cell
def _():
    TIGHT_DTYPES = {"user_id": "int32", "book_id": "int32", "rating": "int32"}
    return (TIGHT_DTYPES,)


@app.cell
def _(RATINGS_CSV, TIGHT_DTYPES, footprint_mib, fraction, measure, pd):
    ratings, load_tight = measure(
        "read_csv, tight dtypes", pd.read_csv, RATINGS_CSV, dtype=TIGHT_DTYPES
    )
    ratings = ratings.sample(frac=fraction.value, random_state=0) if fraction.value < 1 else ratings
    load_tight["footprint_MiB"] = footprint_mib(ratings)
    load_tight["rows"] = len(ratings)
    load_tight
    return load_tight, ratings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The careless join

    Analyses rarely stay on integer IDs. Somebody wants titles and authors next to
    each rating, so the books table gets merged in. Each of the 10,000 titles is
    now repeated once for every rating it received. Since pandas 3.0, text
    columns default to the `str` dtype, which keeps the characters in one
    contiguous Arrow buffer rather than as separate Python objects. That is far
    more compact than the old default, but every repetition of a title is still
    stored in full.
    """)
    return


@app.cell
def _(BOOKS_CSV, pd):
    BOOK_COLS = ["book_id", "title", "authors", "language_code", "original_publication_year"]
    books = pd.read_csv(BOOKS_CSV, usecols=BOOK_COLS)
    books.head()
    return BOOK_COLS, books


@app.cell
def _(books, footprint_mib, measure, ratings):
    def _join():
        return ratings.merge(books, on="book_id", how="left")

    wide, join_stats = measure("merge books, str columns", _join)
    join_stats["footprint_MiB"] = footprint_mib(wide)
    join_stats["rows"] = len(wide)
    join_stats
    return join_stats, wide


@app.cell
def _(wide):
    (wide.memory_usage(deep=True) / 2**20).round(1).rename("MiB").to_frame()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The string columns still dominate. The `category` dtype stores each
    distinct value once and replaces every row with a small integer code, which
    suits columns with few distinct values relative to the number of rows. Every
    string column here qualifies, since there are only 10,000 books. It would
    not suit a column of mostly distinct values, such as review text, where
    `str` is already the better choice.

    Before running the next cell, predict how much smaller the table will get.
    """)
    return


@app.cell
def _(footprint_mib, join_stats, measure, pd, wide):
    _string_cols = ["title", "authors", "language_code"]

    def _as(dtype):
        out = wide.copy()
        for col in _string_cols:
            out[col] = out[col].astype(dtype)
        return out

    wide_cat, cat_stats = measure("strings as category", _as, "category")
    cat_stats["footprint_MiB"] = footprint_mib(wide_cat)
    cat_stats["rows"] = len(wide_cat)

    pd.DataFrame([join_stats, cat_stats])
    return cat_stats, wide_cat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. File formats: CSV against Parquet

    CSV is text, so a reader has to parse every character of every row and guess
    the types again each time. Parquet is a columnar binary format that stores
    the types, compresses each column separately, and lets a reader fetch only
    the columns it needs. We write the compact joined table both ways and then
    read it back.
    """)
    return


@app.cell
def _(DATA, measure, wide_cat):
    _csv = DATA / "wide.csv"
    _pq = DATA / "wide.parquet"
    _, write_csv = measure("write CSV", wide_cat.to_csv, _csv, index=False)
    _, write_pq = measure("write Parquet", wide_cat.to_parquet, _pq, index=False)
    write_csv["file_MiB"] = round(_csv.stat().st_size / 2**20, 1)
    write_pq["file_MiB"] = round(_pq.stat().st_size / 2**20, 1)
    WIDE_CSV, WIDE_PQ = _csv, _pq
    return WIDE_CSV, WIDE_PQ, write_csv, write_pq


@app.cell
def _(WIDE_CSV, WIDE_PQ, footprint_mib, measure, pd):
    _df, read_csv_all = measure("read CSV, all columns", pd.read_csv, WIDE_CSV)
    read_csv_all["footprint_MiB"] = footprint_mib(_df)
    _df, read_pq_all = measure("read Parquet, all columns", pd.read_parquet, WIDE_PQ)
    read_pq_all["footprint_MiB"] = footprint_mib(_df)
    _df, read_pq_two = measure(
        "read Parquet, two columns", pd.read_parquet, WIDE_PQ, columns=["book_id", "rating"]
    )
    read_pq_two["footprint_MiB"] = footprint_mib(_df)
    del _df
    pd.DataFrame([read_csv_all, read_pq_all, read_pq_two])
    return read_csv_all, read_pq_all, read_pq_two


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that the Parquet read restores the `category` columns, whereas the
    CSV read brings back plain `str` columns and the full footprint with them.
    Choosing a format also determines whether the next person inherits your
    dtype decisions.

    ## 7. Reading in chunks

    Some files are larger than memory however they are typed. If the computation
    can be expressed as a combination of partial results, the file can be read a
    piece at a time. The mean rating per book is a case in point: keep a running
    sum and count for each book, and divide at the end.

    Change the chunk size and watch both the time and the peak memory change.
    """)
    return


@app.cell
def _(mo):
    chunk_size = mo.ui.dropdown(
        options={"10,000": 10_000, "100,000": 100_000, "1,000,000": 1_000_000},
        value="100,000",
        label="Rows per chunk",
    )
    chunk_size
    return (chunk_size,)


@app.cell
def _(RATINGS_CSV, TIGHT_DTYPES, chunk_size, measure, pd, ratings):
    def _chunked_mean():
        sums = None
        counts = None
        for chunk in pd.read_csv(RATINGS_CSV, dtype=TIGHT_DTYPES, chunksize=chunk_size.value):
            g = chunk.groupby("book_id")["rating"]
            s, c = g.sum(), g.count()
            sums = s if sums is None else sums.add(s, fill_value=0)
            counts = c if counts is None else counts.add(c, fill_value=0)
        return sums / counts

    chunked, chunk_stats = measure(f"chunked mean, {chunk_size.value:,} rows/chunk", _chunked_mean)
    _full, full_stats = measure(
        "in-memory mean", lambda: ratings.groupby("book_id")["rating"].mean()
    )
    pd.DataFrame([chunk_stats, full_stats])
    return chunk_stats, chunked


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The in-memory version looks faster, but its timing leaves out the load that
    put `ratings` in memory in the first place. Compare the chunked figure with
    the load time from section 4 plus the in-memory figure.

    Not every computation decomposes this neatly. A mean does, because sums and
    counts can be added together; a median does not, because the medians of two
    halves do not determine the median of the whole. Deciding whether a
    computation can be split up and recombined is the central question in
    distributed computing as well, and MapReduce and Spark are built around it.

    ## 8. Exercise

    Consider this question: *which authors with at least 50,000 ratings have the
    highest mean rating?*

    The baseline below answers it the obvious way: read both files with default
    settings, merge, and group. Write `my_answer` so that it returns the same
    top 10 with a peak memory of at most **one tenth** of the baseline's. You may
    choose any dtypes and columns and you may read in chunks, but you must start
    from the files on disk, as the baseline does, and import anything beyond
    pandas inside the function.

    Both functions are measured in a fresh Python process. Inside this notebook,
    memory freed by earlier cells gets reused, which makes peak figures vary by
    hundreds of MiB from one run to the next; a new process starts clean.
    """)
    return


@app.cell
def _(BOOKS_CSV, RATINGS_CSV):
    import inspect
    import json
    import subprocess
    import sys
    import textwrap

    _CHILD_FOOTER = """
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
    print(json.dumps({
        "seconds": round(_elapsed, 2),
        "peak_extra_MiB": round((_peak[0] - _start) / 2**20, 1),
        "index": [str(i) for i in _result.index],
    }))
    """

    def measure_fresh(label, fn):
        """Run fn in a new Python process; return (top-10 index, stats dict)."""
        header = (
            "import pandas as pd\n"
            f"RATINGS_CSV = {str(RATINGS_CSV.resolve())!r}\n"
            f"BOOKS_CSV = {str(BOOKS_CSV.resolve())!r}\n"
        )
        source = textwrap.dedent(inspect.getsource(fn))
        footer = textwrap.dedent(_CHILD_FOOTER).replace("FN", fn.__name__)
        proc = subprocess.run(
            [sys.executable, "-c", header + source + footer],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        index = out.pop("index")
        return index, {"step": label, **out}
    return (measure_fresh,)


@app.cell
def _(measure_fresh, pd):
    def baseline():
        ratings = pd.read_csv(RATINGS_CSV)
        books = pd.read_csv(BOOKS_CSV)
        wide = ratings.merge(books, on="book_id")
        g = wide.groupby("authors")["rating"].agg(["mean", "count"])
        return g[g["count"] >= 50_000].sort_values("mean", ascending=False).head(10)

    ref_index, ref_stats = measure_fresh("exercise: baseline", baseline)
    pd.Series(ref_index, name="baseline top 10")
    return ref_index, ref_stats


@app.cell
def _():
    def my_answer():
        # Replace this with your own version. It must return a DataFrame
        # indexed by author, top 10 by mean rating. Inside this function,
        # pd, RATINGS_CSV and BOOKS_CSV are already defined.
        ratings = pd.read_csv(RATINGS_CSV)
        books = pd.read_csv(BOOKS_CSV)
        wide = ratings.merge(books, on="book_id")
        g = wide.groupby("authors")["rating"].agg(["mean", "count"])
        return g[g["count"] >= 50_000].sort_values("mean", ascending=False).head(10)
    return (my_answer,)


@app.cell
def _(measure_fresh, mo, my_answer, pd, ref_index, ref_stats):
    my_index, my_stats = measure_fresh("exercise: my_answer", my_answer)
    _same = my_index == ref_index
    _target = ref_stats["peak_extra_MiB"] / 10
    _ok = _same and my_stats["peak_extra_MiB"] <= _target
    mo.vstack([
        pd.DataFrame([ref_stats, my_stats]),
        mo.callout(
            mo.md(
                f"Same top 10: **{_same}**. Peak target: **≤ {_target:.0f} MiB**. "
                f"Yours: **{my_stats['peak_extra_MiB']} MiB**."
            ),
            kind="success" if _ok else "warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Everything we measured
    """)
    return


@app.cell
def _(
    cat_stats,
    chunk_stats,
    join_stats,
    load_naive,
    load_tight,
    pd,
    read_csv_all,
    read_pq_all,
    read_pq_two,
    write_csv,
    write_pq,
):
    pd.DataFrame([
        load_naive, load_tight, join_stats, cat_stats,
        write_csv, write_pq, read_csv_all, read_pq_all, read_pq_two, chunk_stats,
    ]).set_index("step")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next week we will hand most of this work to query engines (DuckDB and
    Polars), which choose compact types, read only the columns a query needs,
    and process data in chunks automatically, and we will try them on a table
    about seven times the size of this one.
    """)
    return


if __name__ == "__main__":
    app.run()
