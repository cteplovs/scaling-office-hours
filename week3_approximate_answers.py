# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas>=3.0",
#     "pyarrow>=16",
#     "numpy>=2.0",
#     "altair>=5",
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
    # Week 3: Approximate answers in fixed memory

    In the first two weeks we computed exact answers more cheaply. This week we
    accept answers that are slightly wrong, in exchange for a summary of the
    data (a *sketch*) whose size is fixed in advance, however much data passes
    through it.

    Last week's `approx_quantile` used more memory than the exact median,
    because DuckDB kept one summary per zone pair per thread, and tens of
    thousands of small summaries added up to more than the fares themselves.
    This week we use a single summary for the whole stream, which is the
    situation in which sketches save memory.

    The data are books from Project Gutenberg, like the texts in Homework 1,
    but they contain about four times as many words as the largest of those
    texts. We split words the way Homework 1 does,
    with the regular expression `[\w']+`, and we count three-word phrases
    ("one of the") rather than single words, because there are millions of
    distinct phrases but only about a hundred thousand distinct words.
    """)
    return


@app.cell
def _():
    import inspect
    import json
    import subprocess
    import sys
    import textwrap
    from collections import Counter
    from io import StringIO
    from pathlib import Path
    from urllib.request import urlretrieve

    import altair as alt
    import numpy as np
    import pandas as pd
    return (
        Counter,
        Path,
        StringIO,
        alt,
        inspect,
        json,
        np,
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

    We use the test split of PG-19, a collection of Gutenberg books published
    by DeepMind, which contains 100 books (about 7 million words) in one
    25 MiB Parquet file.
    The selected books are written to a plain text file, which every method
    below reads line by line, as an MRJob mapper would. If you are on a small
    machine, start with fewer books.
    """)
    return


@app.cell
def _(mo):
    books = mo.ui.slider(10, 100, step=10, value=100, label="Books to use", show_value=True)
    books
    return (books,)


@app.cell
def _(Path, books, mo, pd, urlretrieve):
    PG19_URL = (
        "https://huggingface.co/datasets/emozilla/pg19/resolve/main/data/"
        "test-00000-of-00001-29a571947c0b5ccc.parquet"
    )
    DATA = Path("data_week3")
    DATA.mkdir(exist_ok=True)
    _parquet = DATA / "pg19_test.parquet"
    if not _parquet.exists():
        urlretrieve(PG19_URL, _parquet)
    _books = pd.read_parquet(_parquet, columns=["short_book_title", "text"]).head(books.value)
    _path = DATA / f"corpus_{books.value}.txt"
    if not _path.exists():
        _path.write_text("\n".join(_books.text), encoding="utf-8")
    CORPUS = str(_path.resolve())
    mo.vstack([
        mo.md(f"{books.value} books, {_path.stat().st_size / 2**20:.0f} MiB of text. The first five are:"),
        _books[["short_book_title"]].head(5),
    ])
    return (CORPUS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Phrases, hashes and a measuring tool

    `phrase_batches` reads the text a line at a time and yields the three-word
    phrases in lists of about `size` phrases, so that no method has to hold
    the whole text. `hash_phrases` turns each phrase into a 64-bit integer.
    Every sketch below works on these hashes rather than on the phrases, and
    because a good hash spreads phrases evenly over all 2<sup>64</sup> values,
    its bits behave like random bits.
    """)
    return


@app.cell
def _():
    def phrase_batches(path, size=200_000):
        """Yield lists of three-word phrases from a text file, about `size` at a time."""
        import re
        word_re = re.compile(r"[\w']+")
        batch, prev = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                words = prev + word_re.findall(line.lower())
                batch.extend([" ".join(words[i:i + 3]) for i in range(len(words) - 2)])
                prev = words[-2:]  # a phrase can continue onto the next line
                if len(batch) >= size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def hash_phrases(phrases):
        """Return one 64-bit hash (a uint64 array) per phrase."""
        return pd.util.hash_array(np.asarray(phrases, dtype=object))
    return hash_phrases, phrase_batches


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As last week, `measure_fresh` runs a function in a new Python process and
    reports its time and peak memory. The new process has `np`, `pd`,
    `CORPUS`, `phrase_batches` and `hash_phrases` defined, along with any
    other helper functions passed after the function to be measured.
    """)
    return


@app.cell
def _(
    CORPUS,
    StringIO,
    hash_phrases,
    inspect,
    json,
    pd,
    phrase_batches,
    subprocess,
    sys,
    textwrap,
):
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
    if not isinstance(_result, pd.DataFrame):
        _result = pd.DataFrame({"value": [_result]})
    print(json.dumps({
        "seconds": round(_elapsed, 2),
        "peak_extra_MiB": round((_peak[0] - _start) / 2**20, 1),
        "result": _result.reset_index(drop=True).to_json(orient="split", index=False),
    }))
    """

    def measure_fresh(label, fn, *helpers):
        """Run fn in a new Python process; return (result DataFrame, stats dict)."""
        parts = ["import numpy as np", "import pandas as pd", f"CORPUS = {CORPUS!r}"]
        for _f in (phrase_batches, hash_phrases, *helpers, fn):
            parts.append(textwrap.dedent(inspect.getsource(_f)))
        parts.append(textwrap.dedent(_FOOTER).replace("FN", fn.__name__))
        proc = subprocess.run(
            [sys.executable, "-c", "\n".join(parts)], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        result = pd.read_json(StringIO(out.pop("result")), orient="split")
        return result, {"step": label, **out}
    return (measure_fresh,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Exact answers

    We ask two questions: how many distinct phrases there are, and which ten
    phrases are the most common. The exact answers need a `set` and a
    `Counter`, each with one entry per distinct phrase. The first row of the
    table reads every phrase and discards it, which shows how much of the
    memory goes to reading rather than to counting.

    After running it, move the slider in section 1 and run it again. Before you
    do, predict how the peak memory will change with the number of books.
    """)
    return


@app.cell
def _():
    def read_only():
        n = 0
        for batch in phrase_batches(CORPUS):
            n += len(batch)
        return n

    def exact_distinct():
        seen = set()
        for batch in phrase_batches(CORPUS):
            seen.update(batch)
        return len(seen)

    def exact_top10():
        from collections import Counter
        counts = Counter()
        for batch in phrase_batches(CORPUS):
            counts.update(batch)
        return pd.DataFrame(counts.most_common(10), columns=["phrase", "count"])
    return exact_distinct, exact_top10, read_only


@app.cell
def _(mo):
    run_exact = mo.ui.run_button(label="Run the exact versions")
    run_exact
    return (run_exact,)


@app.cell
def _(
    exact_distinct,
    exact_top10,
    measure_fresh,
    mo,
    pd,
    read_only,
    run_exact,
):
    mo.stop(not run_exact.value, mo.md("*Press the button to run the exact versions.*"))
    _n, _s0 = measure_fresh("read and discard", read_only)
    _d, _s1 = measure_fresh("distinct phrases (set)", exact_distinct)
    exact_top, _s2 = measure_fresh("top 10 phrases (Counter)", exact_top10)
    _s0["answer"] = f"{_n.value[0]:,} phrases"
    _s1["answer"] = f"{_d.value[0]:,} distinct"
    _s2["answer"] = "table on the right"
    exact_stats = pd.DataFrame([_s0, _s1, _s2]).set_index("step")
    mo.hstack([exact_stats, exact_top], justify="start", gap=2)
    return exact_stats, exact_top


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Reservoir sampling

    The simplest way to bound memory is to keep a random sample. If the length
    of the stream is not known in advance, a *reservoir* of `k` items does it in
    one pass: keep the first `k` items, and then replace a random member of the
    reservoir with item `i` with probability `k / i`. At every point in the
    stream, each item read so far has the same chance of being in the reservoir.

    We sample lines of text and use the sample to estimate how often some words
    occur. Compare the estimates for common and rare words, and draw a few
    samples of each size.
    """)
    return


@app.cell
def _():
    def reservoir_sample(items, k, seed=0):
        """Return a uniform random sample of k items from an iterable, in one pass."""
        import random
        rng = random.Random(seed)
        sample = []
        for i, item in enumerate(items):
            if i < k:
                sample.append(item)
            else:
                j = rng.randint(0, i)
                if j < k:
                    sample[j] = item
        return sample
    return (reservoir_sample,)


@app.cell
def _(CORPUS, Counter):
    import re

    WORD_RE = re.compile(r"[\w']+")
    word_counts = Counter()
    with open(CORPUS, encoding="utf-8") as _f:
        for _line in _f:
            word_counts.update(WORD_RE.findall(_line.lower()))
    return WORD_RE, word_counts


@app.cell
def _(mo):
    sample_size = mo.ui.dropdown(
        options={"100 lines": 100, "1,000 lines": 1_000, "10,000 lines": 10_000, "100,000 lines": 100_000},
        value="1,000 lines",
        label="Reservoir size",
    )
    redraw = mo.ui.button(value=0, on_click=lambda v: v + 1, label="Draw another sample")
    mo.hstack([sample_size, redraw], justify="start")
    return redraw, sample_size


@app.cell
def _(
    CORPUS,
    Counter,
    WORD_RE,
    pd,
    redraw,
    reservoir_sample,
    sample_size,
    word_counts,
):
    with open(CORPUS, encoding="utf-8") as _f:
        _sample = reservoir_sample(_f, sample_size.value, seed=redraw.value)
    _sample_counts = Counter(w for _line in _sample for w in WORD_RE.findall(_line.lower()))
    _total = sum(word_counts.values())
    _sample_total = max(sum(_sample_counts.values()), 1)
    pd.DataFrame([
        {
            "word": _w,
            "per 10,000 words, all text": round(1e4 * word_counts[_w] / _total, 2),
            "per 10,000 words, sample": round(1e4 * _sample_counts[_w] / _sample_total, 2),
        }
        for _w in ["the", "and", "she", "money", "stomach", "whale"]
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A sample estimates the rate of a common word well, but a word that occurs
    once in ten thousand words appears only a few times in a small sample, if at
    all, and its estimate varies a great deal from one sample to the next. The
    relative error of a sampled count is roughly one over the square root of the
    number of times the item appears in the sample.

    Nor can a sample be used to estimate how many distinct phrases there are,
    because most phrases occur once, and a sample of one per cent of the text contains
    about one per cent of them with no indication of how many were missed. The
    next two sketches address those two problems.

    ## 5. HyperLogLog: counting distinct items

    Hash every phrase and look at the hash's bits. Half of all hashes start with
    a 1, a quarter with 01, an eighth with 001, and so on, so if the longest run
    of leading zeros seen so far is 20, about 2<sup>20</sup> distinct values have
    probably been hashed. The same phrase always has the same hash, so repeats
    do not change the answer.

    One run of zeros gives a very noisy estimate, so HyperLogLog splits the
    stream into `m` = 2<sup>p</sup> groups using the first `p` bits of each
    hash, and stores in one byte per group (a *register*) the largest position
    of the first 1 bit in the rest of the hash. The estimate combines the
    registers with a harmonic mean, and its standard error is about
    1.04 / √`m`. With `p` = 14, the registers take 16 KiB. DuckDB's
    `approx_count_distinct` is an implementation of the same algorithm.
    """)
    return


@app.cell
def _(np):
    def hll_update(registers, hashes):
        """Update HyperLogLog registers in place with a batch of 64-bit hashes."""
        p = int(np.log2(len(registers)))
        index = (hashes >> np.uint64(64 - p)).astype(np.int64)
        rest = ((hashes << np.uint64(p)) >> np.uint64(32)).astype(np.float64)  # the next 32 bits
        rank = np.where(rest > 0, 32 - np.floor(np.log2(np.maximum(rest, 1))), 33)
        np.maximum.at(registers, index, rank.astype(np.uint8))

    def hll_estimate(registers):
        """Estimate the number of distinct hashes from HyperLogLog registers."""
        m = len(registers)
        alpha = {16: 0.673, 32: 0.697, 64: 0.709}.get(m, 0.7213 / (1 + 1.079 / m))
        estimate = alpha * m * m / np.sum(2.0 ** -registers.astype(np.float64))
        zeros = np.count_nonzero(registers == 0)
        if estimate <= 2.5 * m and zeros:  # small-range correction
            estimate = m * np.log(m / zeros)
        return estimate
    return hll_estimate, hll_update


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To compare register sizes quickly, we hash every phrase once and keep only
    the hashes (8 bytes each). The exact number of distinct phrases is taken to
    be the number of distinct hashes; with 64-bit hashes and a few million
    phrases, the chance that two different phrases share a hash is about one in
    a million.
    """)
    return


@app.cell
def _(CORPUS, hash_phrases, np, phrase_batches):
    all_hashes = np.concatenate([hash_phrases(_b) for _b in phrase_batches(CORPUS)])
    unique_hashes, unique_counts = np.unique(all_hashes, return_counts=True)
    true_distinct = len(unique_hashes)
    f"{len(all_hashes):,} phrases, {true_distinct:,} distinct"
    return all_hashes, true_distinct, unique_counts, unique_hashes


@app.cell
def _(all_hashes, hll_estimate, hll_update, np, pd, true_distinct):
    _rows = []
    for _p in range(4, 19):
        _reg = np.zeros(2**_p, dtype=np.uint8)
        hll_update(_reg, all_hashes)
        _est = hll_estimate(_reg)
        _rows.append({
            "p": _p,
            "bytes": 2**_p,
            "estimate": round(_est),
            "error_pct": round(100 * abs(_est - true_distinct) / true_distinct, 2),
            "expected_pct": round(100 * 1.04 / np.sqrt(2**_p), 2),
        })
    hll_sweep = pd.DataFrame(_rows)
    return (hll_sweep,)


@app.cell
def _(mo):
    hll_p = mo.ui.slider(4, 18, value=14, label="p (registers = 2^p bytes)", show_value=True)
    hll_p
    return (hll_p,)


@app.cell
def _(alt, hll_p, hll_sweep, mo, true_distinct):
    _row = hll_sweep.set_index("p").loc[hll_p.value]
    _base = alt.Chart(hll_sweep).encode(
        x=alt.X("bytes:Q", scale=alt.Scale(type="log", base=2), title="register memory (bytes, log scale)"),
    )
    _expected = _base.mark_line(strokeDash=[4, 4], color="gray").encode(
        y=alt.Y("expected_pct:Q", title="error (%)"),
    )
    _observed = _base.mark_line(point=True).encode(
        y="error_pct:Q", tooltip=["p", "bytes", "estimate", "error_pct"],
    )
    _chosen = alt.Chart(hll_sweep[hll_sweep.p == hll_p.value]).mark_point(
        size=200, color="firebrick", filled=True,
    ).encode(x="bytes:Q", y="error_pct:Q")
    mo.vstack([
        (_expected + _observed + _chosen).properties(width=560, height=280),
        mo.md(
            f"With p = {hll_p.value}, the registers take {int(_row.bytes):,} bytes, and the estimate is "
            f"{int(_row.estimate):,} against {true_distinct:,} distinct phrases, an error of "
            f"{_row.error_pct}%. The dashed line is the expected error, 1.04/√m."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each doubling of memory reduces the expected error by a factor of √2, so
    going from 1% to 0.1% costs a hundred times the memory. For comparison,
    the exact `set` in section 3 used hundreds of MiB, and its size grows with
    the number of distinct phrases, while the registers stay the same size
    whether there are a thousand phrases or a billion.

    ## 6. Count-Min sketch: counting how often each item occurs

    A Count-Min sketch is a table of counters with `d` rows and `w` columns.
    Each row has its own hash function, which maps every phrase to one column.
    To add a phrase, add 1 to its counter in every row; to estimate a phrase's
    count, take the smallest of its `d` counters.

    Different phrases share counters, so a counter can only be too high, never
    too low, and the minimum over the rows is the counter least affected by
    other phrases. The overestimate is at most about `e·N/w` for a stream of
    `N` items, with a probability of failure of about `e`<sup>−d</sup>. The
    error therefore depends on the length of the stream, not on the number of
    distinct items, which makes it small for common phrases and large, relative
    to the true count, for rare ones.
    """)
    return


@app.cell
def _(np):
    def cms_indices(hashes, depth, width):
        """Return the column for each hash in each of `depth` rows."""
        h1 = hashes & np.uint64(0xFFFFFFFF)
        h2 = hashes >> np.uint64(32)
        # Row i uses h1 + i * h2, which behaves like an independent hash per row.
        return [((h1 + np.uint64(i) * h2) % np.uint64(width)).astype(np.int64) for i in range(depth)]

    def cms_add(table, hashes):
        """Add one occurrence of each hash to a Count-Min table, in place."""
        depth, width = table.shape
        for row, cols in enumerate(cms_indices(hashes, depth, width)):
            table[row] += np.bincount(cols, minlength=width).astype(table.dtype)

    def cms_query(table, hashes):
        """Estimate the count of each hash from a Count-Min table."""
        depth, width = table.shape
        return np.min([table[row][cols] for row, cols in enumerate(cms_indices(hashes, depth, width))], axis=0)
    return cms_add, cms_indices, cms_query


@app.cell
def _(mo):
    cms_width = mo.ui.slider(8, 22, value=16, label="log2 of width w (depth d = 4)", show_value=True)
    cms_width
    return (cms_width,)


@app.cell
def _(
    CORPUS,
    all_hashes,
    cms_add,
    cms_query,
    cms_width,
    hash_phrases,
    mo,
    np,
    pd,
    phrase_batches,
    unique_counts,
    unique_hashes,
):
    _width = 2**cms_width.value
    _table = np.zeros((4, _width), dtype=np.int32)
    cms_add(_table, all_hashes)

    def _true_count(h):
        _i = np.searchsorted(unique_hashes, h)
        return np.where(unique_hashes[np.minimum(_i, len(unique_hashes) - 1)] == h, unique_counts[_i], 0)

    _common = ["one of the", "out of the", "as well as", "it was a"]
    _batches = phrase_batches(CORPUS, size=5_000)
    for _ in range(10):  # skip the opening pages of the first book
        _first = next(_batches)
    _rare = [_p for _p, _c in zip(_first, _true_count(hash_phrases(_first))) if _c == 1][:4]
    _h = hash_phrases(_common + _rare)
    _points = pd.DataFrame({
        "phrase": _common + _rare,
        "true count": _true_count(_h),
        "estimate": cms_query(_table, _h),
    })
    _all = cms_query(_table, unique_hashes)
    _over = _all - unique_counts
    mo.vstack([
        mo.md(
            f"The table has {_width:,} columns and takes {_table.nbytes / 2**10:,.0f} KiB. "
            f"Across all {len(unique_hashes):,} distinct phrases, the average overestimate is "
            f"{_over.mean():.1f}, and {np.mean(_over == 0):.0%} of the counts are exact."
        ),
        _points,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Exercise: find the ten most common phrases in fixed memory

    Find the ten most common phrases using at most one fifth of the peak memory
    of the exact `Counter` in section 3. The check compares your ten phrases with
    the exact ones, in any order; your counts may be slightly too high.

    A Count-Min table contains counts but not the phrases, so the most common
    phrases cannot be read from it directly. You will need to keep a short list
    of candidate phrases alongside it, and decide when a phrase joins or leaves
    the list. `cms_indices`, `cms_add` and `cms_query` are available in the new
    process. Run the exact versions in section 3 first, then compare the
    memory your function uses with the first row of that table, which reads
    the text and does nothing else; you may find that the sketch is not the
    largest thing in memory.
    """)
    return


@app.cell
def _():
    def my_top10():
        # Replace this with your own version. CORPUS, np, pd, phrase_batches,
        # hash_phrases, cms_indices, cms_add and cms_query are already defined.
        from collections import Counter
        counts = Counter()
        for batch in phrase_batches(CORPUS):
            counts.update(batch)
        return pd.DataFrame(counts.most_common(10), columns=["phrase", "count"])
    return (my_top10,)


@app.cell
def _(
    cms_add,
    cms_indices,
    cms_query,
    exact_stats,
    exact_top,
    measure_fresh,
    mo,
    my_top10,
    pd,
):
    mine, my_stats = measure_fresh("exercise: my_top10", my_top10, cms_indices, cms_add, cms_query)
    _same = set(mine["phrase"]) == set(exact_top["phrase"])
    _target = exact_stats.loc["top 10 phrases (Counter)", "peak_extra_MiB"] / 5
    _ok = _same and my_stats["peak_extra_MiB"] <= _target
    mo.vstack([
        mo.hstack([pd.DataFrame([my_stats]).set_index("step"), mine], justify="start", gap=2),
        mo.callout(
            mo.md(
                f"Same ten phrases: **{_same}**. Peak target: **≤ {_target:.0f} MiB**. "
                f"Yours: **{my_stats['peak_extra_MiB']} MiB**."
            ),
            kind="success" if _ok else "warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both sketches can be merged: two sets of HyperLogLog registers combine by
    taking the larger value in each position, and two Count-Min tables combine
    by adding them. Like the sums and counts behind a mean in week 1, this
    means that separate workers can each summarize part of the data and
    combine their summaries at the end. Next week we will look at that kind of
    parallelism on one machine, and at how much faster it makes a computation.
    """)
    return


if __name__ == "__main__":
    app.run()
