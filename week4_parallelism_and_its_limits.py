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
    # Week 4: Parallelism and its limits

    The first three weeks used one core at a time, apart from whatever the
    query engines did internally. This week we use several cores explicitly,
    measure how much faster that makes a computation, and fit the measurements
    to Amdahl's law to predict how far the speedup could go with more cores.
    We finish with McSherry's COST argument and some arithmetic for deciding
    when a cluster is needed.
    """)
    return


@app.cell
def _():
    import inspect
    import json
    import os
    import re
    import subprocess
    import sys
    import textwrap
    from pathlib import Path
    from urllib.request import urlretrieve

    import altair as alt
    import numpy as np
    import pandas as pd
    import psutil
    return (
        Path,
        alt,
        inspect,
        json,
        np,
        os,
        pd,
        psutil,
        re,
        subprocess,
        sys,
        textwrap,
        urlretrieve,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. How many cores do we have?

    Python offers several ways to count CPUs, and in a container they can
    disagree. `os.cpu_count()` and `psutil` report the CPUs of the physical
    host. The operating system can restrict a process to some of them (the
    *affinity*), and a container can be given a CPU quota (in the *cgroup*
    file `cpu.max`), which limits how much CPU time it may use however many
    CPUs it can see. The number that matters for parallel work is the
    smallest of these.
    """)
    return


@app.cell
def _(os, pd, psutil):
    def _read(path):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return None

    _quota = _read("/sys/fs/cgroup/cpu.max")
    _cgroup_cpus = None
    if _quota and not _quota.startswith("max"):
        _q, _period = _quota.split()
        _cgroup_cpus = int(_q) / int(_period)
    try:
        _affinity = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS has no affinity call
        _affinity = None
    _mem = _read("/sys/fs/cgroup/memory.max")

    USABLE_CPUS = max(1, int(min(c for c in [os.cpu_count(), _affinity, _cgroup_cpus] if c)))
    WORKERS = sorted({k for k in [1, 2, 4, 8, 16, USABLE_CPUS, 2 * USABLE_CPUS] if k <= min(2 * USABLE_CPUS, 16)})
    pd.DataFrame([
        {"source": "os.cpu_count()", "value": os.cpu_count()},
        {"source": "psutil.virtual_memory().total (GiB)", "value": round(psutil.virtual_memory().total / 2**30, 1)},
        {"source": "os.sched_getaffinity(0)", "value": _affinity},
        {"source": "cgroup cpu.max (CPUs)", "value": _cgroup_cpus},
        {"source": "cgroup memory.max (GiB)", "value": round(int(_mem) / 2**30, 1) if _mem and _mem.isdigit() else _mem},
        {"source": "usable CPUs (smallest of the above)", "value": USABLE_CPUS},
        {"source": "worker counts we will try", "value": str(WORKERS)},
    ])
    return USABLE_CPUS, WORKERS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Get the data

    We use the same Gutenberg books as last week (the PG-19 test split), written
    to one plain text file.
    """)
    return


@app.cell
def _(mo):
    books = mo.ui.slider(10, 100, step=10, value=100, label="Books to use", show_value=True)
    books
    return (books,)


@app.cell
def _(Path, books, pd, urlretrieve):
    PG19_URL = (
        "https://huggingface.co/datasets/emozilla/pg19/resolve/main/data/"
        "test-00000-of-00001-29a571947c0b5ccc.parquet"
    )
    DATA = Path("data_week4")
    DATA.mkdir(exist_ok=True)
    _parquet = DATA / "pg19_test.parquet"
    if not _parquet.exists():
        urlretrieve(PG19_URL, _parquet)
    _path = DATA / f"corpus_{books.value}.txt"
    if not _path.exists():
        _texts = pd.read_parquet(_parquet, columns=["text"]).text.head(books.value)
        _path.write_text("\n".join(_texts), encoding="utf-8")
    CORPUS = str(_path.resolve())
    CORPUS_MIB = _path.stat().st_size / 2**20
    f"{books.value} books, {CORPUS_MIB:.0f} MiB of text"
    return CORPUS, CORPUS_MIB, DATA


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Running parallel code from a notebook

    `multiprocessing.Pool` starts worker processes and sends each of them a
    function and its arguments. On macOS, and on Linux from Python 3.14, the
    workers are new interpreters that import the function by name from the
    main program's file, which a notebook cell does not have. `run_script`
    therefore writes the function, the helpers below and a timing harness to
    a file, runs that file in a new process, and reports the elapsed time and
    the result.
    """)
    return


@app.cell
def _(np, re):
    WORD_RE = re.compile(r"[\w']+")

    def count_words(lines):
        """Count the words in an iterable of lines (the map step)."""
        from collections import Counter
        counts = Counter()
        for line in lines:
            counts.update(WORD_RE.findall(line.lower()))
        return counts

    def split_lines(path, n):
        """Read a file and split its lines into n lists of nearly equal length."""
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        size = -(-len(lines) // n)
        return [lines[i:i + size] for i in range(0, len(lines), size)]

    def hll_registers(lines, p=14):
        """HyperLogLog registers (as in week 3) for the three-word phrases in some lines."""
        import pandas as pd
        registers = np.zeros(2**p, dtype=np.uint8)
        prev, batch = [], []
        def add(batch):
            h = pd.util.hash_array(np.asarray(batch, dtype=object))
            index = (h >> np.uint64(64 - p)).astype(np.int64)
            rest = ((h << np.uint64(p)) >> np.uint64(32)).astype(np.float64)
            rank = np.where(rest > 0, 32 - np.floor(np.log2(np.maximum(rest, 1))), 33)
            np.maximum.at(registers, index, rank.astype(np.uint8))
        for line in lines:
            words = prev + WORD_RE.findall(line.lower())
            batch.extend([" ".join(words[i:i + 3]) for i in range(len(words) - 2)])
            prev = words[-2:]
            if len(batch) >= 50_000:
                add(batch)
                batch = []
        if batch:
            add(batch)
        return registers

    def hll_estimate(registers):
        """Estimate the number of distinct items from HyperLogLog registers."""
        m = len(registers)
        return 0.7213 / (1 + 1.079 / m) * m * m / np.sum(2.0 ** -registers.astype(np.float64))
    return count_words, hll_estimate, hll_registers, split_lines


@app.cell
def _(
    CORPUS,
    DATA,
    count_words,
    hll_estimate,
    hll_registers,
    inspect,
    json,
    split_lines,
    subprocess,
    sys,
    textwrap,
):
    _HEADER = f"""
    import re
    from collections import Counter
    from multiprocessing import Pool
    import numpy as np
    CORPUS = {CORPUS!r}
    WORD_RE = re.compile(r"[\\w']+")
    """

    _FOOTER = """
    if __name__ == "__main__":
        import json, time
        _t0 = time.perf_counter()
        _result = FN(*ARGS)
        _elapsed = time.perf_counter() - _t0
        if hasattr(_result, "most_common"):
            _result = _result.most_common(10)
        print(json.dumps({"seconds": round(_elapsed, 2), "result": _result}, default=float))
    """

    def run_script(label, fn, *args, helpers=()):
        """Run fn(*args) in a new Python process, from a file; return (result, stats dict).

        Functions that fn calls, other than the shared helpers, go in `helpers`.
        """
        parts = [textwrap.dedent(_HEADER)]
        for _f in (count_words, split_lines, hll_registers, hll_estimate, *helpers, fn):
            parts.append(textwrap.dedent(inspect.getsource(_f)))
        parts.append(textwrap.dedent(_FOOTER).replace("FN", fn.__name__).replace("ARGS", repr(args)))
        _script = DATA / f"_run_{fn.__name__}.py"
        _script.write_text("\n".join(parts))
        proc = subprocess.run([sys.executable, str(_script)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-3000:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        result = out.pop("result")
        return result, {"step": label, **out}
    return (run_script,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Make it fast before making it parallel

    We estimate π by scattering points at random in a unit square and counting
    the fraction that land inside the quarter circle, which is π/4. We do it
    three ways: a Python loop, NumPy on blocks of a million points, and the same
    Python loop split across several worker processes. Before running them,
    predict how many workers the loop needs to match NumPy.
    """)
    return


@app.cell
def _():
    POINTS = 20_000_000

    def pi_loop(n, seed=0):
        import random
        rng = random.Random(seed)
        inside = 0
        for _ in range(n):
            x = rng.random()
            y = rng.random()
            inside += x * x + y * y < 1
        return 4 * inside / n

    def pi_numpy(n, seed=0):
        rng = np.random.default_rng(seed)
        inside = 0
        for start in range(0, n, 1_000_000):
            m = min(1_000_000, n - start)
            x = rng.random(m)
            y = rng.random(m)
            inside += int(np.count_nonzero(x * x + y * y < 1))
        return 4 * inside / n

    def pi_pool(n, workers):
        with Pool(workers) as pool:
            parts = pool.starmap(pi_loop, [(n // workers, seed) for seed in range(workers)])
        return sum(parts) / workers
    return POINTS, pi_loop, pi_numpy, pi_pool


@app.cell
def _(mo):
    run_pi = mo.ui.run_button(label="Run the π comparison")
    run_pi
    return (run_pi,)


@app.cell
def _(
    POINTS,
    WORKERS,
    mo,
    pd,
    pi_loop,
    pi_numpy,
    pi_pool,
    run_pi,
    run_script,
):
    mo.stop(not run_pi.value, mo.md("*Press the button to run the π comparison.*"))
    _runs = [
        run_script("Python loop", pi_loop, POINTS),
        run_script("NumPy", pi_numpy, POINTS),
    ] + [run_script(f"Python loop, {k} workers", pi_pool, POINTS, k, helpers=[pi_loop]) for k in WORKERS]
    pi_stats = pd.DataFrame([{**s, "estimate": round(r, 5)} for r, s in _runs]).set_index("step")
    pi_stats["speedup vs loop"] = (pi_stats.seconds.iloc[0] / pi_stats.seconds).round(2)
    _loop, _numpy = pi_stats.seconds.iloc[0], pi_stats.seconds.iloc[1]
    _best = pi_stats.seconds.iloc[2:].min()
    mo.vstack([
        pi_stats,
        mo.md(
            f"NumPy on one core is {_loop / _numpy:.0f} times as fast as the loop. "
            f"Even if the loop scaled perfectly, it would need about {_loop / _numpy:.0f} cores "
            f"to match NumPy; the fastest parallel run here took {_best:.2f} s, against "
            f"{_numpy:.2f} s for NumPy."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The run with one worker is slower than the plain loop, because the worker
    is a new Python process that has to start and import its libraries before
    it does any useful work.

    ## 5. Word count as MapReduce on one machine

    Homework 1's word count has a map step (count the words in some lines) and
    a reduce step (add the counts together). `multiprocessing` can run the map
    step on several cores: each worker counts the words in its share of the
    lines and returns a `Counter`, and the main process adds the `Counter`s.

    How the work is divided matters. The first version below sends one line at a
    time to a worker, as a naive translation of a mapper would, and we run it on
    the first 50,000 lines only, next to the serial count of the same lines. The
    second version sends each worker one large chunk of lines.
    """)
    return


@app.cell
def _():
    def wc_serial():
        with open(CORPUS, encoding="utf-8") as f:
            return count_words(f)

    def wc_serial_first(n_lines):
        from itertools import islice
        with open(CORPUS, encoding="utf-8") as f:
            return count_words(islice(f, n_lines))

    def wc_per_line(n_lines, workers):
        from itertools import islice
        total = Counter()
        with open(CORPUS, encoding="utf-8") as f, Pool(workers) as pool:
            for counts in pool.imap_unordered(count_words, ([line] for line in islice(f, n_lines))):
                total.update(counts)
        return total

    def wc_chunks(workers):
        total = Counter()
        with Pool(workers) as pool:
            for counts in pool.imap_unordered(count_words, split_lines(CORPUS, workers)):
                total.update(counts)
        return total
    return wc_chunks, wc_per_line, wc_serial, wc_serial_first


@app.cell
def _(mo):
    run_wc = mo.ui.run_button(label="Run the word counts")
    run_wc
    return (run_wc,)


@app.cell
def _(
    USABLE_CPUS,
    WORKERS,
    mo,
    pd,
    run_script,
    run_wc,
    wc_chunks,
    wc_per_line,
    wc_serial,
    wc_serial_first,
):
    mo.stop(not run_wc.value, mo.md("*Press the button to run the word counts.*"))
    _granularity = pd.DataFrame([
        run_script("serial, first 50,000 lines", wc_serial_first, 50_000)[1],
        run_script(f"one line per task, {USABLE_CPUS} workers", wc_per_line, 50_000, USABLE_CPUS)[1],
    ]).set_index("step")
    _ref, _serial = run_script("serial, all lines", wc_serial)
    _runs = [run_script(f"{k} workers", wc_chunks, k) for k in WORKERS]
    _same = all(r == _ref for r, _ in _runs)
    wc_stats = pd.DataFrame([{**s, "workers": k} for k, (_, s) in zip(WORKERS, _runs)]).set_index("step")
    wc_serial_seconds = _serial["seconds"]
    wc_stats["speedup vs 1 worker"] = (wc_stats.seconds.iloc[0] / wc_stats.seconds).round(2)
    mo.vstack([
        _granularity,
        mo.md(
            f"Counting all the lines in one process took {wc_serial_seconds:.2f} s. "
            f"The chunked runs below give the same ten most common words: **{_same}**."
        ),
        wc_stats,
    ])
    return wc_serial_seconds, wc_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sending one line per task is far slower than counting serially, because
    each line has to be pickled, sent to a worker, counted, and sent back as a
    `Counter`, and that overhead is much larger than the counting itself.

    ## 6. Amdahl's law

    Suppose a fraction `s` of a program's running time cannot be parallelized
    (here, starting the workers, reading and splitting the file, and adding up
    the `Counter`s) and the rest divides evenly among `k` workers. The speedup
    over one worker is then

    $$S(k) = \frac{1}{s + (1 - s)/k},$$

    which can never exceed 1/`s`, however many workers there are. We estimate
    `s` from the measured speedups, using only runs with no more workers than
    usable CPUs, and extend the fitted curve to 64 workers.
    """)
    return


@app.cell
def _(USABLE_CPUS, alt, mo, np, pd, wc_stats):
    _fit = wc_stats[(wc_stats.workers > 1) & (wc_stats.workers <= USABLE_CPUS)]
    if len(_fit):
        _x = 1 - 1 / _fit.workers.to_numpy()
        _y = 1 / _fit["speedup vs 1 worker"].to_numpy() - 1 / _fit.workers.to_numpy()
        serial_fraction = float(np.clip((_x * _y).sum() / (_x * _x).sum(), 0.001, 1))
    else:
        serial_fraction = None
    mo.stop(serial_fraction is None, mo.md("*Amdahl's law needs at least two usable CPUs to fit.*"))

    _k = np.unique(np.round(np.logspace(0, 6, 61, base=2)))
    _curves = pd.concat([
        pd.DataFrame({"workers": _k, "speedup": 1 / (serial_fraction + (1 - serial_fraction) / _k),
                      "line": f"Amdahl, s = {serial_fraction:.2f}"}),
        pd.DataFrame({"workers": _k[_k <= 16], "speedup": _k[_k <= 16], "line": "perfect scaling"}),
    ])
    _lines = alt.Chart(_curves).mark_line().encode(
        x=alt.X("workers:Q", scale=alt.Scale(type="log", base=2), title="workers (log scale)"),
        y=alt.Y("speedup:Q", title="speedup over one worker"),
        color=alt.Color("line:N", title=None),
        strokeDash=alt.StrokeDash("line:N", legend=None),
    )
    _points = alt.Chart(wc_stats.reset_index()).mark_point(filled=True, size=80, color="black").encode(
        x="workers:Q", y="speedup vs 1 worker:Q", tooltip=["step", "seconds", "speedup vs 1 worker"],
    )
    mo.vstack([
        (_lines + _points).properties(width=560, height=300),
        mo.md(
            f"The fitted serial fraction is {serial_fraction:.2f}, so no number of workers can make this "
            f"program more than {1 / serial_fraction:.1f} times as fast as one worker, and 64 workers "
            f"would give about {1 / (serial_fraction + (1 - serial_fraction) / 64):.1f} times. Points to the "
            f"right of {USABLE_CPUS} workers use more workers than CPUs, and are not part of the fit."
        ),
    ])
    return (serial_fraction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Scalability, but at what COST?

    In 2015, Frank McSherry, Michael Isard and Derek Murray compared published
    results for graph-processing systems running on clusters with a
    single-threaded program on a laptop. They defined a system's COST (the
    Configuration that Outperforms a Single Thread) as the number of cores it
    needs before it beats a competent single-threaded implementation. Many
    systems that scaled well had a COST of hundreds of cores, and some were
    slower than the single thread in every configuration reported.

    Section 4 is a small example of the same thing: the parallel loop scales,
    but because each worker runs slow code, it needs about as many cores as the
    summary under that table gives before it matches NumPy on one core. A good
    speedup curve does not show whether the program being sped up is
    efficient, so the comparison that matters is with the best single-threaded
    program you can write.

    ## 8. When is a cluster needed?

    The table below gives times extrapolated from the word count above, using this machine's
    measured rate for one worker and the fitted serial fraction. It assumes that
    the serial fraction stays the same as the data grow, and that a cluster's
    cores run at the same rate as these, which is optimistic for a cluster
    because the data have to be sent over a network.
    """)
    return


@app.cell
def _(CORPUS_MIB, USABLE_CPUS, mo, pd, serial_fraction, wc_stats):
    _rate = CORPUS_MIB / wc_stats.seconds.iloc[0]  # MiB per second for one worker

    def _fmt(seconds):
        for _unit, _size in [("days", 86400), ("hours", 3600), ("minutes", 60)]:
            if seconds >= _size:
                return f"{seconds / _size:.1f} {_unit}"
        return f"{seconds:.0f} s"

    def _time(mib, cores):
        return mib / _rate * (serial_fraction + (1 - serial_fraction) / cores)

    pd.DataFrame([
        {
            "data": _label,
            f"this machine ({USABLE_CPUS} CPUs)": _fmt(_time(_mib, USABLE_CPUS)),
            "one 64-core machine": _fmt(_time(_mib, 64)),
            "1,000 cores": _fmt(_time(_mib, 1000)),
        }
        for _label, _mib in [("1 GiB", 2**10), ("100 GiB", 100 * 2**10), ("10 TiB", 10 * 2**20), ("1 PiB", 2**30)]
    ]).pipe(lambda d: mo.vstack([mo.md(f"One worker counts words at about {_rate:.0f} MiB/s."), d]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With a fixed serial fraction, adding cores stops helping long before a
    thousand, and in practice frameworks such as Spark are built to avoid a
    fixed serial step by also distributing the reading and the reduce. A
    cluster becomes necessary when the data do not fit on the disks of one
    machine, or when the time on the largest machine you can use is longer than
    you can wait. Cloud machines with more than a hundred cores and several TiB
    of memory can be rented by the hour, which places that point well beyond the
    size of the data in this course.

    ## 9. Exercise: count distinct phrases in parallel

    Last week's HyperLogLog registers can be merged: registers computed from
    different parts of the data combine by taking the larger value in each
    position. `my_hll` below computes the registers for the whole text in one
    process. Rewrite it so that it uses `workers` processes, each computing
    registers for part of the lines, and merges the results. `hll_registers`,
    `hll_estimate`, `split_lines`, `Pool`, `np` and `CORPUS` are available.

    The check compares your estimate with the serial one (it should agree to
    within 0.1%) and requires a speedup of at least 1.4 with one worker per
    usable CPU.
    """)
    return


@app.cell
def _():
    def my_hll(workers):
        # Replace this with a parallel version.
        with open(CORPUS, encoding="utf-8") as f:
            registers = hll_registers(f)
        return hll_estimate(registers)

    def hll_serial():
        with open(CORPUS, encoding="utf-8") as f:
            return hll_estimate(hll_registers(f))
    return hll_serial, my_hll


@app.cell
def _(hll_serial, run_script):
    hll_ref, hll_ref_stats = run_script("serial HyperLogLog", hll_serial)
    return hll_ref, hll_ref_stats


@app.cell
def _(USABLE_CPUS, hll_ref, hll_ref_stats, mo, my_hll, pd, run_script):
    mo.stop(USABLE_CPUS < 2, mo.md("*This machine has one usable CPU, so a parallel version cannot be faster.*"))
    _mine, _stats = run_script(f"my_hll, {USABLE_CPUS} workers", my_hll, USABLE_CPUS)
    _error = abs(_mine - hll_ref) / hll_ref
    _speedup = hll_ref_stats["seconds"] / _stats["seconds"]
    _ok = _error <= 0.001 and _speedup >= 1.4
    mo.vstack([
        pd.DataFrame([{**hll_ref_stats, "estimate": round(hll_ref)}, {**_stats, "estimate": round(_mine)}]).set_index("step"),
        mo.callout(
            mo.md(
                f"Difference from the serial estimate: **{_error:.3%}** (at most 0.1%). "
                f"Speedup: **{_speedup:.2f}** (at least 1.4)."
            ),
            kind="success" if _ok else "warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Over the four weeks we used less memory by choosing data types and file
    formats, handed the work to query engines that read only what a query
    needs, accepted approximate answers of a fixed size, and used several cores.
    Each of those steps made a larger dataset manageable on one machine, and the
    arithmetic in section 8 is a way to check whether the data you have in mind
    still fit.
    """)
    return


if __name__ == "__main__":
    app.run()
