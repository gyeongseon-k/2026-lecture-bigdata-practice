#!/usr/bin/env python3
"""Week 4 · Task 2 — Find the size where exact stops being possible.

Textbook §4.1 (the stream model), §4.4, §4.5.

Sketches exist because the exact answer does not fit. That sentence is easy to
agree with and hard to feel, so this task makes you watch it happen on your own
machine: hold every distinct item in a set, keep raising the stream size, and
record where your laptop stops coping.

    python3 task2_limits.py --sizes 100000,400000,1600000
    python3 task2_limits.py --sizes 6400000            # keep going

Your numbers will not match anybody else's. That is the point.
"""
import argparse, json, os, platform, resource, time, tracemalloc


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def machine():
    return {"platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version()}


def stream(n, distinct_ratio=0.4, seed=246):
    """A stream of n items with about n*distinct_ratio distinct values."""
    import random
    rng = random.Random(seed)
    span = max(1, int(n * distinct_ratio))
    for _ in range(n):
        yield f"key-{rng.randrange(span)}"


def rss_peak():
    """이 프로세스가 지금까지 실제로 점유한 최대 메모리 (바이트).

    tracemalloc 이 보고하는 값은 '파이썬 객체가 요청한 양' 이고, 이쪽은
    '운영체제가 실제로 내준 양' 이다. 후자가 진짜 한계를 결정한다 - 인터프리터
    자체, 할당자 여유분, 그리고 tracemalloc 자신의 장부까지 포함하기 때문이다.
    A2 의 '무엇이 먼저 바닥났나' 는 이 숫자로 판단해야 맞다.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def exact_distinct(n, trace=True):
    """The honest answer: hold every distinct item.

    trace=False 면 tracemalloc 을 켜지 않는다. 메모리는 못 재지만 시간이 정확해진다.

    왜 나눠야 하는가:
      tracemalloc 은 할당을 전부 추적하므로 이 루프를 약 5.2 배 느리게 만든다
      (n=200,000 에서 0.125s -> 0.653s 로 실측). 그런데 이 과제의 A3/A4 는
      메모리 질문이라 tracemalloc 이 계측 도구 자체이고, A2 는 '시간과 메모리 중
      무엇이 먼저 바닥났나' 라서 시간이 정확해야 한다.
      끄면 A3/A4 를 잃고 켜면 A2 가 틀리므로, 두 번 나눠 잰다.
      메모리는 traced 실행에서, 시간은 no-trace 실행에서 읽는다.

      덤으로 tracemalloc 은 자기 장부에도 메모리를 쓴다. distinct 가 수백만
      개가 되면 그 장부 때문에 먼저 터질 수 있어서, 큰 n 은 no-trace 쪽이
      더 멀리 간다.
    """
    if trace:
        tracemalloc.start()
    t0 = time.perf_counter()
    seen = set()
    for x in stream(n):
        seen.add(x)
    elapsed = time.perf_counter() - t0
    peak = None
    if trace:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return len(seen), elapsed, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="100000,400000,1600000")
    p.add_argument("--no-trace", action="store_true",
                   help="tracemalloc 없이 시간만 잰다 (시간이 정확해지고 더 큰 n 까지 간다)")
    a = p.parse_args()
    trace = not a.no_trace
    os.makedirs(OUT, exist_ok=True)

    try:
        from task1_sketches import flajolet_martin
    except Exception:
        flajolet_martin = None

    rows = []
    for n in [int(x) for x in a.sizes.split(",")]:
        true, t_exact, m_exact = exact_distinct(n, trace=trace)
        # traced=True 인 행은 시간이 약 5 배 부풀려져 있다.
        # 시간은 traced=False 행에서, 메모리는 traced=True 행에서 읽어야 한다.
        row = {"n": n, "traced": trace, "true_distinct": true,
               "exact_s": t_exact, "exact_peak_bytes": m_exact,
               "rss_after_exact_bytes": rss_peak()}

        if flajolet_martin is not None:
            try:
                if trace:
                    tracemalloc.start()
                t0 = time.perf_counter()
                est = flajolet_martin(stream(n))
                t_fm = time.perf_counter() - t0
                m_fm = None
                if trace:
                    _, m_fm = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                row.update({"fm_estimate": est, "fm_s": t_fm,
                            "fm_peak_bytes": m_fm,
                            "fm_ratio": est / true if true else None,
                            "rss_after_fm_bytes": rss_peak()})
            except NotImplementedError:
                if trace:
                    tracemalloc.stop()

        rows.append(row)
        mb = lambda v: f"{v / 1e6:>8.1f} MB" if v is not None else f"{'-':>11}"
        line = (f"  n={n:>10,}  distinct {true:>9,}   exact {t_exact:>7.2f}s "
                f"{mb(m_exact)}")
        if "fm_s" in row:
            line += (f"   |  fm {row['fm_s']:>7.2f}s {mb(row['fm_peak_bytes'])}"
                     f"  {row['fm_ratio']:.2f}x")
        line += f"   | RSS {rss_peak() / 1e6:>7.0f} MB"
        print(line)

    path = os.path.join(OUT, "limits.json")
    prior = json.load(open(path)) if os.path.exists(path) else {"runs": []}
    prior["machine"] = machine()
    prior["runs"].extend(rows)
    json.dump(prior, open(path, "w"), indent=2)
    print(f"\n  -> out/limits.json  ({len(prior['runs'])} measurement(s))")
    print("  Keep raising --sizes until the exact version is unbearable. "
          "Record where, and what ran out.")


if __name__ == "__main__":
    main()
