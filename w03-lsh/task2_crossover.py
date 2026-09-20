#!/usr/bin/env python3
"""Week 3 · Task 2 — Find the crossover on your own machine.

Textbook §3.4.

Everybody knows brute force is quadratic and LSH is not. That is not the
interesting question. The interesting question is **where, on the machine in
front of you, does it start to matter** - and that answer is yours alone. It
depends on your CPU, your memory, and how big your shingle sets are.

This script gives you the timing loop. The two methods are yours: import them
from Task 1 and Task 3.

    python3 task2_crossover.py --sizes 500,1000,2000,4000
    python3 task2_crossover.py --sizes 8000,16000          # keep going

Write down where it hurts. That is the deliverable.
"""
import argparse, json, os, platform, random, time, tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def machine():
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


def build_scaled(n, seed=None):
    """bench.build() 와 같은 성격의 문서를 n 개 만든다.

    bench.build() 는 문서를 2,120 개만 만들고, 그래서 bench.build()[:n] 로는
    n 을 그 위로 올릴 수 없다. 이 과제는 '불쾌해질 때까지' 키우는 것이 목적이므로
    개수만 자유롭게 늘릴 수 있는 생성기가 필요하다. bench.py 는 채점 harness 라
    수정하지 않고, 같은 규칙을 여기서 다시 쓴다.

    bench.build() 에서 그대로 가져온 성질:

      - 문서 하나 = 어휘 5,000 개 중에서 고른 샤글 60 개
      - 어휘 크기 5,000 은 n 이 커져도 고정한다. 무관한 두 문서의 유사도는
        '5,000 중 60 개를 두 번 뽑아 얼마나 겹치나' 로 정해지고 (약 0.006)
        문서 수와 무관하다. 어휘를 같이 키우면 문서들이 점점 안 겹치게 되어
        n 마다 난이도가 달라져 버린다. 고정해야 크기별 비교가 성립한다.
      - 근접중복은 원본을 복사해 샤글 4~14 개를 바꿔 심는다
      - 심는 개수는 n 에 비례시킨다 (원본 비율 120/2120 = 약 5.7%).
        개수를 120 으로 고정하면 n 이 클수록 '찾을 것이 거의 없는' 쉬운 문제가
        되어 크기별 비교가 공정하지 않다.
      - 마지막에 섞는다. 원본과 사본이 나란히 붙어 있으면 앞에서부터 비교하는
        방식이 부당하게 유리해진다.

    시드를 고정하므로 같은 n 은 항상 같은 문서를 만든다. n=2120 을 넣으면
    bench.build() 와 완전히 같은 문서가 나온다 (아래 --selftest 로 확인).
    """
    import bench
    rng = random.Random(bench.SEED if seed is None else seed)

    # 심을 개수를 bench 의 비율대로 정하고, 나머지를 원본으로 만든다.
    ratio = bench.PLANTED / (bench.N_DOCS + bench.PLANTED)
    n_planted = max(1, round(n * ratio))
    n_base = n - n_planted

    docs = [set(rng.sample(range(bench.VOCAB), bench.SHINGLES))
            for _ in range(n_base)]

    # 근접중복 심기: 원본을 하나 골라 복사하고 샤글 몇 개만 갈아끼운다.
    for _ in range(n_planted):
        i = rng.randrange(n_base)
        clone = set(docs[i])
        for _ in range(rng.randint(4, 14)):
            clone.discard(rng.choice(list(clone)))
            clone.add(rng.randrange(bench.VOCAB))
        docs.append(clone)

    rng.shuffle(docs)
    return docs


def selftest():
    """build_scaled(2120) 이 bench.build() 와 같은지 확인한다."""
    import bench
    mine, theirs = build_scaled(2120), bench.build()
    same = mine == theirs
    print(f"  build_scaled(2120) == bench.build()  ->  {same}")
    for n in (500, 4000):
        docs = build_scaled(n)
        avg = sum(len(d) for d in docs) / len(docs)
        print(f"  n={n:>5}  문서 {len(docs):>5} 개  평균 샤글 {avg:.1f} 개")
    return 0 if same else 1


def timed(fn, *args, trace=True):
    """Wall time and peak memory of one call.

    trace=False 면 tracemalloc 을 켜지 않는다. 메모리는 못 재지만 시간이 정확해진다.

    왜 이 선택지가 필요한가 (A4 에 해당하는 측정 오류):
      tracemalloc 은 파이썬의 모든 메모리 할당을 추적하므로, 할당이 잦은 코드를
      크게 느리게 만든다. 문제는 그 벌점이 두 방식에 똑같이 붙지 않는다는 것이다.
        - 브루트포스: 비교 한 번에 임시 집합 두 개 -> 약 1.4 배 느려짐
        - LSH: 시그니처를 갱신할 때마다 리스트를 새로 만듦 -> 약 12 배 느려짐
      둘을 한 번에 재면 LSH 만 심하게 손해를 봐서 교차점이 실제보다 훨씬 오른쪽으로
      밀린다. 그래서 시간과 메모리를 따로 잰다.
    """
    if not trace:
        t0 = time.perf_counter()
        result = fn(*args)
        return result, time.perf_counter() - t0, None

    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="250,500,1000,2000",
                   help="comma-separated document counts to try")
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--no-trace", action="store_true",
                   help="tracemalloc 없이 시간만 잰다 (시간이 정확해짐)")
    p.add_argument("--selftest", action="store_true",
                   help="생성기가 bench.build() 와 같은 데이터를 만드는지 확인")
    a = p.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    os.makedirs(OUT, exist_ok=True)

    import bench
    from task3_scale import BruteForce
    try:
        from task3_scale import YourFinder
    except Exception:
        YourFinder = None

    rows = []
    for n in [int(x) for x in a.sizes.split(",")]:
        docs = build_scaled(n)
        sim = bench.Counter()
        trace = not a.no_trace
        _, t_brute, m_brute = timed(BruteForce(a.threshold).find, docs, sim,
                                    trace=trace)
        c_brute = sim.calls

        # traced=True 인 행은 시간이 부풀려져 있다. 시간은 traced=False 행을,
        # 메모리는 traced=True 행을 봐야 한다.
        row = {"n": n, "traced": trace, "brute_s": t_brute, "brute_calls": c_brute,
               "brute_peak_bytes": m_brute}

        if YourFinder is not None:
            sim2 = bench.Counter()
            try:
                _, t_lsh, m_lsh = timed(YourFinder(a.threshold).find, docs, sim2,
                                        trace=trace)
                row.update({"lsh_s": t_lsh, "lsh_calls": sim2.calls,
                            "lsh_peak_bytes": m_lsh})
            except NotImplementedError:
                pass

        rows.append(row)
        mb = lambda v: f"{v/1e6:>7.1f}MB" if v is not None else f"{'-':>9}"
        line = f"  n={n:>6}  brute {t_brute:>8.2f}s  {c_brute:>12,} cmp  {mb(m_brute)}"
        if "lsh_s" in row:
            line += (f"   |  lsh {row['lsh_s']:>7.2f}s  {row['lsh_calls']:>9,} cmp"
                     f"  {mb(row['lsh_peak_bytes'])}")
        print(line)

    path = os.path.join(OUT, "crossover.json")
    prior = json.load(open(path)) if os.path.exists(path) else {"runs": []}
    prior["machine"] = machine()
    prior["runs"].extend(rows)
    json.dump(prior, open(path, "w"), indent=2)
    print(f"\n  -> out/crossover.json  ({len(prior['runs'])} measurement(s))")
    print("  Keep raising --sizes until something becomes unpleasant. Record where.")


if __name__ == "__main__":
    main()
