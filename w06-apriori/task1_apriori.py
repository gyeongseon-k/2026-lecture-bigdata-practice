#!/usr/bin/env python3
"""Week 6 · Task 1 — A-Priori, and why the second pass is the whole problem.

Textbook §6.1 (market-basket model), §6.2 (A-Priori).

Finding frequent pairs is easy until you count how many pairs there are. With
n items there are n(n-1)/2 of them, and for a supermarket's 10,000 items that
is 50 million counters before you have read a single basket.

A-Priori's observation is one sentence long: **a pair cannot be frequent unless
both of its items are frequent.** One extra pass over the data buys you a much
smaller second pass, and the whole chapter is consequences of that trade.

    python3 task1_apriori.py --verify
"""
import argparse
from itertools import combinations

# Small enough to check by hand. support threshold 3 unless stated.
BASKETS = [
    {"bread", "milk"},
    {"bread", "diaper", "beer", "egg"},
    {"milk", "diaper", "beer", "cola"},
    {"bread", "milk", "diaper", "beer"},
    {"bread", "milk", "diaper", "cola"},
    {"bread", "milk", "beer"},
    {"diaper", "beer"},
]


def frequent_singletons(baskets, support):
    """Items appearing in at least `support` baskets. Return {item: count}."""
    # 항목별 등장 바구니 수를 센다. 바구니가 set 이라 같은 항목이 한 바구니에
    # 두 번 들어갈 수 없으므로, 이 세기가 곧 '몇 개의 바구니에 나왔나' 가 된다.
    counts = {}
    for basket in baskets:
        for item in basket:
            counts[item] = counts.get(item, 0) + 1

    # support '이상' 만 남긴다 (초과가 아니라 >=).
    return {item: c for item, c in counts.items() if c >= support}


def frequent_pairs(baskets, support):
    """Pairs appearing in at least `support` baskets, using A-Priori.

    Return {frozenset({a, b}): count}.

    The requirement that makes this A-Priori rather than brute force: pass two
    must only ever count pairs **both of whose items were frequent in pass one**.
    If you count every pair and filter afterwards you have written the thing
    A-Priori exists to avoid, and it will pass this harness while failing the
    point. Task 3 measures whether you actually did it.
    """
    # 패스 1. 여기서 살아남은 항목만 패스 2 로 넘긴다.
    singles = frequent_singletons(baskets, support)
    survivors = set(singles)        # dict 를 set() 에 넣으면 '키' 만 모인다

    # 패스 2.
    pair_counts = {}
    for basket in baskets:
        # A-Priori 의 핵심. 조합을 만들기 '전에' 바구니를 줄인다. 살아남지
        # 못한 항목은 여기서 빠지므로, 그 항목이 들어간 쌍은 애초에 만들어지지
        # 않는다 (만들어서 세고 나중에 버리는 것이 아니다).
        items = sorted(basket & survivors)

        # 줄인 목록으로만 2 개 조합을 만든다.
        for pair in combinations(items, 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # 개수가 support 이상인 쌍만 남긴다. 이 거르기는 '개수' 기준이고,
    # 세어 보기 전에는 개수를 알 수 없으므로 피할 수 없는 단계다.
    # 금지된 것은 '항목이 빈발했는지' 를 여기서 따지는 것이다 - 그 정보는
    # 패스 1 에서 이미 알고 있었고, 위에서 바구니를 줄일 때 이미 썼다.
    return {frozenset(p): c for p, c in pair_counts.items() if c >= support}


def association_rules(baskets, support, min_confidence):
    """Rules I -> j from the frequent pairs.

    For a frequent pair {i, j}, confidence of i -> j is
    support({i, j}) / support({i}).

    Return [(antecedent, consequent, confidence, lift), ...] for rules whose
    confidence is at least `min_confidence`, sorted by confidence descending.

    `lift` is confidence divided by the consequent's own support fraction. A
    rule with high confidence and lift near 1 tells you nothing - the consequent
    was common anyway - and §6.1.3 is about why that matters more than it looks.
    """
    singles = frequent_singletons(baskets, support)
    pairs = frequent_pairs(baskets, support)
    n = len(baskets)

    rules = []
    for pair, pair_count in pairs.items():
        # frozenset 은 순서가 없으므로 sorted 로 두 항목을 꺼낸다.
        a, b = sorted(pair)

        # 쌍 하나에서 방향 두 개를 만든다. confidence 는 대칭이 아니므로
        # (a->b) 와 (b->a) 는 서로 다른 주장이고 값도 다를 수 있다.
        for head, tail in ((a, b), (b, a)):
            # 분모는 '출발점' 의 support. 방향마다 달라지는 자리다.
            confidence = pair_count / singles[head]
            if confidence >= min_confidence:
                # 분모는 '도착점' 의 자체 비율. confidence 와 반대쪽이다.
                lift = confidence / (singles[tail] / n)
                rules.append((head, tail, confidence, lift))

    # confidence 내림차순. 원소 하나가 (head, tail, confidence, lift) 이므로
    # 네 번째가 아니라 세 번째 값([2]) 이 confidence 다.
    rules.sort(key=lambda r: r[2], reverse=True)
    return rules


# ------------------------------------------------------------------- harness
def verify():
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<44} {got}"
              + ("" if ok else f"\n{'':>54}want {want}"))
        fails += not ok

    try:
        singles = frequent_singletons(BASKETS, 3)
    except NotImplementedError:
        print("  frequent_singletons is still a stub"); return 1
    # counted by hand: bread 5, milk 5, diaper 5, beer 5, cola 2, egg 1
    check("frequent singletons at support 3",
          dict(sorted(singles.items())),
          {"beer": 5, "bread": 5, "diaper": 5, "milk": 5})

    try:
        pairs = frequent_pairs(BASKETS, 3)
    except NotImplementedError:
        print("  frequent_pairs is still a stub"); return 1
    got = {tuple(sorted(k)): v for k, v in pairs.items()}
    check("frequent pairs at support 3", dict(sorted(got.items())),
          {("beer", "bread"): 3, ("beer", "diaper"): 4, ("beer", "milk"): 3,
           ("bread", "diaper"): 3, ("bread", "milk"): 4, ("diaper", "milk"): 3})

    # cola appears twice, so it must never reach pass two
    check("infrequent items excluded from pairs",
          any("cola" in k or "egg" in k for k in got), False)

    try:
        rules = association_rules(BASKETS, 3, 0.75)
    except NotImplementedError:
        print("  association_rules is still a stub"); return 1
    # diaper -> beer: support 4, diaper 5, confidence 0.8
    hit = [r for r in rules if r[0] == "diaper" and r[1] == "beer"]
    check("diaper -> beer confidence", round(hit[0][2], 3) if hit else None, 0.8)
    check("rules are sorted by confidence",
          all(rules[i][2] >= rules[i + 1][2] for i in range(len(rules) - 1)), True)

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())
