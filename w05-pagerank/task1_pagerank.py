#!/usr/bin/env python3
"""Week 5 · Task 1 — PageRank, and the two ways it breaks.

Textbook §5.1 (PageRank), §5.1.3 (dead ends), §5.1.4 (spider traps).

PageRank is a random surfer following links forever, and the rank of a page is
how often the surfer is there. As one line of linear algebra it is
`r = M r`, and as written it does not work on the actual web.

Two structures break it, and the fix for both is the same one line. Build the
broken version first so you can see each failure, then fix it.

    python3 task1_pagerank.py --verify
"""
import argparse

# A: -> B, C      B: -> C      C: -> A
SIMPLE = {"A": ["B", "C"], "B": ["C"], "C": ["A"]}

# C has no out-links at all. Rank leaks out of the graph and everything -> 0.
DEAD_END = {"A": ["B", "C"], "B": ["C"], "C": []}

# C and D only point at each other. They absorb everything.
SPIDER_TRAP = {"A": ["B"], "B": ["C"], "C": ["D"], "D": ["C"]}


def pagerank(graph, beta=0.85, iterations=100, tol=1e-10):
    """Rank every node. Return {node: rank}, summing to 1.

    `beta` is the probability the surfer follows a link. With probability
    1 - beta they teleport to a node chosen uniformly.

    You have to handle both of these, and the textbook handles them the same way:

      dead ends    a node with no out-links. Where does its rank go, and where
                   should it go instead?
      spider traps a group of nodes that only link to each other. Without
                   teleporting, they end up with all of it

    Stop early when the ranks stop moving - `tol` is the L1 change below which
    you should call it converged. Return the ranks, and set `pagerank.iterations`
    to how many you actually used, because Task 2 measures that.

    ------------------------------------------------------------------------
    한 번의 반복에서 노드 i 의 새 점수는 세 조각의 합이다.

        새 점수[i] = beta * (링크로 받은 몫[i])        <- 링크를 따라간 서퍼
                   + beta * (막다른 길에 고인 점수) / n   <- 갇힌 점수를 흩뿌림
                   + (1 - beta) / n                    <- 텔레포트

    뒤의 두 조각은 '어떤 양을 n 개 노드에 1/n 씩 균등하게 흩뿌린다' 는 점에서
    완전히 같은 동작이다. 흩뿌리는 이유만 다르다. 그래서 하나의 메커니즘이
    막다른 길과 거미 덫이라는 서로 달라 보이는 두 실패를 동시에 고친다.

    막다른 길 처리 방식은 '모든 노드에 균등 분배' 를 택했다 (교과서 §5.1.3).
    다른 선택지도 방어 가능하지만 이것을 고른 이유:

      - Task 3 의 기준 구현인 DenseMatrix 가 같은 방식을 쓴다
        (task3_sparse.py 의 'dead end: spread it everywhere'). Task 3 은 그
        답과 1e-9 안에서 일치해야 하므로 여기서 다른 방식을 쓰면 주차 안에
        알고리즘이 두 개가 된다.
      - 합이 1 로 보존되는 것이 식에서 증명된다. 고인 점수를 S 라 하면
        beta * (1 - S)  +  beta * S  +  (1 - beta)  =  beta + (1 - beta)  =  1.
      - 텔레포트와 같은 동작이므로 '같은 수정이 두 실패를 고친다' 는 설명이
        한 문장으로 성립한다.

    대가: 막다른 길 자신도 1/n 몫을 돌려받아 점수가 약간 부풀려진다.
    교과서의 '막다른 길을 재귀적으로 제거하는' 방식은 이를 피하지만 훨씬
    복잡하고, DenseMatrix 와 숫자가 달라진다.
    """
    nodes = list(graph)
    n = len(nodes)
    if n == 0:                      # 빈 그래프. 나눗셈 전에 빠져나간다.
        pagerank.iterations = 0
        return {}

    # 나가는 링크가 없는 노드. 반복마다 다시 찾을 필요가 없으니 한 번만 모은다.
    dead_ends = [v for v in nodes if not graph[v]]

    # 처음에는 모두에게 똑같이 1/n. 서퍼가 어디 있을지 전혀 모르는 상태다.
    r = {v: 1.0 / n for v in nodes}

    step = 0                        # 반복을 한 번도 못 돌 경우를 위한 초기값
    for step in range(1, iterations + 1):
        # --- 흩뿌릴 양을 먼저 계산한다 (모든 노드에 똑같이 들어가는 몫) ---
        # leak: 지금 막다른 길에 고여 있는 점수의 합. 고치지 않으면 증발할 몫이다.
        leak = sum(r[v] for v in dead_ends)
        # base: 노드 하나가 링크와 무관하게 받는 양.
        #   beta * leak / n  = 갇힌 점수를 균등 분배 (막다른 길 처리)
        #   (1 - beta) / n   = 텔레포트
        base = beta * leak / n + (1.0 - beta) / n

        # 모든 노드를 base 로 시작해 두고, 여기에 링크로 받은 몫을 더해 간다.
        nr = {v: base for v in nodes}

        # --- 링크를 따라 점수를 나눠준다 ---
        for v in nodes:
            outs = graph[v]
            if outs:
                # 내 점수를 나가는 링크 개수로 쪼개서 각 상대에게 준다.
                # beta 를 여기서 한 번만 곱해 두면 상대마다 곱할 필요가 없다.
                share = beta * r[v] / len(outs)
                for w in outs:
                    nr[w] += share
            # outs 가 비어 있으면(막다른 길) 아무에게도 주지 않는다.
            # 그 몫은 위에서 leak 으로 이미 계산해 base 에 녹여 두었다.

        # --- 수렴 판정: L1 변화량 ---
        # 모든 노드의 '변화 절댓값' 을 더한 값. 이것이 tol 보다 작으면
        # 점수가 더 이상 움직이지 않는다고 보고 멈춘다 (R4).
        delta = sum(abs(nr[v] - r[v]) for v in nodes)
        r = nr
        if delta < tol:
            break

    # R5. Task 2 가 getattr(pagerank, "iterations") 로 읽어 가는 값이다.
    # 함수 객체에 붙여 두는 방식이라 어색하지만 과제가 지정한 인터페이스다.
    pagerank.iterations = step
    return r


def pagerank_no_teleport(graph, iterations=100):
    """The broken version: beta = 1, no teleporting. Build this too.

    It exists so you can watch both failures happen rather than take them on
    trust. The harness checks that it really does fail.

    ------------------------------------------------------------------------
    pagerank 과 다른 곳은 딱 하나다: 위 식에서 '균등하게 흩뿌리는' 두 조각을
    통째로 뺐다. 남은 것은 링크로 받은 몫뿐이다.

        새 점수[i] = (링크로 받은 몫[i])

    주의할 함정: 이름이 no_teleport 라서 beta = 1 로만 바꾸면 될 것 같지만,
    그러면 R6 을 통과하지 못한다. beta = 1 을 넣으면 텔레포트 항 (1-beta)/n 은
    0 이 되어 사라지지만 막다른 길 구제 항 beta * leak / n 은 살아남고,
    그 항만으로도 합이 1 로 보존되어 '막다른 길이 그래프를 말려 버린다' 는
    검사(합 < 0.5)에서 떨어진다. 그래서 두 조각을 모두 제거했다.

    실제로 어떻게 실패하는가:

      DEAD_END     합이 1 -> 0.667 -> 0.167 -> 0 으로 줄어든다. 막다른 길은
                   나눠줄 상대가 없어서 들고 있던 점수가 증발하고, 다음
                   반복에서 그만큼 줄어든 채로 또 증발한다. 점수가 사라진다.
      SPIDER_TRAP  합은 계속 1 이다 (C 와 D 둘 다 나가는 링크가 있으니 새는
                   곳이 없다). 그런데 서퍼가 C <-> D 를 왕복하며 빠져나오지
                   못해 두 번째 반복에서 C + D = 1.0 이 된다. 점수가 몰린다.

    증상이 전혀 달라 보이지만(사라진다 vs 몰린다) 고치는 방법은 같다.
    """
    nodes = list(graph)
    n = len(nodes)
    if n == 0:
        return {}

    r = {v: 1.0 / n for v in nodes}

    for _ in range(iterations):
        # pagerank 와 달리 base 가 없다. 0 에서 시작하므로, 링크로 아무것도
        # 받지 못한 노드는 0 이 되고 막다른 길이 들고 있던 점수는 사라진다.
        nr = {v: 0.0 for v in nodes}
        for v in nodes:
            outs = graph[v]
            if outs:
                share = r[v] / len(outs)      # beta 를 곱하지 않는다 (beta = 1)
                for w in outs:
                    nr[w] += share
        r = nr
        # 조기 종료도 넣지 않는다. 망가진 모습을 끝까지 보는 것이 목적이다.

    return r


# ------------------------------------------------------------------- harness
def verify():
    fails = 0

    def check(label, ok, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<48} {detail}")
        fails += not ok

    try:
        r = pagerank(SIMPLE)
    except NotImplementedError:
        print("  pagerank is still a stub"); return 1

    check("ranks sum to 1", abs(sum(r.values()) - 1) < 1e-6, f"{sum(r.values()):.6f}")
    check("every node ranked", set(r) == set(SIMPLE), sorted(r))
    # C is pointed at by both A and B, so it has to come first.
    check("C outranks A and B", r["C"] > r["A"] and r["C"] > r["B"],
          {k: round(v, 4) for k, v in sorted(r.items())})

    try:
        broken = pagerank_no_teleport(DEAD_END)
    except NotImplementedError:
        print("  pagerank_no_teleport is still a stub"); return 1
    check("without the fix, a dead end drains the graph",
          sum(broken.values()) < 0.5, f"total rank {sum(broken.values()):.4f}")

    fixed = pagerank(DEAD_END)
    check("with the fix, rank is conserved",
          abs(sum(fixed.values()) - 1) < 1e-6, f"{sum(fixed.values()):.6f}")

    trapped = pagerank_no_teleport(SPIDER_TRAP)
    check("without the fix, a spider trap takes everything",
          trapped["C"] + trapped["D"] > 0.95,
          f"C+D = {trapped['C'] + trapped['D']:.4f}")

    untrapped = pagerank(SPIDER_TRAP)
    check("with the fix, A and B keep some rank",
          untrapped["A"] > 0.01 and untrapped["B"] > 0.01,
          {k: round(v, 4) for k, v in sorted(untrapped.items())})

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())
