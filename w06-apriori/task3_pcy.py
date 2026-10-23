#!/usr/bin/env python3
"""Week 6 · Task 3 — Make pass two fit.

Textbook §6.3 (PCY), §6.3.2 - §6.3.4.

`PlainApriori` does what Task 1 asked: use pass one to drop infrequent items,
then count every pair of surviving items. That is already much better than
counting all pairs. It is still not enough, because the surviving items are the
common ones, and the common ones appear together constantly.

The harness measures **the peak number of pair counters you held**, because
that is the thing that decides whether the algorithm runs at all. §6.3 is about
spending pass one's spare memory to shrink it.

    python3 bench.py
    python3 bench.py --yours

Correctness first: you must find exactly the same frequent pairs. Finding fewer
is not an optimisation.
"""
from collections import Counter
from itertools import combinations


class PlainApriori:
    """Pass one drops infrequent items. Pass two counts every surviving pair."""

    def __init__(self, support):
        self.support = support
        self.peak_counters = 0

    def run(self, baskets):
        counts = Counter()
        for basket in baskets:
            counts.update(basket)
        frequent = {i for i, c in counts.items() if c >= self.support}

        pair_counts = Counter()
        for basket in baskets:
            items = sorted(basket & frequent)
            for pair in combinations(items, 2):
                pair_counts[pair] += 1
            self.peak_counters = max(self.peak_counters, len(pair_counts))

        return {frozenset(p): c for p, c in pair_counts.items()
                if c >= self.support}


class YourAlgorithm:
    """Your frequent-pair finder.

        __init__(support)
        run(baskets) -> {frozenset({a, b}): count}
        .peak_counters -> the most pair counters you ever held at once

    Same pairs as the baseline. Fewer counters.

    Pass one only needs one integer per item, and there are not many items. The
    rest of your memory is sitting idle while you do it. §6.3 spends it: hash
    every pair you see in pass one into a fixed array of buckets, and count the
    buckets rather than the pairs.

    A bucket whose total is below the support threshold cannot contain a
    frequent pair. In pass two you skip every pair landing in such a bucket -
    and the bucket array collapses to a bitmap, one bit each, before you need
    the memory for counters.

    Two things to be careful of:

      * a bucket being frequent does not make its pairs frequent. It is a
        filter, not an answer
      * `peak_counters` is on your honour. Count the pair counters you hold at
        the same time. The bitmap is not a pair counter, but if you keep the
        full bucket counts alive into pass two, that is not free either -
        observation.md asks about it
    """

    # 버킷 개수. 쌍 등장 총량이 342 만 회이므로 버킷당 평균 3.4 로, support 50
    # 보다 훨씬 작다. 버킷을 적게 잡으면 서로 다른 쌍이 한 버킷에 뭉쳐 합계가
    # 부풀려지고, 비트맵이 전부 1 이 되어 필터가 아무 일도 하지 않는다.
    N_BUCKETS = 1_000_003      # 소수

    # 쌍 (i, j) 를 i*HASH_MULT + j 로 묶은 뒤 N_BUCKETS 로 나눈 나머지를 쓴다.
    #
    # 조건은 '둘 다 소수' 가 아니라 **HASH_MULT % N_BUCKETS 가 커야** 한다.
    # 나머지가 작으면 해시가 선형 결합으로 붕괴한다. 실제로 겪은 예:
    #   HASH_MULT = 2,000,003 일 때 2,000,003 % 1,000,003 = 1,000,000 ≡ -3
    #   -> bucket ≡ -3x + y 가 되어, x,y 가 0~1999 이면 값이 약 8,000 가지뿐
    #   -> 버킷 1,000,003 개 중 7,800 개(0.78%)만 쓰이고 평균 적재가 439 로
    #      support 50 을 훌쩍 넘어 비트맵이 80% 가 1 이 됐다 (절감 2.5%)
    # 1,299,709 % 1,000,003 = 299,706 으로 크다. 같은 데이터에서 버킷 64% 를
    # 쓰고 평균 적재 5.3 이 되어 절감이 98.8% 로 올라갔다.
    HASH_MULT = 1_299_709      # 소수. N_BUCKETS 로 나눈 나머지가 299,706 (크다)

    def __init__(self, support):
        self.support = support
        self.peak_counters = 0          # PlainApriori 와 같은 초기값

    def run(self, baskets):
        support = self.support
        nb = self.N_BUCKETS
        mult = self.HASH_MULT

        # ───────────────────────── 패스 1 ─────────────────────────
        # 항목 개수를 세면서, 동시에 '보이는 모든 쌍' 을 버킷에 던진다.
        # 이 시점에는 어떤 항목이 빈발한지 아직 모르므로 쌍을 미리 거를 수 없다.
        item_counts = {}
        index = {}                      # 항목 -> 번호. 해시에 쓸 정수를 만든다
        bucket_counts = [0] * nb

        for basket in baskets:
            items = sorted(basket)      # 순서를 고정해야 같은 쌍이 같은 버킷에 간다
            for item in items:
                item_counts[item] = item_counts.get(item, 0) + 1
                if item not in index:
                    # 처음 보는 항목에 번호를 발급한다. 내장 hash() 를 쓰지 않는
                    # 이유: 문자열 hash() 는 실행마다 값이 바뀌어(해시 무작위화)
                    # 같은 데이터에서도 결과가 달라진다. 번호는 바구니 순서가
                    # 고정이므로 항상 같고, 항목이 정수든 문자열이든 동작한다.
                    index[item] = len(index)

            # 항목 이름을 번호로 한 번만 바꿔 둔다. 쌍마다 딕셔너리를 두 번씩
            # 찾는 것보다 바구니당 한 번씩 찾는 쪽이 훨씬 적다.
            codes = [index[i] for i in items]
            for x, y in combinations(codes, 2):
                bucket_counts[(x * mult + y) % nb] += 1

        # ──────────────────── 비트맵으로 압축 ────────────────────
        survivors = {i for i, c in item_counts.items() if c >= support}

        # 버킷당 1 비트. (nb + 7) // 8 바이트면 nb 개 비트가 들어간다.
        # bool 리스트([False] * nb) 로 하면 칸마다 포인터 8 바이트가 들어가
        # 8 MB 가 되지만, 이렇게 비트로 눌러 담으면 125 KB 다 (64 배 차이).
        bitmap = bytearray((nb + 7) // 8)
        for b in range(nb):
            if bucket_counts[b] >= support:
                # b >> 3 은 b // 8  (몇 번째 바이트)
                # b & 7  은 b % 8   (그 바이트 안에서 몇 번째 비트)
                bitmap[b >> 3] |= 1 << (b & 7)

        # 합계 배열을 버린다. 이 이름표가 마지막 참조이므로 CPython 이 바로
        # 메모리를 회수하고, 패스 2 는 비트맵(125 KB)만 들고 간다.
        del bucket_counts

        # ───────────────────────── 패스 2 ─────────────────────────
        # 쌍을 세는 조건은 두 개이고 둘 다 만족해야 한다.
        #   ① 두 항목이 모두 패스 1 을 통과했다   -> items 를 만들 때 걸러진다
        #   ② 그 쌍의 버킷 비트가 1 이다          -> if 문에서 걸러진다
        pair_counts = {}
        for basket in baskets:
            items = sorted(basket & survivors)          # 조건 ①
            codes = [index[i] for i in items]
            for (a, x), (b, y) in combinations(zip(items, codes), 2):
                h = (x * mult + y) % nb
                if bitmap[h >> 3] >> (h & 7) & 1:       # 조건 ②
                    pair = (a, b)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
            # PlainApriori:43 과 같은 기준 - 딕셔너리에 든 '서로 다른 쌍 키' 의
            # 개수를 바구니마다 재서 최댓값을 남긴다. 비트맵이 걸러낸 쌍은
            # 키를 만들지 않으므로 여기서 자동으로 빠진다.
            self.peak_counters = max(self.peak_counters, len(pair_counts))

        # 통과한 쌍도 실제 개수로 다시 판정한다. 버킷 때문에 운 좋게 통과한
        # 쌍은 여기서 떨어지므로, 답의 기준은 기준선과 완전히 같다.
        return {frozenset(p): c for p, c in pair_counts.items() if c >= support}
