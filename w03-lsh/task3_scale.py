#!/usr/bin/env python3
"""Week 3 · Task 3 — Find the same pairs without comparing everything.

Textbook §3.4.

`BruteForce` compares every pair. On 3,000 documents that is 4.5 million
comparisons and it is completely correct. On 3 million documents it is 4.5
trillion and it is completely useless.

Beat it. Find the same near-duplicate pairs while making far fewer comparisons.

    python3 bench.py
    python3 bench.py --yours

The harness counts every call you make to `similarity()`. That is your score.
It also checks **recall** - which of the truly similar pairs you found. Skipping
comparisons is easy; skipping comparisons without losing the pairs is the task.
"""


class BruteForce:
    """Correct, and quadratic."""

    def __init__(self, threshold):
        self.threshold = threshold

    def find(self, docs, similarity):
        """docs is [set_of_shingles, ...]. Return {(i, j), ...} with i < j."""
        out = set()
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                if similarity(docs[i], docs[j]) >= self.threshold:
                    out.add((i, j))
        return out


class YourFinder:
    """Your near-duplicate finder.

        __init__(threshold)
        find(docs, similarity) -> {(i, j), ...}

    `similarity(a, b)` is the only way to compare two documents, and every call
    is counted. Everything else - signatures, banding, bucketing - is free, in
    the sense that the harness does not charge you for it. That is deliberate:
    it is also roughly true at scale, where the comparison is the expensive
    part and the hashing is linear.

    Two knobs decide everything:

        the number of hashes in a signature
        how many bands you split it into

    §3.4.2 gives you the relationship between those and the probability that a
    pair at similarity s becomes a candidate. It is an S-curve, and where its
    step sits is something you choose. Choose it on purpose and be able to say
    why in observation.md - a threshold of 0.8 does not mean bands should be
    anything in particular until you have done the arithmetic.

    You may reuse your Task 1 code.
    """

    # ------------------------------------------------------------------
    # n 과 b 를 고른 근거 (§3.4.2)
    #
    # 유사도 s 인 쌍이 후보가 될 확률:   P(s) = 1 - (1 - s^r)^b,  r = n/b
    # 그 S-커브의 계단 위치:             약 (1/b)^(1/r)
    #
    # n=120, b=40 -> r=3 이므로 계단은 (1/40)^(1/3) = 0.292.
    # 임계값 0.6 보다 '아래' 에 일부러 놓았다. 이유:
    #
    #   - 누락(진짜 쌍을 후보에서 빼먹는 것)은 되돌릴 수 없다. 그 쌍은 영영 못 찾고
    #     recall 이 바로 깎인다. 반면 오탐(안 비슷한 쌍을 후보에 넣는 것)은
    #     뒤에서 similarity() 로 실제 값을 재서 걸러내므로 답이 틀리지 않는다.
    #     비교 횟수만 조금 더 쓸 뿐이다.
    #   - 그래서 둘 중에는 오탐 쪽으로 기울이는 게 맞다. 계단을 임계값 아래로.
    #
    # 이 선택의 실제 값:
    #   P(0.6)  = 1 - (1 - 0.6^3)^40 = 0.99994   <- 경계선 쌍도 거의 다 잡는다
    #   P(0.006)= 1 - (1 - 0.006^3)^40 = 8.6e-6  <- 무관한 쌍은 사실상 안 걸린다
    #     (이 데이터의 무관한 쌍은 어휘 5000 개에서 샤글 60 개를 뽑은 것이라
    #      평균 유사도가 0.006 쯤이다. 220 만 쌍 * 8.6e-6 = 20 개 남짓만 헛걸린다.)
    #
    # 반대로 계단을 임계값 '위' 로 올리면 (예: n=60, b=10 -> r=6, 계단 0.681)
    #   P(0.6) = 0.706 까지 떨어진다. 진짜 쌍 10 개 중 3 개를 놓친다는 뜻이고
    #   R3(recall 90%) 에서 바로 탈락한다.
    #
    # r 을 줄이는 쪽이 n 을 늘리는 쪽보다 효율적이라는 점도 계산으로 나왔다.
    #   n=128, b=32 (r=4) -> P(0.6) = 0.988
    #   n=120, b=40 (r=3) -> P(0.6) = 0.99994   같은 해시 개수로 훨씬 낫다
    # ------------------------------------------------------------------
    N_HASHES = 120
    BANDS = 40

    # 해시를 (a*x + b) % P 꼴로 만들 때 쓰는 큰 소수. 2^31 - 1 (메르센 소수).
    PRIME = (1 << 31) - 1

    def __init__(self, threshold):
        self.threshold = threshold

        # 해시 함수 120 개를 만든다. 진짜 해시 함수를 120 개 손으로 쓸 수는 없으니
        # 교과서 방식대로 (a*x + b) % P 의 계수 (a, b) 를 120 쌍 뽑아서 쓴다.
        # 시드를 고정하는 이유: 실행할 때마다 결과가 달라지면 점수를 비교할 수 없다.
        import random
        rng = random.Random(20260920)
        self.params = [(rng.randrange(1, self.PRIME), rng.randrange(0, self.PRIME))
                       for _ in range(self.N_HASHES)]

    # ---------------------------------------------------------------- 시그니처
    def _signatures(self, docs):
        """문서(샤글 집합) 들을 길이 120 짜리 민해시 시그니처로 줄인다.

        Task 1 과 같은 '행을 바깥 루프로 한 번만' 방식이다. 다만 여기서는 행이
        0..n_rows 로 정해져 있지 않고 임의의 샤글 값이라, 먼저 역색인을 만들어
        '실제로 등장한 행' 만 돌게 한다. 등장하지도 않은 샤글까지 전부 도는 것보다
        훨씬 빠르고, 행을 한 번만 훑는다는 성질은 그대로다.
        """
        n, prime = self.N_HASHES, self.PRIME

        # 1) 역색인: 샤글 -> 그 샤글을 가진 문서 번호 목록.
        #    문서를 딱 한 번 훑으면서 만든다.
        rows = {}
        for c, doc in enumerate(docs):
            for shingle in doc:
                rows.setdefault(shingle, []).append(c)

        # 2) 시그니처 표를 무한대로 초기화. 앞으로 min 으로 계속 줄여 나간다.
        sig = [[float("inf")] * n for _ in docs]

        # 3) 행(샤글) 하나씩 돌면서, 그 행을 가진 모든 문서를 한꺼번에 갱신한다.
        for shingle, members in rows.items():
            # 샤글이 정수가 아니어도 되도록 hash() 로 한 번 정수화한다.
            # & 로 부호를 떼는 이유: 음수 해시가 나오면 % 결과가 지저분해진다.
            x = hash(shingle) & 0x7FFFFFFF

            # 이 행의 해시값 120 개. 행마다 한 번만 계산한다.
            # (문서마다 다시 계산하면 같은 값을 수십 번 중복으로 구하게 된다.)
            hv = [(a * x + b) % prime for a, b in self.params]

            # 이 행을 가진 문서들의 시그니처를 원소별 min 으로 갱신.
            for c in members:
                cur = sig[c]
                sig[c] = [p if p < q else q for p, q in zip(cur, hv)]

        return sig

    # ------------------------------------------------------------------- 밴딩
    def _candidates(self, sig):
        """시그니처를 40 개 밴드로 잘라, 한 밴드라도 완전히 같은 쌍만 뽑는다.

        Task 1 의 lsh_candidates 와 같은 논리다. 여기서는 120 / 40 이 딱 떨어져서
        나머지 행 문제가 생기지 않는다 (Task 1 의 R5 결정은 '남는 행은 마지막
        밴드에 합친다' 였고, 그 규칙을 그대로 써도 결과는 동일하다).
        """
        n, bands = self.N_HASHES, self.BANDS
        rows_per_band = n // bands

        pairs = set()
        for b in range(bands):
            start = b * rows_per_band
            end = n if b == bands - 1 else (b + 1) * rows_per_band

            # 같은 조각을 가진 문서들을 같은 통에 모은다.
            # list 는 dict 의 key 가 될 수 없으므로 tuple 로 바꾼다.
            buckets = {}
            for i, s in enumerate(sig):
                buckets.setdefault(tuple(s[start:end]), []).append(i)

            # 한 통에 2 개 이상 있으면 그 안의 모든 쌍이 후보.
            for members in buckets.values():
                if len(members) < 2:
                    continue
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        pairs.add((members[x], members[y]))

        # set 이라 여러 밴드에서 중복으로 걸린 쌍도 한 번만 남는다.
        # -> 같은 쌍을 두 번 비교하는 낭비가 자동으로 없어진다.
        return pairs

    # ------------------------------------------------------------------- 본체
    def find(self, docs, similarity):
        sig = self._signatures(docs)
        candidates = self._candidates(sig)

        # 후보로 살아남은 쌍에 대해서만 진짜 유사도를 잰다.
        # similarity() 호출은 여기서만 일어나고, 이 횟수가 곧 점수다.
        # 이 마지막 검증 덕분에 오탐은 답을 오염시키지 못한다 - 걸러질 뿐이다.
        out = set()
        for i, j in candidates:
            if similarity(docs[i], docs[j]) >= self.threshold:
                out.add((i, j) if i < j else (j, i))
        return out
