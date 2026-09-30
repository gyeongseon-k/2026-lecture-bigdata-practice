#!/usr/bin/env python3
"""Week 4 · Task 1 — Answer questions about a stream you cannot store.

Textbook §4.3 (sampling), §4.4 (Bloom filter), §4.5 (Flajolet-Martin).

The premise of the whole chapter: the stream is longer than your memory, it
goes past once, and you still have to answer. Every method here trades an exact
answer for a bounded amount of space, and the job is to know exactly what you
traded.

You build three, and the harness checks each against the truth it is
approximating.

    python3 task1_sketches.py --verify
"""
import argparse, hashlib, math, random, statistics


class BloomFilter:
    """Membership, with one-sided error.

    A Bloom filter never says "no" about something you inserted. It sometimes
    says "yes" about something you did not. That asymmetry is the entire design
    and it is why it is useful for "have I seen this before" and useless for
    "is this definitely in the set".

    `m` bits, `k` hash functions.
    """

    def __init__(self, m, k, seed=246):
        self.m = m          # 비트 개수
        self.k = k          # 해시 함수 개수
        self.seed = seed

        # m 개의 비트를 bytearray 에 눌러 담는다. 1 바이트가 8 비트이므로
        # 필요한 바이트 수는 m/8 을 올림한 값이다.
        #
        # bytearray(m) 으로 '1 비트당 1 바이트' 를 써도 동작은 같지만 메모리를
        # 8 배 쓴다. 이 장(章)의 주제가 '공간을 상수로 묶는 것' 이므로 여기서
        # 8 배를 낭비하는 것은 앞뒤가 맞지 않는다. Task 3 은 memory_bits() 가
        # 예산을 넘으면 실격이라 그때는 이 방식이 선택이 아니라 필수다.
        self.bits = bytearray((m + 7) // 8)

    def _indices(self, item):
        """이 아이템이 켜야 할 비트 번호 k 개를 만들어 낸다.

        해시 함수 k 개를 손으로 쓸 수는 없으니, 같은 해시 알고리즘에 서로 다른
        key 를 줘서 k 개의 독립적인 해시로 쓴다. key 가 다르면 결과가 완전히
        달라지므로 사실상 다른 함수다.

        yield 를 쓰는 이유: __contains__ 에서 비트가 하나만 꺼져 있어도 즉시
        답이 나오는데, 그때 나머지 해시는 계산할 필요가 없다. 리스트로 만들면
        무조건 k 개를 다 계산하지만 yield 면 필요한 만큼만 계산한다.
        """
        data = str(item).encode()
        for i in range(self.k):
            digest = hashlib.blake2b(data, digest_size=8,
                                     key=f"{self.seed}-{i}".encode()).digest()
            # 8 바이트를 정수로 바꾸고 m 으로 나눈 나머지 -> 0 <= 번호 < m
            yield int.from_bytes(digest, "big") % self.m

    def add(self, item):
        for idx in self._indices(item):
            # idx >> 3 은 idx // 8  (몇 번째 바이트인가)
            # idx & 7  은 idx % 8   (그 바이트 안에서 몇 번째 비트인가)
            # |= 로 그 비트만 1 로 만든다. 다른 비트는 건드리지 않는다.
            self.bits[idx >> 3] |= 1 << (idx & 7)

    def __contains__(self, item):
        # k 개 비트가 '전부' 1 이어야 yes. 하나라도 0 이면 all() 이 바로 False 를
        # 돌려주고 남은 해시는 계산되지 않는다.
        #
        # R1(거짓 음성 없음)이 여기서 구조적으로 보장된다. add 는 비트를 1 로만
        # 만들고, 0 으로 되돌리는 코드가 이 클래스에 아예 없다. 따라서 넣은
        # 아이템의 k 개 비트는 반드시 1 이고, 반드시 yes 가 나온다.
        return all(self.bits[idx >> 3] >> (idx & 7) & 1
                   for idx in self._indices(item))

    def expected_fp_rate(self, n_inserted):
        """The textbook's predicted false-positive rate after n insertions.

        §4.4.2 derives it. Return the number, do not measure it - the harness
        measures separately and compares the two.

        유도 (다트 던지기로 생각하면 쉽다):
          아이템 n 개를 넣으면 다트를 k*n 번 던지는 것과 같다.
          다트 한 번이 '특정 비트 하나' 를 빗나갈 확률          = 1 - 1/m
          k*n 번 모두 빗나갈 확률                              = (1 - 1/m)^(k*n)
          m 이 클 때 이 값은 e^(-k*n/m) 에 수렴한다.
          따라서 어떤 비트가 1 일 확률                          = 1 - e^(-k*n/m)
          거짓 양성 = 넣지도 않은 아이템의 k 개 비트가 모두 1
                                                              = (1 - e^(-k*n/m))^k
        """
        frac_set = 1.0 - math.exp(-self.k * n_inserted / self.m)
        return frac_set ** self.k


# Flajolet-Martin 의 편향 보정 상수. 해시 하나의 '연속 0 최고 기록' R 은
# 2^R 이 distinct 수보다 평균적으로 조금 크게 나오는 편향이 있고, 원 논문이
# 계산한 보정값이 이 숫자다. 곱해 주지 않으면 추정치가 약 1.29 배 부풀려진다.
_FM_PHI = 0.77351


def _fm_registers(stream, n_buckets, seed):
    """스트림을 한 번만 지나가며 버킷별 '연속 0 최고 기록' 을 모은다.

    반환값은 길이 n_buckets 의 리스트. 스트림 자체는 어디에도 저장하지 않고,
    len() 도 부르지 않는다. 그래서 리스트든 제너레이터든 똑같이 동작한다.
    (Task 2 는 제너레이터를 넘긴다. len() 이나 list() 를 쓰면 거기서 깨진다.)

    교과서(§4.5)는 해시 함수 n 개를 만들어 '모든 아이템에 n 개를 전부' 적용한다.
    그러면 아이템당 해시 계산이 n 번이라, Task 2 의 640 만 짜리 스트림에서는
    4 억 번이 되어 현실적으로 돌지 않는다.

    그래서 여기서는 해시를 아이템당 한 번만 계산하고, 그 값의 하위 비트로
    '이 아이템은 몇 번 추정기 담당인가' 를 고른다 (stochastic averaging).
      - 추정기는 여전히 n_buckets 개이고 서로 독립이다
      - 아이템당 비용은 1/n_buckets 로 줄어든다
      - 대신 추정기 하나가 보는 distinct 수도 1/n_buckets 이 되므로,
        마지막에 n_buckets 를 곱해 되돌린다
    §4.5.3 이 말하는 '그룹으로 묶어 조합한다' 를 한 걸음 더 밀고 나간 형태다.
    """
    key = str(seed).encode()
    p = n_buckets.bit_length() - 1        # 버킷 고르는 데 쓸 비트 수 (64 -> 6)
    mask = n_buckets - 1                  # 하위 p 비트만 남기는 마스크
    regs = [-1] * n_buckets               # -1 = 이 버킷엔 아직 아무것도 안 왔다

    for item in stream:
        h = int.from_bytes(
            hashlib.blake2b(str(item).encode(), digest_size=8, key=key).digest(),
            "big")
        bucket = h & mask                 # 하위 p 비트 -> 담당 추정기
        lane = h >> p                     # 남은 비트 -> 여기서 연속 0 을 센다

        # 뒤쪽 연속 0 의 개수. lane & -lane 은 '가장 낮은 1 비트' 만 남기는
        # 관용구이고, 그 비트의 자리수 - 1 이 곧 뒤쪽 0 의 개수다.
        # 예: lane=...11000 -> lane & -lane = 1000(2진) -> bit_length 4 -> 3 개
        r = (lane & -lane).bit_length() - 1 if lane else 64 - p

        # 최고 기록만 남긴다. 같은 아이템이 100 번 와도 해시가 같아서 기록이
        # 바뀌지 않는다 -> 중복이 자동으로 제거된다. 이게 FM 의 핵심이다.
        if r > regs[bucket]:
            regs[bucket] = r

    return regs


def _fm_combine(regs, rule="median-then-mean", n_groups=4):
    """버킷별 기록을 하나의 추정치로 합친다.

    규칙 네 가지를 모두 구현해 두었다. 어느 것이 좋은지는 논증이 아니라 측정으로
    정했고, 그 표가 observation.md 에 들어간다. 요약하면:

      mean               운 좋은 버킷 하나가 전체를 지배한다. 2^R 은 지수값이라
                         한 버킷이 R 을 3 더 찍으면 그 버킷 혼자 8 배를 끌어올린다.
                         24 개 시나리오 중 6 개만 2 배 이내. 최대 23.70x. 쓸 수 없다.
      median             이상치에 강해서 24/24 통과. 다만 최소가 0.53x 로
                         실패선(0.5x) 까지 여유가 1.06 배뿐이다. 통과하지만 위험하다.
      mean-then-median   §4.5.3 을 문자 그대로: 그룹 안 평균 -> 그룹 간 중앙값.
                         의외로 나쁘다 (그룹 4 개면 10/24). 이유는 분명하다.
                         이상치는 '그룹 안' 에서 생기는데, 먼저 평균을 내면 그
                         그룹의 값이 오염된다. 그룹 수가 적으면 오염된 그룹이
                         과반이 되어 중앙값으로도 걸러지지 않는다.
      median-then-mean   순서를 뒤집는다. 그룹 안에서 중앙값으로 이상치를
                         '생긴 자리에서' 제거하고, 그룹 간 평균으로 해상도를
                         되찾는다. 24/24 통과에 최소 0.63x, 실패선까지 1.27 배
                         여유. 이것을 기본값으로 쓴다.

    §4.5.3 의 '두 번 조합한다' 는 아이디어는 맞았고, 두 연산의 순서가 중요했다.
    """
    # 기록 R 을 추정치 2^R 로 바꾼다. 아무것도 안 온 버킷은 0 으로 둔다.
    vals = [2.0 ** r if r >= 0 else 0.0 for r in regs]
    n = len(vals)

    if rule == "mean":
        core = sum(vals) / n
    elif rule == "median":
        core = statistics.median(vals)
    elif rule in ("mean-then-median", "median-then-mean"):
        size = max(1, n // n_groups)
        groups = [vals[i:i + size] for i in range(0, n, size)]
        if rule == "mean-then-median":
            core = statistics.median([sum(g) / len(g) for g in groups])
        else:
            inner = [statistics.median(g) for g in groups]
            core = sum(inner) / len(inner)
    else:
        raise ValueError(f"unknown rule: {rule}")

    # 버킷 하나가 보는 distinct 는 전체의 1/n 이므로 n 을 곱해 되돌리고,
    # 2^R 의 편향을 보정 상수로 깎는다.
    return _FM_PHI * n * core


def flajolet_martin(stream, n_hashes=64, seed=246, rule="median-then-mean"):
    """Estimate how many DISTINCT items went past, in almost no memory.

    §4.5. Hash each item, count trailing zeros in the hash, keep the maximum.
    A maximum of R suggests about 2^R distinct items, because seeing R trailing
    zeros is a 1-in-2^R event.

    One hash gives an estimate with enormous variance, so you use many and
    combine them. How you combine them matters a great deal:

      * averaging 2^R directly is dominated by whichever hash got lucky - the
        values are exponential, so one outlier swamps the rest
      * the median is robust but can only ever be a power of two
      * §4.5.3 suggests grouping, and combining twice

    The harness accepts anything **within a factor of two** of the truth. That is
    not a generous tolerance, it is an honest one: this method really is that
    crude, and HyperLogLog exists because of it. Getting inside a factor of two
    reliably is the requirement; getting closer than that is not expected here.

    Return your estimate as a float.

    rule 로 조합 방식을 고를 수 있게 해 두었다. 네 가지를 모두 구현해서 24 개
    시나리오로 비교했고, 기본값 "median-then-mean" 을 고른 근거는 _fm_combine 의
    docstring 과 observation.md 에 있다.
    """
    # 추정기 개수를 2 의 거듭제곱으로 맞춘다. 하위 비트로 버킷을 고르기 때문에
    # 2 의 거듭제곱이 아니면 버킷들이 균등하게 안 채워진다. (64 -> 64, 100 -> 64)
    n_buckets = 1 << max(0, n_hashes.bit_length() - 1)
    regs = _fm_registers(stream, n_buckets, seed)
    return _fm_combine(regs, rule)


def reservoir_sample(stream, k, seed=246):
    """Keep k items uniformly at random from a stream of unknown length.

    §4.3. Every item that went past must end up with the same probability k/n
    of being in your sample, and you only ever hold k of them.

    Return a list of k items (or fewer if the stream was shorter).

    알고리즘 R (§4.3):
      처음 k 개는 무조건 담는다.
      그 뒤 i 번째(0 부터 셈) 아이템은 k/(i+1) 확률로 받아들이고,
      받아들이면 이미 앉아 있는 k 개 중 하나를 무작위로 밀어낸다.

    seed 를 반드시 쓴다. 채점기가 seed=0..3999 로 4,000 번 다르게 부르면서
    균등성을 확인하기 때문에, 시드를 무시하면 4,000 번 모두 같은 답이 나와
    균등성 검사에서 떨어진다.
    """
    rng = random.Random(seed)
    reservoir = []

    # enumerate 로 '지금까지 몇 개 봤는지' 만 센다. 스트림 전체 길이는 묻지 않는다.
    # len(stream) 을 쓰면 제너레이터에서 깨지고, 애초에 스트림 모델에서는
    # 알 수 없는 값이다.
    for i, item in enumerate(stream):
        if i < k:
            # 아직 자리가 남았다. 그냥 앉힌다.
            reservoir.append(item)
        else:
            # ---- 스트림 길이를 몰라도 되는 이유가 전부 이 두 줄에 있다 ----
            # 확률의 분모가 '전체 길이 n' 이 아니라 '지금까지 본 개수 i+1' 이다.
            # 그래서 스트림이 여기서 끝나든 100 만 개가 더 오든,
            # 지금 이 순간의 표본은 '지금까지 본 것들에 대해' 이미 균등하다.
            # 언제 멈춰도 맞는다 -> 끝을 알 필요가 없다.
            j = rng.randrange(i + 1)          # 0 .. i 중 하나
            if j < k:                         # k/(i+1) 확률로 채택
                reservoir[j] = item           # 앉아 있던 j 번을 밀어낸다

    return reservoir


# ------------------------------------------------------------------- harness
def verify():
    fails = 0
    rng = random.Random(246)

    def check(label, ok, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<46} {detail}")
        fails += not ok

    # --- Bloom: no false negatives, ever
    try:
        bf = BloomFilter(m=8192, k=5)
    except NotImplementedError:
        print("  BloomFilter is still a stub"); return 1
    inserted = [f"item-{i}" for i in range(800)]
    for x in inserted:
        bf.add(x)
    check("no false negatives", all(x in bf for x in inserted))

    absent = [f"other-{i}" for i in range(20_000)]
    fp = sum(1 for x in absent if x in bf) / len(absent)
    predicted = bf.expected_fp_rate(len(inserted))
    close = abs(fp - predicted) < max(0.02, predicted * 0.5)
    check("measured false-positive rate matches theory", close,
          f"measured {fp:.3%}, predicted {predicted:.3%}")

    # --- Flajolet-Martin: a factor of two is what this method gives you
    try:
        distinct = 20_000
        stream = [f"k{rng.randrange(distinct)}" for _ in range(120_000)]
        est = flajolet_martin(stream)
    except NotImplementedError:
        print("  flajolet_martin is still a stub"); return 1
    true_distinct = len(set(stream))
    ratio = est / true_distinct
    check("distinct estimate within a factor of 2", 0.5 <= ratio <= 2.0,
          f"estimated {est:,.0f}, true {true_distinct:,} ({ratio:.2f}x)")

    # --- Reservoir: uniform over many trials
    try:
        counts = [0] * 20
        trials = 4000
        for t in range(trials):
            s = reservoir_sample(range(20), 5, seed=t)
            for i in s:
                counts[i] += 1
    except NotImplementedError:
        print("  reservoir_sample is still a stub"); return 1
    expected = trials * 5 / 20
    spread = (max(counts) - min(counts)) / expected
    check("reservoir is uniform across items", spread < 0.15,
          f"spread {spread:.1%} around {expected:.0f}")

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())
