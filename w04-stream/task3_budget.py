#!/usr/bin/env python3
"""Week 4 · Task 3 — Same memory, fewer mistakes.

Textbook §4.4 (Bloom filters), §4.5 (counting distinct).

`NaiveFilter` is a membership filter in a fixed number of bits. It works. It
also makes far more mistakes than it has to with the memory it was given, and
it does so for a reason you can find by reading §4.4.2 and doing one derivative.

You get **exactly the same number of bits**. Make fewer mistakes.

    python3 bench.py
    python3 bench.py --yours

The rule that makes this interesting: a false negative is not allowed. Ever.
The whole point of this structure is that "no" means no. A filter that gets a
better score by occasionally forgetting something it was given has not improved
anything, it has broken the contract.
"""
import hashlib, math


class NaiveFilter:
    """One hash function, and the bits it was given."""

    def __init__(self, n_bits, seed=246):
        self.n_bits = n_bits
        self.seed = seed
        self.bits = bytearray(n_bits)

    def _index(self, item):
        d = hashlib.blake2b(str(item).encode(), digest_size=8,
                            key=str(self.seed).encode()).digest()
        return int.from_bytes(d, "big") % self.n_bits

    def add(self, item):
        self.bits[self._index(item)] = 1

    def __contains__(self, item):
        return bool(self.bits[self._index(item)])

    def memory_bits(self):
        return self.n_bits


class YourFilter:
    """Your filter.

        __init__(n_bits, seed=246)
        add(item)
        item in filter  ->  bool
        memory_bits()   ->  how many bits you are using

    `memory_bits()` must not exceed the `n_bits` you were given. The harness
    checks. Counting only some of your memory is not an optimisation.

    §4.4.2 gives the false-positive rate of a filter with m bits, k hashes and
    n items inserted. There is a k that minimises it, and it depends on m/n.
    The harness tells you n before you start, so you have no excuse for guessing.

    Then there is a second question, which is worth more: the harness inserts
    a **known** number of items, but a real stream does not tell you n in
    advance. What would you do then? You do not have to implement it - but
    observation.md asks.

    ------------------------------------------------------------------------
    NaiveFilter 와 다른 점은 딱 하나다: 해시를 1 개가 아니라 k 개 쓴다.
    비트는 똑같이 n_bits 개만 쓴다. 그게 전부다.

    왜 k 인가 (§4.4.2 를 k 에 대해 미분):

      FP(k) = (1 - e^(-kn/m))^k

      p = e^(-kn/m)  (아직 0 인 비트의 비율) 로 치환하면
        k = -(m/n)·ln p           이고
        ln FP = k·ln(1-p) = -(m/n)·ln(p)·ln(1-p)

      m/n 은 상수이므로 ln(p)·ln(1-p) 를 최대화하면 된다.
        d/dp [ln p·ln(1-p)] = ln(1-p)/p - ln(p)/(1-p) = 0
      이 식은 p <-> 1-p 대칭이므로 최댓값은 p = 1/2 에서 나온다.

      p = 1/2  ->  e^(-kn/m) = 1/2  ->  kn/m = ln 2

              m
        k* = --- · ln 2
              n

      여기서는 m/n = 80,000/8,000 = 10 이므로 k* = 10·ln2 = 6.93 -> 7.
      그때의 오류율은 (1 - e^(-0.7))^7 = 0.819% 이고, 이것이 아이템당
      10 비트에서의 '바닥' 이다. 넘어설 수 없고 도달만 할 수 있다.

      p = 1/2 이 최적이라는 것은 '비트의 정확히 절반이 켜졌을 때가 최적'
      이라는 뜻이다. 비어 있으면 정보를 덜 쓴 것이고, 차 있으면 아무 질문에나
      yes 가 나온다.
    """

    # 채점기가 미리 알려주는 삽입 개수 (bench.py 의 N_INSERT).
    # 실제 스트림이라면 이 값을 알 수 없다. 그 경우 어떻게 할지는
    # observation.md 의 R6 질문이고, 여기서는 알려준 값을 쓴다.
    EXPECTED_ITEMS = 8_000

    def __init__(self, n_bits, seed=246):
        self.n_bits = n_bits
        self.seed = seed

        # k 를 상수 7 로 박지 않고 공식에서 계산한다. 그래야 예산이나 삽입
        # 개수가 바뀌어도 최적값을 따라가고, 7 이 어디서 나온 숫자인지가
        # 코드에 남는다. 최소 1 개는 있어야 하므로 max 로 바닥을 깐다.
        self.k = max(1, round((n_bits / self.EXPECTED_ITEMS) * math.log(2)))

        # 비트를 bytearray 에 눌러 담는다. n_bits=80,000 이면 10,000 바이트.
        # bytearray(n_bits) 로 '비트당 1 바이트' 를 쓰면 실제 사용량이
        # 640,000 비트가 되어 R3(예산 80,000 비트) 에서 즉시 실격이다.
        self.bits = bytearray((n_bits + 7) // 8)

    def _indices(self, item):
        """이 아이템이 켜야 할 비트 번호 k 개.

        NaiveFilter 의 _index 와 같은 방식이되 key 에 해시 번호를 섞어
        서로 다른 k 개의 해시로 쓴다. yield 라서 __contains__ 가 중간에
        멈추면 남은 해시는 계산되지 않는다.
        """
        data = str(item).encode()
        for i in range(self.k):
            digest = hashlib.blake2b(data, digest_size=8,
                                     key=f"{self.seed}-{i}".encode()).digest()
            yield int.from_bytes(digest, "big") % self.n_bits

    def add(self, item):
        for idx in self._indices(item):
            # idx >> 3 = 몇 번째 바이트, idx & 7 = 그 안에서 몇 번째 비트
            self.bits[idx >> 3] |= 1 << (idx & 7)

    def __contains__(self, item):
        # k 개가 전부 1 이어야 yes. 하나라도 0 이면 즉시 False.
        #
        # R4(거짓 음성 0) 가 여기서 구조적으로 보장된다. add 는 비트를 1 로만
        # 만들고, 0 으로 되돌리는 코드가 이 클래스에 없다. 따라서 넣은 아이템은
        # 반드시 전부 1 이고 반드시 yes 다. 확률이 낮은 것이 아니라 경로가 없다.
        return all(self.bits[idx >> 3] >> (idx & 7) & 1
                   for idx in self._indices(item))

    def memory_bits(self):
        """실제로 쓰는 메모리를 전부 센다 (R3).

        데이터 크기에 비례해 자라는 저장소는 self.bits 하나뿐이다. 아이템을
        따로 보관하는 set 이나 list 는 없다 - 있으면 그건 최적화가 아니라
        예산을 속이는 것이다.

        len(self.bits) * 8 로 세는 이유: 바이트 배열이므로 실제 점유는 바이트
        단위이고, n_bits 가 8 의 배수가 아니면 마지막 바이트에 남는 비트도
        실제로는 할당되어 있다. 그 몫까지 정직하게 포함한다.
        (n_bits=80,000 이면 10,000 바이트 = 정확히 80,000 비트.)

        self.k, self.seed, self.n_bits 는 아이템 수와 무관한 상수 몇 개라
        데이터 구조의 크기에 들어가지 않는다.
        """
        return len(self.bits) * 8
