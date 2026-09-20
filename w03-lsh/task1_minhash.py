#!/usr/bin/env python3
"""Week 3 · Task 1 — Minhash and LSH, built from the matrix up.

Textbook §3.2 - §3.4.

Comparing every pair is quadratic, so it stops being possible somewhere around
a hundred thousand documents. The way out is two ideas stacked:

    minhash   replace a set with a short signature, such that the chance two
              signatures agree in a position equals their Jaccard similarity
    LSH       hash bands of those signatures so that similar pairs collide and
              you only ever compare the ones that did

You build both. The textbook's §3.3.5 example is small enough to check by hand,
and the harness checks you against it.

    python3 task1_minhash.py --verify
"""
import argparse

# §3.3.5. Rows are elements 0..4, columns are the sets S1..S4.
BOOK = [[1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 0, 1, 0]]
# The two hash functions the textbook uses on the row numbers.
BOOK_HASHES = [lambda r: (r + 1) % 5, lambda r: (3 * r + 1) % 5]


def jaccard(a, b):
    """|a and b| / |a or b|. Empty union is 0, not an error."""
    # 합집합을 먼저 구한다. 두 집합이 모두 비어 있으면 합집합도 비어 있고,
    # 그대로 나누면 ZeroDivisionError 가 난다. R1 이 요구하는 대로 0 을 돌려준다.
    union = a | b
    if not union:
        return 0
    # 겹치는 원소 수 / 전체 원소 수.
    return len(a & b) / len(union)


def minhash_signatures(columns, hashes, n_rows):
    """Build the signature matrix, one pass over the rows.

    `columns` is [set_of_row_numbers, ...], one entry per document.
    Return [[sig for each hash] for each column].

    The algorithm in §3.3.5 walks each row **once** and updates the signature
    of every column that has a 1 in it:

        sig[h][c] = min(sig[h][c], h(r))

    Doing it that way is the point. If you sort or re-scan per column you have
    written something correct that does not survive a dataset that does not fit
    in memory, and not fitting in memory is what this course is about.
    """
    # 시그니처 표를 무한대로 초기화한다. 앞으로 min() 으로 계속 줄여 나갈 것이므로
    # "아직 아무 값도 못 봤다" 를 뜻하는 출발점이 무한대여야 한다.
    # 모양은 [문서][해시] = 그 해시에 대한 지금까지의 최솟값.
    sig = [[float("inf")] * len(hashes) for _ in columns]

    # R2 의 핵심: 바깥 루프가 '행' 이다. 행렬을 딱 한 번만 위에서 아래로 훑는다.
    # 행 하나를 읽은 그 순간에 모든 문서의 시그니처를 동시에 갱신하므로,
    # 행렬 전체를 메모리에 들고 있지 않아도 (한 줄씩 흘려보내도) 동작한다.
    for r in range(n_rows):
        # 이 행의 해시값들을 먼저 한 번만 계산해 둔다.
        # 문서마다 다시 계산하면 같은 h(r) 을 쓸데없이 여러 번 구하게 된다.
        row_hashes = [h(r) for h in hashes]

        # 이 행에 1 이 있는 열(문서)들만 갱신한다.
        for c, column in enumerate(columns):
            if r not in column:
                continue  # 이 문서에는 이 행이 없다 -> 건드리지 않는다
            for k, value in enumerate(row_hashes):
                if value < sig[c][k]:
                    sig[c][k] = value

    return sig


def lsh_candidates(signatures, bands):
    """Split each signature into `bands` bands and hash each band.

    Two columns are candidates if they land in the same bucket for **at least
    one** band. Return {(i, j), ...} with i < j.

    The signature length must divide evenly by `bands`, or you have to decide
    what to do with the remainder. Say what you decided.
    """
    if not signatures or bands <= 0:
        return set()

    n = len(signatures[0])          # 시그니처 한 개의 길이 (해시 개수)
    bands = min(bands, n)           # 밴드가 행보다 많을 수는 없다
    rows_per_band = n // bands      # 밴드 하나가 가져갈 기본 행 수

    # --- R5 결정: 나눠떨어지지 않고 남는 행은 마지막 밴드에 합친다. ---
    # 이유 1) 남는 행을 버리면 그 행이 담고 있던 정보를 통째로 잃는다.
    #         시그니처는 이미 원본을 크게 줄여 놓은 요약본이라 더 버릴 여유가 없다.
    # 이유 2) 남는 행으로 '행 수가 모자란 짧은 밴드' 를 따로 만들면, 그 밴드는
    #         비교하는 행이 적어서 우연히 일치할 확률이 훨씬 높아진다. 즉 안 비슷한
    #         쌍까지 후보로 끌어들여 (오탐) 비교 횟수만 늘어난다.
    # 이유 3) 마지막 밴드에 합치면 그 밴드만 조금 길어질 뿐이다. 길어진 밴드는
    #         일치 조건이 더 빡세져서 후보가 조금 줄 뿐, 놓치는 쌍이 생기지는 않는다.
    #         (다른 밴드에서 걸리면 되므로 recall 손해가 작다.)
    # 요약하면 "버리지 않고, 짧은 밴드도 만들지 않는" 쪽을 택했다.

    # 밴드별 구간 [시작, 끝) 을 미리 계산한다. 마지막 밴드의 끝은 항상 n 이다.
    spans = []
    for b in range(bands):
        start = b * rows_per_band
        end = n if b == bands - 1 else (b + 1) * rows_per_band
        spans.append((start, end))

    candidates = set()
    for b, (start, end) in enumerate(spans):
        # 이 밴드에서 같은 조각을 가진 문서들을 같은 통(bucket)에 모은다.
        # 조각을 tuple 로 만드는 이유는 list 가 dict 의 key 로 쓰일 수 없기 때문.
        buckets = {}
        for i, sig in enumerate(signatures):
            key = tuple(sig[start:end])
            buckets.setdefault(key, []).append(i)

        # 같은 통에 2개 이상 들어 있으면 그 안의 모든 쌍이 후보다.
        for members in buckets.values():
            if len(members) < 2:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    # i < j 로 정규화해서 (a,b) 와 (b,a) 가 따로 세어지지 않게 한다.
                    candidates.add((members[x], members[y]))

    # set 이므로 여러 밴드에서 동시에 걸린 쌍도 자동으로 한 번만 남는다.
    return candidates


# ------------------------------------------------------------------- harness
def columns_from_matrix(matrix):
    n_rows, n_cols = len(matrix), len(matrix[0])
    return [{r for r in range(n_rows) if matrix[r][c]} for c in range(n_cols)]


def verify():
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<44} {got}"
              + ("" if ok else f"\n{'':>54}want {want}"))
        fails += not ok

    cols = columns_from_matrix(BOOK)
    try:
        # S1 = {0,3}, S4 = {0,2,3}: intersection 2, union 3
        check("jaccard(S1, S4)", round(jaccard(cols[0], cols[3]), 4), round(2 / 3, 4))
        check("jaccard(S1, S2)", jaccard(cols[0], cols[1]), 0.0)
        check("jaccard on empty sets", jaccard(set(), set()), 0)
    except NotImplementedError:
        print("  jaccard is still a stub"); return 1

    try:
        sig = minhash_signatures(cols, BOOK_HASHES, len(BOOK))
    except NotImplementedError:
        print("  minhash_signatures is still a stub"); return 1

    # Figure 3.4 in the textbook.
    check("signature of S1", sig[0], [1, 0])
    check("signature of S2", sig[1], [3, 2])
    check("signature of S3", sig[2], [0, 0])
    check("signature of S4", sig[3], [1, 0])

    try:
        cands = lsh_candidates([[1, 0], [3, 2], [0, 0], [1, 0]], bands=2)
    except NotImplementedError:
        print("  lsh_candidates is still a stub"); return 1
    # With one row per band, S1 and S4 are identical, so they must collide.
    check("S1 and S4 are candidates", (0, 3) in cands, True)
    check("S1 and S2 are not", (0, 1) in cands, False)

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    if not fails:
        print("  Note that S1 and S4 agree in both signature positions, which "
              "estimates\n  their similarity as 1.0 when it is actually 2/3. "
              "Two hashes is not many.")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())
