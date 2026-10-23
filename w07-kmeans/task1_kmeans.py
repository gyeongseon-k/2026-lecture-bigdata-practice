#!/usr/bin/env python3
"""Week 7 · Task 1 — k-means, and the thing that decides the answer.

Textbook §7.1 (clustering, the curse of dimensionality), §7.3 (k-means).

k-means is four lines: pick centroids, assign every point to the nearest one,
move each centroid to the mean of its points, repeat. It always converges.

It converges to a **local** optimum, and which one depends entirely on where you
started. That is not a footnote - it is the difference between a clustering that
means something and one that does not, and you can watch it happen on nine points.

    python3 task1_kmeans.py --verify
"""
import argparse, math, random

# Three obvious blobs of three points each. Any sensible clustering finds them.
BLOBS = [(0.0, 0.0), (0.2, 0.1), (0.1, 0.2),
         (8.0, 8.0), (8.2, 8.1), (8.1, 8.2),
         (0.0, 8.0), (0.2, 8.1), (0.1, 8.2)]


def euclidean(a, b):
    """Distance between two points of any dimension."""
    # 같은 자리 좌표끼리 짝지어, 차이를 제곱해 모두 더한 뒤 제곱근을 취한다.
    # zip 이 길이에 맞춰 짝지어 주므로 차원 수를 미리 알 필요가 없다 (R1).
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def assign(points, centroids):
    """Nearest centroid for each point. Return a list of centroid indices.

    Ties go to the lower index, so that everybody's answer is the same.
    """
    labels = []
    for p in points:
        # "지금까지 가장 가까운 거리"를 무한대로 두면 첫 비교가 무조건 성립한다.
        best, best_index = float("inf"), 0
        for j, c in enumerate(centroids):
            d = euclidean(p, c)
            # 0번 중심부터 순서대로 보면서 "더 작을 때만" 갱신한다.
            # 거리가 같으면 갱신하지 않으므로 먼저 본 낮은 인덱스가 살아남는다 (R2).
            if d < best:
                best, best_index = d, j
        labels.append(best_index)
    return labels


def update(points, labels, k, previous=None):
    """Move each centroid to the mean of its points.

    An empty cluster has no mean. Decide what you do - and say so, because the
    choice changes the result and there is more than one defensible answer.

    An empty cluster keeps its previous centroid, so `previous` must be given
    whenever one can occur. Without it, an empty cluster raises ValueError
    rather than passing silently.
    """
    centroids = []
    # 실제로 등장한 레이블만 돌면 빈 클러스터 자리가 사라져 중심 번호가 어긋난다.
    # 그래서 0 부터 k-1 까지 빠짐없이 돈다.
    for j in range(k):
        group = [p for p, l in zip(points, labels) if l == j]
        if group:
            # zip(*group) 은 점별로 묶인 좌표를 축별로 다시 묶어 준다.
            # 축마다 평균을 내면 그 클러스터의 무게중심이 된다.
            centroids.append(tuple(sum(axis) / len(group) for axis in zip(*group)))
        elif previous is not None:
            # R3 선택: 빈 클러스터는 예전 중심을 그대로 유지한다.
            centroids.append(previous[j])
        else:
            # 빈 클러스터가 생겼는데 되돌릴 중심이 없으면 조용히 넘기지 않는다.
            raise ValueError(
                f"cluster {j} is empty and no previous centroid was given")
    return centroids


def kmeans(points, k, init="random", seed=246, max_iter=100):
    """Run to convergence. Return (labels, centroids, sse).

    `sse` is the sum over points of the squared distance to their own centroid.
    It is what k-means minimises, and it is how you compare two runs.

    `init` is either:
        "random"   pick k points at random as the starting centroids
        "++"       k-means++ (§7.3.2): pick the first at random, then pick each
                   next one with probability proportional to the squared distance
                   to the nearest centroid already chosen

    Set `kmeans.iterations` to how many passes you used.
    """
    # 반복을 한 번도 돌지 않으면 labels 가 [None, ...] 로 남아 sse_of 에서
    # 터진다. 뒤늦게 터뜨리지 말고 들어온 값이 틀렸다는 것을 여기서 알린다.
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")

    # 전용 난수 발생기. 전역 random 을 쓰면 seed 가 적용되지 않아 40 시드 비교가
    # 무의미해진다.
    rng = random.Random(seed)

    # ----------------------------------------------------- 시작 중심 고르기
    if init == "random":
        # R4: 서로 다른 점 k 개를 그대로 시작 중심으로 쓴다. 같은 덩어리에서 두 개를
        # 뽑는 나쁜 출발은 그대로 일어나며, 그것이 이 과제에서 보려는 현상이다.
        centroids = rng.sample(points, k)

    elif init == "++":
        # R5: k-means++ (§7.3.2). 첫 중심만 무작위로 뽑고, 이후에는 "이미 뽑힌
        # 중심들 중 가장 가까운 것까지의 거리의 제곱"에 비례하는 확률로 뽑는다.
        centroids = [rng.choice(points)]
        while len(centroids) < k:
            # 점마다 가중치를 구한다. 이미 뽑힌 점은 거리가 0 이라 가중치도 0 이
            # 되므로 다시 뽑히지 않는다.
            weights = [min(euclidean(p, c) for c in centroids) ** 2
                       for p in points]
            total = sum(weights)

            if total > 0.0:
                # 누적합 방식: 가중치만큼 긴 막대를 일렬로 이어 붙이고 화살 한 발.
                # 난수 소비는 중심 하나당 정확히 1회다.
                arrow = rng.random() * total
                running, pick = 0.0, None
                for q, w in zip(points, weights):
                    running += w
                    if arrow < running:
                        pick = q
                        break
                if pick is None:
                    # arrow < total 이고 running 은 같은 순서로 더해 total 에
                    # 도달하므로 이론상 도달하지 않는다. 조용히 넘기지 않는다.
                    raise ValueError("weighted pick failed")
            else:
                # 모든 점이 이미 뽑힌 중심과 같은 위치인 경우. 가중치가 전부 0 이라
                # 비례 추출이 불가능하므로, 아직 뽑지 않은 점 중에서 고른다.
                remaining = [q for q in points if q not in centroids]
                if not remaining:
                    raise ValueError(
                        f"cannot pick {k} distinct starting centroids")
                pick = rng.choice(remaining)

            centroids.append(pick)

    else:
        raise ValueError(f"unknown init: {init!r}")

    # ----------------------------------------------------------- 반복
    # 레이블은 정수 리스트라 == 비교가 안전하다. 중심(실수)으로 수렴을 판정하면
    # 부동소수점 오차 때문에 영원히 "다르다"가 나올 수 있다.
    labels = [None] * len(points)
    for step in range(1, max_iter + 1):
        new_labels = assign(points, centroids)
        changed = new_labels != labels
        labels = new_labels
        # 네 번째 인자로 현재 중심을 넘긴다. 빈 클러스터는 이 값을 그대로 유지한다.
        centroids = update(points, labels, k, centroids)
        # 중심 갱신이 break 보다 먼저다. 그래야 돌려주는 중심이 최종 레이블의
        # 평균이 된다.
        if not changed:
            break

    # R7: for 문이 끝난 뒤에도 step 에는 멈춘 반복 번호가 남아 있다.
    kmeans.iterations = step
    return labels, centroids, sse_of(points, labels, centroids)


def sse_of(points, labels, centroids):
    """Sum of squared distances from each point to its assigned centroid."""
    # 거리가 아니라 거리의 "제곱" 합이다 (R6). euclidean 이 제곱근을 씌우므로
    # ** 2 로 되돌려서 더한다.
    return sum(euclidean(p, centroids[l]) ** 2 for p, l in zip(points, labels))


# ------------------------------------------------------------------- harness
def verify():
    fails = 0

    def check(label, ok, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<50} {detail}")
        fails += not ok

    try:
        check("distance in 2d", abs(euclidean((0, 0), (3, 4)) - 5) < 1e-9)
        check("distance in 5d",
              abs(euclidean((0,) * 5, (1,) * 5) - math.sqrt(5)) < 1e-9)
    except NotImplementedError:
        print("  euclidean is still a stub"); return 1

    try:
        labels, cents, sse = kmeans(BLOBS, 3, init="++", seed=1)
    except NotImplementedError:
        print("  kmeans is still a stub"); return 1

    groups = {}
    for p, l in zip(BLOBS, labels):
        groups.setdefault(l, []).append(p)
    check("three clusters of three", sorted(len(g) for g in groups.values()) == [3, 3, 3],
          sorted(len(g) for g in groups.values()))
    check("the three blobs were recovered",
          all(max(q[0] for q in g) - min(q[0] for q in g) < 1 and
              max(q[1] for q in g) - min(q[1] for q in g) < 1
              for g in groups.values()))
    check("sse is small when clusters are right", sse < 1.0, f"sse {sse:.4f}")

    # Bad luck with random starts should sometimes be visibly worse.
    randoms = [kmeans(BLOBS, 3, init="random", seed=s)[2] for s in range(40)]
    plusplus = [kmeans(BLOBS, 3, init="++", seed=s)[2] for s in range(40)]
    check("random init sometimes lands badly", max(randoms) > min(randoms) * 5,
          f"sse from {min(randoms):.3f} to {max(randoms):.3f}")
    check("k-means++ is more reliable than random",
          sum(plusplus) / len(plusplus) <= sum(randoms) / len(randoms),
          f"mean sse: ++ {sum(plusplus) / len(plusplus):.3f} vs "
          f"random {sum(randoms) / len(randoms):.3f}")

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())
