#!/usr/bin/env python3
"""Week 7 · Task 3 — The same clustering, with far fewer distances.

Textbook §7.3, and §7.1.3 for why distance is the expensive part.

`NaiveKMeans` does what the definition says: every iteration, compare every
point with every centroid. That is n x k distance computations per pass, and on
a real dataset the distance computation is almost the entire cost.

Most of that work is wasted. After the first couple of iterations, most points
do not change cluster, and most centroids barely move. The triangle inequality
lets you prove a point cannot have changed cluster **without measuring it**.

    python3 bench.py
    python3 bench.py --yours

Correctness first: your final SSE must match the baseline's. Fewer distances
with a worse clustering is not an improvement, it is a different algorithm.
"""


class NaiveKMeans:
    """Every point against every centroid, every iteration."""

    def __init__(self, k, max_iter=50):
        self.k, self.max_iter = k, max_iter

    def run(self, points, initial_centroids, distance):
        centroids = list(initial_centroids)
        labels = [None] * len(points)
        for self.iterations in range(1, self.max_iter + 1):
            changed = False
            for i, p in enumerate(points):
                best, bi = None, 0
                for j, c in enumerate(centroids):
                    d = distance(p, c)
                    if best is None or d < best - 1e-12:
                        best, bi = d, j
                if labels[i] != bi:
                    labels[i] = bi
                    changed = True
            centroids = _means(points, labels, self.k, centroids)
            if not changed:
                break
        return labels, centroids


def _means(points, labels, k, previous):
    out = []
    for j in range(k):
        group = [p for p, l in zip(points, labels) if l == j]
        if group:
            out.append(tuple(sum(v) / len(group) for v in zip(*group)))
        else:
            out.append(previous[j])
    return out


class YourKMeans:
    """Your k-means.

        __init__(k, max_iter=50)
        run(points, initial_centroids, distance) -> (labels, centroids)

    Same starting centroids, same `distance` function - which is counted. That
    counter is your score.

    The idea, from §7.3 and the triangle inequality:

        if a point is much closer to its current centroid than that centroid
        moved, and than any other centroid could have come, then it cannot have
        changed cluster, and you do not have to measure anything

    Keeping enough bookkeeping to make that test cheap, without keeping so much
    that the bookkeeping costs more than the distances, is the task.

    You may call `distance` on two centroids, and those calls are counted too -
    which is the correct accounting, since that is real work. There are k of
    them and n points, and k is much smaller than n. That asymmetry is the
    whole opening.
    """

    def __init__(self, k, max_iter=50):
        self.k, self.max_iter = k, max_iter

    def _scan(self, p, centroids, distance):
        """모든 중심을 재서 (최근접 인덱스, 그 거리, 두 번째로 가까운 거리)를 준다.

        최근접 판정은 NaiveKMeans 36-38 줄과 똑같은 규칙(d < best - 1e-12)이다.
        두 번째 거리는 이미 잰 값에서 고르므로 distance 를 더 부르지 않는다.
        """
        ds = [distance(p, c) for c in centroids]
        best, bi = None, 0
        for j, d in enumerate(ds):
            if best is None or d < best - 1e-12:
                best, bi = d, j
        others = [d for j, d in enumerate(ds) if j != bi]
        return bi, best, min(others) if others else float("inf")

    def run(self, points, initial_centroids, distance):
        k, n = self.k, len(points)
        centroids = list(initial_centroids)
        labels = [None] * n
        # 점마다 들고 있는 추가 숫자는 두 개뿐이다.
        upper = [0.0] * n   # u(x): 자기 중심까지 거리의 상한
        lower = [0.0] * n   # l(x): 다른 어떤 중심까지 거리의 하한

        for self.iterations in range(1, self.max_iter + 1):
            changed = False

            if self.iterations == 1:
                # 첫 반복은 증명할 근거가 없으므로 n x k 를 전부 잰다.
                for i, p in enumerate(points):
                    labels[i], upper[i], lower[i] = self._scan(
                        p, centroids, distance)
                changed = True          # labels 가 None 에서 바뀌었다
            else:
                # s[j] = 중심 j 에서 가장 가까운 다른 중심까지의 거리.
                # 반복당 k(k-1)/2 번이고, n x k 에 비하면 거의 공짜다.
                s = [float("inf")] * k
                for a in range(k):
                    for b in range(a + 1, k):
                        d = distance(centroids[a], centroids[b])
                        if d < s[a]:
                            s[a] = d
                        if d < s[b]:
                            s[b] = d

                for i, p in enumerate(points):
                    ci = labels[i]
                    # u <= l 이면 다른 중심이 더 가까울 수 없고,
                    # u <= s/2 이면 삼각부등식으로 자기 중심이 최근접임이 증명된다.
                    bound = s[ci] * 0.5
                    if lower[i] > bound:
                        bound = lower[i]
                    if upper[i] <= bound:
                        continue        # 아무것도 재지 않는다

                    # 상한이 느슨해서 걸렸을 수도 있으니 실제 거리로 한 번 조인다.
                    upper[i] = distance(p, centroids[ci])
                    if upper[i] <= bound:
                        continue

                    # 그래도 증명이 안 되면 이 점만 전부 다시 잰다.
                    bi, best, second = self._scan(p, centroids, distance)
                    upper[i], lower[i] = best, second
                    if bi != ci:
                        labels[i] = bi
                        changed = True

            # NaiveKMeans 42-44 줄과 같은 순서: 중심 갱신이 break 보다 먼저다.
            # 빈 클러스터 처리도 같은 _means 를 그대로 쓴다.
            new_centroids = _means(points, labels, self.k, centroids)
            if not changed:
                centroids = new_centroids
                break

            # 중심이 움직인 거리로 두 경계를 갱신한다. 반복당 k 번이다.
            moved = [distance(centroids[j], new_centroids[j]) for j in range(k)]
            worst = max(moved)
            for i in range(n):
                upper[i] += moved[labels[i]]
                lower[i] -= worst
            centroids = new_centroids

        return labels, centroids
