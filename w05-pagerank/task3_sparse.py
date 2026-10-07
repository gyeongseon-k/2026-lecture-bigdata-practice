#!/usr/bin/env python3
"""Week 5 · Task 3 — PageRank on a graph that will not fit as a matrix.

Textbook §5.2 (efficient PageRank), §5.2.1 - §5.2.3.

`DenseMatrix` is PageRank written the way the equations are written: build the
transition matrix M, multiply. It is correct, it is easy to read, and it stores
n^2 numbers for a graph with almost no edges.

The web's matrix is about 99.9999% zeros. Storing them is the problem, and
§5.2 is the chapter about not doing that.

    python3 bench.py
    python3 bench.py --yours

Correctness first: the harness compares your ranks against the dense version
element by element. A fast PageRank that ranks pages differently is a different
algorithm, not a faster one.
"""


class DenseMatrix:
    """PageRank as written in the equations. Stores n^2 floats."""

    def __init__(self, beta=0.85, tol=1e-10, max_iter=100):
        self.beta, self.tol, self.max_iter = beta, tol, max_iter

    def run(self, graph):
        nodes = list(graph)
        n = len(nodes)
        index = {v: i for i, v in enumerate(nodes)}

        # the full transition matrix, zeros and all
        M = [[0.0] * n for _ in range(n)]
        for v, outs in graph.items():
            if outs:
                share = 1.0 / len(outs)
                for w in outs:
                    M[index[w]][index[v]] = share
            else:
                for i in range(n):           # dead end: spread it everywhere
                    M[i][index[v]] = 1.0 / n

        r = [1.0 / n] * n
        for self.iterations in range(1, self.max_iter + 1):
            nr = [0.0] * n
            for i in range(n):
                row = M[i]
                s = 0.0
                for j in range(n):
                    if row[j]:
                        s += row[j] * r[j]
                nr[i] = self.beta * s + (1 - self.beta) / n
            delta = sum(abs(a - b) for a, b in zip(nr, r))
            r = nr
            if delta < self.tol:
                break
        return {v: r[index[v]] for v in nodes}

    def memory_floats(self):
        return getattr(self, "_n", 0) ** 2


class YourPageRank:
    """Your PageRank.

        __init__(beta=0.85, tol=1e-10, max_iter=100)
        run(graph) -> {node: rank}
        memory_floats() -> the largest number of floats you held at once

    Same ranks, to within 1e-9 per node. Far less memory.

    `graph` is {node: [out-neighbours]}. Note what that already is: an adjacency
    list, which is the sparse representation. The dense version throws that
    structure away and then pays to get it back.

    Two things to be careful about, and they are the same two as Task 1:

      * dead ends, whose rank has to go somewhere
      * the teleport term, which touches every node and is therefore the one
        part that looks like it needs a dense operation - it does not, and
        working out why is the point of §5.2.3

    `memory_floats()` is on your honour and the harness reads it. Count the
    numbers you actually hold at once.

    ------------------------------------------------------------------------
    M 을 아예 만들지 않는다. 입력으로 받은 graph 가 이미 희소 표현이므로
    그것을 그대로 읽는다. DenseMatrix 가 n^2 = 1,440,000 개를 쓰는 자리에서
    이쪽이 들고 있는 float 은 딱 두 배열이다.

        r    현재 점수    n 개
        nr   다음 점수    n 개
        ---------------------
        합계             2n 개   (n=1,200 이면 2,400 개 -> 600 배 적다)

    간선은 float 을 하나도 차지하지 않는다. 간선마다 보내는 비율은 항상
    1/len(graph[v]) 이고 그때그때 계산할 수 있어서 저장할 이유가 없다.
    그래서 보관량이 간선 수 m 과 무관하다.

    왜 s 배열을 따로 두지 않는가:
      교과서 식 nr[i] = beta * s[i] + (1-beta)/n 을 그대로 쓰면 s 를 담을
      배열이 하나 더 필요해 보이고, 그러면 r + s + nr = 3n 을 동시에 들고
      있게 된다. 그 경우 memory_floats() 는 2n 이 아니라 3n 으로 신고해야 한다.
      하지만 s[v] 와 nr[v] 는 같은 노드의 값이고 nr[v] 를 구할 때 다른 노드의
      s 를 참조하지 않으므로, nr 을 s 역할로 쓰다가 마지막에 제자리에서
      변환하면 배열 하나로 끝난다. 신고값과 실제 배열 개수가 일치하도록
      이 방식을 택했다.

    DenseMatrix 와 1e-9 안에서 일치시키기 위해 맞춘 것:
      - 시작값 전원 1/n
      - leak/n 을 beta 를 곱하기 '전' 의 누적값에 넣는다 (dense 는 막다른 길
        열을 1/n 으로 채워 s 안에 포함시키므로 beta 가 함께 곱해진다)
      - beta 는 노드당 마지막에 한 번만 곱한다
      - L1 변화량을 측정한 뒤 교체하고, 그다음에 tol 과 비교한다
      - 기본값 beta=0.85, tol=1e-10, max_iter=100 도 동일
    """

    def __init__(self, beta=0.85, tol=1e-10, max_iter=100):
        self.beta, self.tol, self.max_iter = beta, tol, max_iter
        self.iterations = 0
        # run() 안에서 실제로 만든 배열 크기를 기록한다. 외부에서 넣어준 값을
        # 쓰지 않고 직접 센 값을 신고한다.
        self._peak_floats = 0

    def run(self, graph):
        n = len(graph)
        if n == 0:                       # 빈 그래프. 1/n 나눗셈 전에 빠져나간다.
            self.iterations = 0
            self._peak_floats = 0
            return {}

        beta, tol = self.beta, self.tol
        # 텔레포트 몫은 모든 노드에 같은 값이고 반복 중에 변하지 않는다.
        # 그래서 루프 밖에서 한 번만 계산한다. (R5 가 묻는 지점이다)
        teleport = (1.0 - beta) / n

        # 나가는 링크가 없는 노드. 그래프가 반복 중에 변하지 않으니 한 번만 찾는다.
        # 담고 있는 것은 노드 '이름'(문자열) 이라 float 이 아니다.
        dead_ends = [v for v, outs in graph.items() if not outs]

        # 배열 1: 현재 점수. 전원 1/n 에서 출발한다 (DenseMatrix 와 동일).
        r = {v: 1.0 / n for v in graph}

        # 동시에 들고 있는 float 은 r 과 nr 두 개뿐이다.
        # 그 밖의 float 은 beta, tol, teleport, dead_share, share, delta 같은
        # 스칼라 몇 개로, 그래프 크기와 무관한 고정 개수다.
        self._peak_floats = 2 * n

        step = 0
        for step in range(1, self.max_iter + 1):
            # --- 모든 노드가 링크와 무관하게 받는 몫 ---
            # 막다른 길에 고인 점수를 n 으로 나눠 균등 분배한다 (Task 1 과 같은
            # 방식). beta 는 여기서 곱하지 않는다 - dense 가 이 몫을 s 안에
            # 포함시켜 나중에 beta 를 곱하기 때문에 순서를 맞춘다.
            dead_share = sum(r[v] for v in dead_ends) / n

            # 배열 2: 다음 점수. 지금은 아직 's'(beta 를 곱하기 전의 누적값)
            # 역할이다. 전원을 dead_share 로 깔아 두고 여기에 링크 기여를 쌓는다.
            nr = {v: dead_share for v in graph}

            # --- 링크를 따라 점수를 나눠준다 (주는 쪽 기준) ---
            for v, outs in graph.items():
                if outs:
                    # 내 점수를 나가는 링크 개수로 쪼갠다. 링크마다 다시
                    # 나눌 필요가 없으니 한 번만 계산한다.
                    share = r[v] / len(outs)
                    for w in outs:
                        nr[w] += share
                # outs 가 비어 있으면 아무에게도 주지 않는다. 그 몫은 위에서
                # dead_share 로 이미 집계했으므로 이중 계산이 아니다.

            # --- 제자리 변환 + L1 변화량을 한 번의 순회로 ---
            # 여기서 nr 이 's' 에서 '다음 점수' 로 바뀐다. nr[v] 를 구할 때
            # 다른 노드의 값을 보지 않으므로 같은 칸을 덮어써도 안전하다.
            # 덕분에 배열이 하나 더 필요하지 않다.
            delta = 0.0
            for v in graph:
                nr[v] = beta * nr[v] + teleport
                delta += abs(nr[v] - r[v])

            r = nr
            if delta < tol:
                break

        self.iterations = step
        return r

    def memory_floats(self):
        """동시에 들고 있던 float 의 최대 개수.

        r 과 nr 두 배열뿐이므로 2n 이다. 세지 않은 것과 이유:

          graph       두 구현이 똑같이 받는 입력이고, 담고 있는 것은 문자열과
                      리스트다. float 이 아니다. DenseMatrix 도 세지 않는다.
          dead_ends   노드 이름(문자열) 목록. float 이 아니다.
          스칼라       beta, tol, teleport, dead_share, share, delta 등
                      그래프 크기와 무관한 고정 개수(약 6 개). 포함해도
                      2,400 -> 2,406 으로 결과가 달라지지 않지만, n^2 만
                      신고하는 DenseMatrix 의 관례와 맞추기 위해 제외하고
                      대신 여기에 적어 둔다.
        """
        return self._peak_floats
