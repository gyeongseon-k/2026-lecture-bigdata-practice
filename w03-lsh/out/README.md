# Week 3 — Finding Similar Items

## 1. 과제 개요

교재 3장 §3.1~§3.5의 유사 항목 찾기를 다뤘다. 모든 문서 쌍을 비교하면 비교 횟수가 O(n²)로 늘어나므로, 문서를 샤글 집합과 MinHash 시그니처로 표현하고 LSH로 후보를 추렸다. Task 1은 구현, Task 2는 시간·메모리 측정, Task 3은 데이터에 적용하는 과제였다.

## 2. Task별 구현

### Task 1 — `task1_minhash.py`

- `jaccard(a, b)`: 교집합 크기를 합집합 크기로 나눴다. 두 집합이 모두 비어 있으면 0을 반환했다.
- `minhash_signatures()`: R2에 따라 행을 바깥 루프로 두고 한 번만 통과했다. 시그니처를 무한대로 초기화한 뒤 최솟값으로 갱신했다. 각 행의 해시값은 한 번 계산해 문서마다 재사용했다.
- `lsh_candidates()`: 밴드 조각을 튜플로 만들어 밴드별 버킷에 넣고, 같은 버킷에 모인 문서의 쌍을 후보로 뽑았다. 여러 밴드에서 나온 중복 쌍은 set으로 제거했다.

R5는 남는 행을 마지막 밴드에 합치는 방식으로 정했다. 이유는 `observation.md`에 정리했다. `--verify`로 교재 §3.3.5 예제와 비교해 `S1=[1,0], S2=[3,2], S3=[0,0], S4=[1,0]`을 확인했다.

### Task 2 — `task2_crossover.py`

구현 stub이 없는 측정 과제로, 제공된 스크립트에서 두 가지를 보완했다.

- `bench.build()`가 최대 2,120개 문서만 생성해 `build_scaled(n)`을 추가했다. `bench.py`는 수정하지 않고 상수를 읽어 같은 생성 규칙을 재현했다. `--selftest`로 n=2120에서 기존 결과와 같은지 확인했다.
- tracemalloc을 켜면 LSH가 약 12배, 브루트포스가 약 1.5배 느려졌다. `timed()`에서 측정 여부를 선택하도록 수정하고 `--no-trace` 옵션을 추가했다. 시간과 메모리를 따로 측정하고 JSON의 `traced` 필드로 구분했다.

문서 수 250~8000의 11개 크기를 측정했고, 교차점은 n≈320으로 추정했다.

### Task 3 — `task3_scale.py`의 `YourFinder`

문서를 길이 120의 시그니처로 만들고 40개 밴드로 나눈 뒤, 후보 쌍에 대해서만 `similarity()`를 호출했다. 샤글은 행 번호가 아닌 0~4999 범위의 정수이므로 `(a*x + b) % p` 형태의 해시를 만들었다. 샤글에서 문서 목록으로 연결되는 역색인을 만들어 실제 등장한 샤글만 처리했고, 결과를 비교할 수 있도록 난수 시드를 고정했다.

시그니처 길이 n=120, 밴드 수 b=40, 밴드당 행 수 r=3으로 정했다. 계단 위치는 약 0.292이며, S-커브 계산과 선택 이유는 `observation.md`에 정리했다. 비교 횟수는 2,246,140회에서 276회로 약 99.99% 줄었고, recall과 precision은 모두 100%로 strong 등급을 받았다.

## 3. 실행 방법

`w03-lsh` 폴더에서 실행한다. Task 2는 tracemalloc의 영향 때문에 같은 크기를 두 번 측정한다. `--no-trace`를 주면 tracemalloc을 끄고 시간만 정확히 재고, 옵션 없이 실행하면 tracemalloc을 켜서 피크 메모리를 잰다. 아래는 6개 크기로 측정하는 예시이며, 전체 11개 크기를 재현할 때는 `out/crossover.json`에 기록된 크기로 `--sizes`를 맞춘다.

```bash
# 가상환경
source ../.venv/bin/activate

# Task 1 검증
python3 task1_minhash.py --verify

# Task 2 검증 및 측정
python3 task2_crossover.py --selftest
python3 task2_crossover.py --sizes 250,500,1000,2000,4000,8000 --no-trace  # 시간 측정
python3 task2_crossover.py --sizes 250,500,1000,2000,4000,8000             # 메모리 측정

# Task 3 채점 및 전체 확인
python3 bench.py --yours
python3 test_tasks.py
python3 ../check.py w03
```

## 4. 결과 파일

결과 파일은 `out/`에 정리했다.

| 파일 | 내용 |
| --- | --- |
| `bench.txt` | Task 3 채점 결과 |
| `crossover.json` | Task 2 측정 원본. `traced`로 시간용·메모리용 측정 구분 |
| `curve.md` | Task 2의 A1~A8 답변과 측정 결과 해석 |
| `observation.md` | Task별 관찰과 9개 질문 답변 |

## 5. 어려웠던 점

MinHash 값이 일치할 확률이 Jaccard 유사도와 같아지는 이유를 이해하는 데 시간이 걸렸다. 또 tracemalloc이 두 알고리즘에 서로 다른 영향을 주는 것을 보고, 측정 도구의 영향도 확인해야 한다는 점을 배웠다.
