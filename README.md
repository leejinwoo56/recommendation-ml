## Projects

### 1. Spark K-Means Diameter Analysis

**File:** `src/spark_kmeans_diameter_analysis.py`

#### 개요
PySpark를 이용해 2차원 또는 다차원 점 데이터를 읽고, farthest-first 방식으로 초기 중심점을 선택한 뒤 각 점을 가장 가까운 중심점에 할당하여 클러스터를 구성하고, 각 클러스터의 diameter를 계산해 평균 diameter를 구하는 프로젝트입니다.

#### 구현 내용
- 입력 데이터의 좌표값 파싱
- 유클리드 거리 기반 점 간 거리 계산
- farthest-first traversal 방식의 초기 centroid 선택
- 각 점을 가장 가까운 centroid에 할당
- 클러스터별 diameter 계산
- 전체 클러스터의 평균 diameter 산출

#### 배운 점
- PySpark를 이용해 대용량 점 데이터에 대한 군집 분석 흐름을 구현하는 방법
- k-means에서 초기 중심점 선택이 결과에 미치는 영향
- 클러스터링 결과를 평균 diameter 같은 지표로 평가하는 방법
- 분산 환경에서 데이터 로딩, 브로드캐스트, 그룹화 연산을 구성하는 방식

---

### 2. SVD, PCA, and Power Iteration

**File:** `src/svd_pca_power_iteration.py`

#### 개요
행렬에 대해 power iteration을 사용하여 지배적인 고유벡터와 고유값을 구하고, 이를 바탕으로 SVD를 직접 구현한 뒤 rank-k approximation, energy retention, PCA projection, 그리고 random projection과의 distance correlation 비교까지 수행하는 프로젝트입니다.

#### 구현 내용
- 행렬 곱을 이용한 \(M^T M\), \(M M^T\) 계산
- power iteration 기반 dominant eigenvector 추정
- deflation을 통한 다중 고유값 계산
- SVD 직접 구현
- rank-k matrix approximation 계산
- retained energy 비율 계산
- PCA via SVD 구현
- random projection과 PCA의 distance correlation 비교

#### 배운 점
- SVD와 PCA의 수학적 연결 관계를 구현을 통해 이해
- power iteration과 deflation이 고유값 문제에서 어떻게 사용되는지 학습
- 차원 축소 시 retained energy가 가지는 의미 이해
- PCA와 random projection을 비교하며 차원 축소 품질을 평가하는 방법 학습

---

### 3. Collaborative Filtering Recommender

**File:** `src/collaborative_filtering_recommender.py`

#### 개요
사용자-아이템 평점 행렬을 기반으로, user-based collaborative filtering과 item-based collaborative filtering을 각각 구현하여 추천 결과를 생성하는 프로젝트입니다.

#### 구현 내용
- 평점 파일을 읽어 user-item utility matrix 구성
- 사용자 평균 평점을 기준으로 정규화된 rating vector 생성
- cosine similarity 기반 사용자 간 유사도 계산
- user-based collaborative filtering을 통한 예측 점수 계산
- item-based collaborative filtering을 통한 예측 점수 계산
- 상위 추천 아이템 정렬 및 출력

#### 배운 점
- collaborative filtering의 핵심 아이디어를 직접 구현하며 이해
- user-based 방식과 item-based 방식의 차이점 학습
- cosine similarity가 추천 시스템에서 어떻게 활용되는지 이해
- sparse rating matrix를 dict 기반으로 다루는 방법 경험

---

### 4. BPR Matrix Factorization Recommender

**File:** `src/bpr_matrix_factorization_recommender.py`

#### 개요
implicit feedback 데이터를 바탕으로 BPR-MF(Bayesian Personalized Ranking Matrix Factorization)를 학습하여 사용자-아이템 선호도를 예측하고, 테스트 데이터에 대한 추천 점수를 출력하는 프로젝트입니다.

#### 구현 내용
- implicit feedback 형태의 사용자-아이템 상호작용 데이터 로드
- positive item / negative item sampling
- BPR objective 기반 SGD 학습
- user latent vector와 item latent vector 학습
- item bias 학습
- popularity score 기반 fallback prediction 구성
- 학습된 모델을 이용한 테스트 데이터 점수 예측

#### 배운 점
- explicit rating이 아닌 implicit feedback 기반 추천의 개념 이해
- BPR가 pairwise ranking 문제를 어떻게 학습하는지 학습
- matrix factorization에서 latent factor가 가지는 의미 이해
- cold-start 또는 미관측 데이터에 대해 popularity fallback을 사용하는 아이디어 경험

---

## Skills Demonstrated

- Python
- PySpark
- NumPy
- K-Means Clustering
- Cluster Evaluation
- Power Iteration
- Eigenvalue Decomposition
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)
- Random Projection
- Distance Correlation
- Recommender Systems
- Collaborative Filtering
- Cosine Similarity
- Matrix Factorization
- Bayesian Personalized Ranking (BPR)
- Algorithm Implementation
