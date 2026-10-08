"""
인공신경망(다층 퍼셉트론, MLP)을 이용한 자동차 연비 예측 프로그램
---------------------------------------------------------------
- 데이터: UCI Auto MPG (공개 데이터, 자동차 392대)
  입력 9개: 실린더 수, 배기량, 마력, 무게, 가속 시간, 연식, 제조 지역(미국/일본/유럽, 원-핫)
  출력 1개: 연비 mpg (miles per gallon)
- 모델: 입력(9) -> 은닉층1(16, ReLU) -> 은닉층2(8, ReLU) -> 출력(1, 선형)
- 학습: 순전파 -> MSE 손실 -> 역전파(연쇄법칙) -> 미니배치 경사하강법 + L2 정규화 + 조기 종료
- 평가: 5-겹 교차검증으로 신경망과 다중 선형회귀(최소제곱법)를 비교
- 신경망은 딥러닝 라이브러리(TensorFlow, PyTorch 등) 없이 NumPy로 직접 구현

실행: python ann_predict.py   (필요 패키지: numpy, pandas, matplotlib — Google Colab 기본 포함)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------
# 하이퍼파라미터
# ------------------------------------------------------------
LAYERS   = [9, 16, 8, 1]    # 층별 뉴런 수 (입력 - 은닉1 - 은닉2 - 출력)
EPOCHS   = 1000             # 최대 반복 횟수
BATCH    = 32               # 미니배치 크기
LR       = 0.01             # 학습률
L2       = 1e-3             # L2 정규화 계수
PATIENCE = 100              # 조기 종료 인내 횟수
K        = 5                # 교차검증 겹 수

# ------------------------------------------------------------
# 1. 데이터 불러오기
# ------------------------------------------------------------
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"

def load_data():
    try:
        df = pd.read_csv("mpg.csv")            # 같은 폴더에 파일이 있으면 사용
    except FileNotFoundError:
        df = pd.read_csv(URL)                  # 없으면 인터넷에서 불러오기
    df = df.dropna(subset=["horsepower"])      # 마력 결측치 6대 제거 → 392대
    num_cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
    origin = pd.get_dummies(df["origin"]).astype(float)   # 범주형 → 원-핫 인코딩 (europe, japan, usa)
    X = np.hstack([df[num_cols].to_numpy(float), origin.to_numpy()])
    y = df["mpg"].to_numpy(float).reshape(-1, 1)
    names = num_cols + [f"origin_{c}" for c in origin.columns]
    return X, y, names

# ------------------------------------------------------------
# 2. 신경망 구성 요소
# ------------------------------------------------------------
def relu(z):      return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)
def mse(p, t):    return float(np.mean((p - t) ** 2))
def r2(p, t):     return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))

class MLP:
    def __init__(self, sizes):
        # He 초기화: ReLU 층에서 기울기 소실/폭발을 줄이기 위해 분산을 2/n_in 으로 설정
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((1, b)) for b in sizes[1:]]

    def forward(self, X):
        """순전파: 각 층의 z(가중합)와 a(활성값)를 저장해 두었다가 역전파에 사용"""
        self.a, self.z = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.a[-1] @ W + b
            self.z.append(z)
            # 마지막 층은 회귀(연속값 출력)이므로 활성화 함수 없음(항등 함수)
            self.a.append(z if i == len(self.W) - 1 else relu(z))
        return self.a[-1]

    def backward(self, y, lr, l2):
        """역전파: 손실 L = mean((ŷ - y)^2) 의 기울기를 출력층부터 거꾸로 전달하며 가중치 갱신"""
        m = y.shape[0]
        delta = 2 * (self.a[-1] - y) / m                # δ = ∂L/∂z (출력층)
        for i in reversed(range(len(self.W))):
            dW = self.a[i].T @ delta + l2 * self.W[i]   # ∂L/∂W (+ L2 정규화 항)
            db = delta.sum(0, keepdims=True)            # ∂L/∂b
            if i > 0:                                   # 연쇄법칙으로 이전 층의 δ 계산
                delta = (delta @ self.W[i].T) * relu_grad(self.z[i - 1])
            self.W[i] -= lr * dW                        # 경사하강법: W ← W − η·∂L/∂W
            self.b[i] -= lr * db

    def get(self):    return [W.copy() for W in self.W], [b.copy() for b in self.b]
    def set(self, p): self.W, self.b = p

# ------------------------------------------------------------
# 3. 학습 함수 (미니배치 경사하강법 + 조기 종료)
# ------------------------------------------------------------
def train(Xtr, ytr, Xva, yva, verbose=False):
    model = MLP(LAYERS)
    hist_tr, hist_va = [], []
    best, best_p, wait = np.inf, None, 0
    for ep in range(1, EPOCHS + 1):
        perm = rng.permutation(len(Xtr))                # 매 epoch 데이터 순서를 섞음
        for s in range(0, len(Xtr), BATCH):
            b = perm[s:s + BATCH]
            model.forward(Xtr[b])
            model.backward(ytr[b], LR, L2)
        l_tr, l_va = mse(model.forward(Xtr), ytr), mse(model.forward(Xva), yva)
        hist_tr.append(l_tr); hist_va.append(l_va)
        if l_va < best:                                 # 검증 손실 최솟값일 때 가중치 저장
            best, best_p, wait = l_va, model.get(), 0
        else:
            wait += 1
            if wait >= PATIENCE:                        # 개선이 없으면 중단 → 과적합 방지
                break
        if verbose and ep % 50 == 0:
            print(f"  epoch {ep:4d} | train MSE {l_tr:.4f} | val MSE {l_va:.4f}")
    model.set(best_p)
    return model, hist_tr, hist_va

def linear_regression(Xtr, ytr, Xte):
    """비교용: 다중 선형회귀(최소제곱법)  w = argmin ||Xw - y||²"""
    add1 = lambda A: np.hstack([np.ones((len(A), 1)), A])
    w = np.linalg.lstsq(add1(Xtr), ytr, rcond=None)[0]
    return add1(Xte) @ w

# ------------------------------------------------------------
# 4. K-겹 교차검증
# ------------------------------------------------------------
X, y, names = load_data()
print(f"데이터: {X.shape[0]}대, 입력 특성 {X.shape[1]}개\n  {names}\n")

folds = np.array_split(rng.permutation(len(X)), K)
res = {k: [] for k in ["ann_r2", "ann_rmse", "ann_mae", "lin_r2", "lin_rmse", "lin_mae"]}
plot_data = None
all_true, all_ann, all_lin = [], [], []   # 모든 fold의 테스트 예측을 모아 그래프에 사용

for k in range(K):
    te = folds[k]
    rest = np.concatenate([folds[j] for j in range(K) if j != k])
    n_va = len(rest) // 6                               # 나머지 중 약 15%를 검증용으로
    va, tr = rest[:n_va], rest[n_va:]

    # 표준화(평균 0, 표준편차 1) — 통계량은 학습 데이터로만 계산(정보 누설 방지)
    x_mu, x_sd = X[tr].mean(0), X[tr].std(0)
    y_mu, y_sd = y[tr].mean(), y[tr].std()
    nx = lambda A: (A - x_mu) / x_sd
    ny = lambda A: (A - y_mu) / y_sd

    print(f"[Fold {k+1}/{K}] train {len(tr)} / val {len(va)} / test {len(te)}")
    model, h_tr, h_va = train(nx(X[tr]), ny(y[tr]), nx(X[va]), ny(y[va]), verbose=(k == 0))

    true = y[te]
    pred = model.forward(nx(X[te])) * y_sd + y_mu      # 원래 단위(mpg)로 복원
    lin  = linear_regression(nx(X[tr]), ny(y[tr]), nx(X[te])) * y_sd + y_mu

    for tag, p in [("ann", pred), ("lin", lin)]:
        res[f"{tag}_r2"].append(r2(p, true))
        res[f"{tag}_rmse"].append(np.sqrt(mse(p, true)))
        res[f"{tag}_mae"].append(float(np.mean(np.abs(p - true))))
    print(f"  → 신경망 R² {res['ann_r2'][-1]:.3f} | 선형회귀 R² {res['lin_r2'][-1]:.3f}  (학습 {len(h_tr)} epoch)")

    all_true.append(true); all_ann.append(pred); all_lin.append(lin)
    if k == 0:
        plot_data = (h_tr, h_va)

avg = lambda key: np.mean(res[key])
sd  = lambda key: np.std(res[key])
print("\n=============== 5-겹 교차검증 평균 결과 ===============")
print(f"신경망(MLP)   R² {avg('ann_r2'):.3f} ± {sd('ann_r2'):.3f} | RMSE {avg('ann_rmse'):.2f} | MAE {avg('ann_mae'):.2f} mpg")
print(f"다중 선형회귀 R² {avg('lin_r2'):.3f} ± {sd('lin_r2'):.3f} | RMSE {avg('lin_rmse'):.2f} | MAE {avg('lin_mae'):.2f} mpg")

# ------------------------------------------------------------
# 5. 시각화 (학습곡선: Fold 1 / 산점도: 5개 fold 테스트 예측 전체)
# ------------------------------------------------------------
h_tr, h_va = plot_data
true, pred, lin = np.vstack(all_true), np.vstack(all_ann), np.vstack(all_lin)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(h_tr, label="Train"); ax[0].plot(h_va, label="Validation")
ax[0].set_title("Learning curve (Fold 1, standardized MSE)"); ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("MSE"); ax[0].set_yscale("log"); ax[0].legend(); ax[0].grid(alpha=.3)
lo, hi = true.min(), true.max()
ax[1].scatter(true, lin, s=16, alpha=.5, marker="x", label=f"Linear (R²={r2(lin, true):.3f})")
ax[1].scatter(true, pred, s=18, alpha=.8, label=f"Neural net (R²={r2(pred, true):.3f})")
ax[1].plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x (perfect)")
ax[1].set_title("5-fold test predictions: actual vs predicted mpg")
ax[1].set_xlabel("Actual mpg"); ax[1].set_ylabel("Predicted mpg"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("ann_result.png", dpi=130)
print("\n그래프 저장: ann_result.png")

# ------------------------------------------------------------
# 6. 새 자동차 연비 예측 예시 (전체 데이터로 다시 학습한 모델 사용)
# ------------------------------------------------------------
idx = rng.permutation(len(X)); n_va = len(X) // 7
va, tr = idx[:n_va], idx[n_va:]
x_mu, x_sd = X[tr].mean(0), X[tr].std(0); y_mu, y_sd = y[tr].mean(), y[tr].std()
final, _, _ = train((X[tr] - x_mu) / x_sd, (y[tr] - y_mu) / y_sd, (X[va] - x_mu) / x_sd, (y[va] - y_mu) / y_sd)
# [실린더, 배기량, 마력, 무게(lb), 가속(s), 연식(19xx), europe, japan, usa]
new_car = np.array([[4, 120, 95, 2400, 16.0, 80, 0, 1, 0]])
mpg = final.forward((new_car - x_mu) / x_sd)[0, 0] * y_sd + y_mu
print(f"예시) 1980년식 일본 4기통 95마력 2400lb 차량의 예상 연비: {mpg:.1f} mpg")
