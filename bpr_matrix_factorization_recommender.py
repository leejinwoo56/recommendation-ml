import sys
import math
import random
from collections import defaultdict

import numpy as np


def load_implicit_ratings(path):

    user_pos_items = defaultdict(set)
    all_items = set()

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            user, movie, rating, ts = parts
            u = int(user)
            i = int(movie)
            user_pos_items[u].add(i)
            all_items.add(i)

    return user_pos_items, list(all_items)


def train_bpr_mf(
    user_pos_items,
    all_items,
    n_factors=40,
    n_epochs=20,
    n_samples_per_epoch=100000,
    lr=0.01,
    reg=0.01,
    seed=42
):

    random.seed(seed)
    np.random.seed(seed)

    users = list(user_pos_items.keys())
    n_users = len(users)
    n_items = len(all_items)

    # latent vector �룯�뜃由곤옙�넅 (占쎌삂占쏙옙占� 占쎌삏占쎈쑁揶쏉옙)
    P = {}  # user -> p_u
    for u in users:
        P[u] = 0.01 * np.random.randn(n_factors)

    Q = {}  # item -> q_i
    b_i = {}  # item bias
    for i in all_items:
        Q[i] = 0.01 * np.random.randn(n_factors)
        b_i[i] = 0.0

    def sample_triplet():

        while True:
            u = random.choice(users)
            pos_items = user_pos_items[u]
            if not pos_items:
                continue
            i = random.choice(list(pos_items))
            # negative 占쎄묘占쎈탣筌랃옙
            while True:
                j = random.choice(all_items)
                if j not in pos_items:
                    break
            return u, i, j

    for epoch in range(n_epochs):
        loss = 0.0

        for _ in range(n_samples_per_epoch):
            u, i, j = sample_triplet()

            p_u = P[u]
            q_i = Q[i]
            q_j = Q[j]

            # score(u, i) - score(u, j)
            x_ui = b_i[i] + np.dot(p_u, q_i)
            x_uj = b_i[j] + np.dot(p_u, q_j)
            x_uij = x_ui - x_uj

            # log-sigmoid
            sigmoid = 1.0 / (1.0 + math.exp(-x_uij))
            grad = 1.0 - sigmoid  # ascent 獄쎻뫚堉�

            p_old = p_u.copy()
            q_i_old = q_i.copy()
            q_j_old = q_j.copy()

            # user latent
            P[u] += lr * (grad * (q_i_old - q_j_old) - reg * p_old)

            # positive item latent & bias
            Q[i] += lr * (grad * p_old - reg * q_i_old)
            b_i[i] += lr * (grad - reg * b_i[i])

            # negative item latent & bias
            Q[j] += lr * (-grad * p_old - reg * q_j_old)
            b_i[j] += lr * (-grad - reg * b_i[j])

            
            loss += -math.log(sigmoid + 1e-10) + reg * (
                np.dot(p_old, p_old)
                + np.dot(q_i_old, q_i_old)
                + np.dot(q_j_old, q_j_old)
                + b_i[i] ** 2
                + b_i[j] ** 2
            )

        avg_loss = loss / float(n_samples_per_epoch)
        print(f"[BPR] epoch {epoch+1}/{n_epochs}, avg loss={avg_loss:.4f}", file=sys.stderr)
        if (epoch+1) % 500 == 0 :
            lr *= 0.8
            print(f"[BPR] epoch {epoch+1}: reduce lr 占쎈꼥 {lr}", file=sys.stderr)

    return P, Q, b_i


def predict_score_bpr(u, i, P, Q, b_i, pop_fallback=None):

    if (u in P) and (i in Q):
        return b_i.get(i, 0.0) + float(np.dot(P[u], Q[i]))
    else:
        if pop_fallback is not None:
            return pop_fallback.get(i, 0.0)
        return 0.0


def compute_popularity_scores(user_pos_items):

    item_counts = defaultdict(int)
    for u, items in user_pos_items.items():
        for i in items:
            item_counts[i] += 1

    pop_raw = {}
    for i, c in item_counts.items():
        pop_raw[i] = math.log(1.0 + c)

    if not pop_raw:
        return {}

    vals = list(pop_raw.values())
    vmin = min(vals)
    vmax = max(vals)

    pop_score = {}
    if vmax == vmin:
        for i in item_counts.keys():
            pop_score[i] = 0.5
    else:
        for i, v in pop_raw.items():
            pop_score[i] = (v - vmin) / (vmax - vmin)

    return pop_score


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hw2_3c.py path/to/ratings.txt path/to/ratings_test_or_val_test.txt", file=sys.stderr)
        sys.exit(1)

    train_data = sys.argv[1]  
    test_data = sys.argv[2]   

    # 1) implicit 
    print(f"[hw2_3c BPR] Loading training data from {train_data} ...", file=sys.stderr)
    user_pos_items, all_items = load_implicit_ratings(train_data)
    print(f"[hw2_3c BPR] #users={len(user_pos_items)}, #items={len(all_items)}", file=sys.stderr)

    # 2) popularity fallback
    print("[hw2_3c BPR] Computing popularity fallback scores ...", file=sys.stderr)
    pop_scores = compute_popularity_scores(user_pos_items)

    # 3) BPR-MF 
    print("[hw2_3c BPR] Training BPR-MF model ...", file=sys.stderr)
    P, Q, b_i = train_bpr_mf(
        user_pos_items,
        all_items,
        n_factors=80,          # latent dimension 
        n_epochs=400,           # epoch
        n_samples_per_epoch=90000,  
        lr=0.01,
        reg=0.02,
        seed=42
    )

    print(f"[hw2_3c BPR] Scoring test data from {test_data} ...", file=sys.stderr)
    with open('output3c200.txt', 'w') as out, open(test_data, 'r') as fin:
        for line in fin:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            user_id, movie_id, timestamp = parts
            u = int(user_id)
            i = int(movie_id)

            score = predict_score_bpr(u, i, P, Q, b_i, pop_fallback=pop_scores)
            out.write(f"{user_id},{movie_id},{score},{timestamp}\n")

    print("[hw2_3c BPR] Done. Saved predictions to output3c.txt", file=sys.stderr)
