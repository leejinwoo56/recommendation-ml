import sys
# Take average of top-k similar user's ratings
topk_users_to_average = 10
# Take average of top-k similar items ratings
topk_items_to_average = 10
# Considering items 1 to 1000
num_items_for_prediction = 1000
# Top-k predictions of items with highest ratings
topk_items = 5
# Target user's id
target_user_id = 600


def cosine(a, b):
    """
    INPUT: two vectors a and b
    OUTPUT: cosine similarity between a and b

    DESCRIPTION:
    Takes two vectors and returns the cosine similarity.
    Here, vectors are represented as dicts: {index: value}.
    Only overlapping indices are used in the dot product.
    """
    if not a or not b:
        return 0.0

    # common keys only (others are implicitly 0)
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0

    dot = 0.0
    for k in common:
        dot += a[k] * b[k]

    norm_a = 0.0
    for v in a.values():
        norm_a += v * v
    norm_a = norm_a ** 0.5

    norm_b = 0.0
    for v in b.values():
        norm_b += v * v
    norm_b = norm_b ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def get_matrix(file_name):
    """
    INPUT: file name
    OUTPUT: utility matrix from the file

    DESCRIPTION:
    Reads the utility matrix from the file.
    Returns a dict containing:
      - user_ratings: {user: {item: rating}}
      - user_norm:   {user: {item: rating - user_mean}}
      - item_ratings:{item: {user: rating}}
      - item_norm:   {item: {user: rating - user_mean}}
      - user_mean:   {user: mean_rating}
    Ratings <= 0 are treated as 'no rating' (blank).
    """
    user_ratings = {}
    item_ratings = {}

    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                user_id = int(parts[0])
                item_id = int(parts[1])
                rating = float(parts[2])
            except ValueError:
                continue

            # Treat 0 as "no rating" (for user 600 on items 1~1000, etc.)
            if rating <= 0.0:
                continue

            if user_id not in user_ratings:
                user_ratings[user_id] = {}
            user_ratings[user_id][item_id] = rating

            if item_id not in item_ratings:
                item_ratings[item_id] = {}
            item_ratings[item_id][user_id] = rating

    # Compute per-user mean ratings
    user_mean = {}
    for u, items in user_ratings.items():
        if items:
            s = 0.0
            c = 0
            for r in items.values():
                s += r
                c += 1
            user_mean[u] = s / c
        else:
            user_mean[u] = 0.0

    # Build normalized user and item matrices
    user_norm = {}
    for u, items in user_ratings.items():
        mean_u = user_mean[u]
        nd = {}
        for i, r in items.items():
            nd[i] = r - mean_u
        user_norm[u] = nd

    item_norm = {}
    for i, users in item_ratings.items():
        nd = {}
        for u, r in users.items():
            mean_u = user_mean[u]
            nd[u] = r - mean_u
        item_norm[i] = nd

    return {
        "user_ratings": user_ratings,
        "user_norm": user_norm,
        "item_ratings": item_ratings,
        "item_norm": item_norm,
        "user_mean": user_mean,
    }


def user_based(umatrix, user_id):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using user-based collaborative
    filtering.
    - Similarity between users is cosine similarity on normalized ratings.
    - For each item, prediction is the simple average of original ratings
      from the top-k most similar users who rated that item.
    """
    user_ratings = umatrix["user_ratings"]
    user_norm = umatrix["user_norm"]

    if user_id not in user_norm:
        return []

    target_vec = user_norm[user_id]

    # Compute similarity to all other users
    sims = []
    for u, vec in user_norm.items():
        if u == user_id:
            continue
        sim = cosine(target_vec, vec)
        sims.append((sim, u))

    # Sort by similarity descending, pick top-k
    sims.sort(reverse=True, key=lambda x: x[0])
    neighbors = [u for _, u in sims[:topk_users_to_average]]

    # Predict for items 1..num_items_for_prediction
    predictions = []
    for item_id in range(1, num_items_for_prediction + 1):
        total = 0.0
        cnt = 0
        for u in neighbors:
            r = user_ratings.get(u, {}).get(item_id)
            if r is not None:
                total += r
                cnt += 1
        if cnt > 0:
            pred = total / cnt
            predictions.append((item_id, pred))

    # Sort by predicted rating desc, then item_id asc
    predictions.sort(key=lambda x: (-x[1], x[0]))

    return predictions[:topk_items]


def item_based(umatrix, user_id):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using item-based collaborative
    filtering.
    - Item similarity is cosine similarity on normalized item rating vectors.
    - For each target item I (1..1000), find top-k similar items J whose
      IDs are OUTSIDE [1, 1000].
    - Prediction for I is the average of the target user's ORIGINAL ratings
      on those similar items J that the user actually rated.
    """
    user_ratings = umatrix["user_ratings"]
    item_ratings = umatrix["item_ratings"]
    item_norm = umatrix["item_norm"]

    target_user_ratings = user_ratings.get(user_id, {})
    if not target_user_ratings:
        return []

    # Candidate neighbor items: items with ID > num_items_for_prediction
    candidate_items = [iid for iid in item_norm.keys() if iid > num_items_for_prediction]

    predictions = []

    for item_id in range(1, num_items_for_prediction + 1):
        base_vec = item_norm.get(item_id)
        # If this item has never been rated, skip
        if not base_vec:
            continue

        # Find similar items (only items outside [1, 1000])
        sims = []
        for other_id in candidate_items:
            other_vec = item_norm.get(other_id)
            if not other_vec:
                continue
            sim = cosine(base_vec, other_vec)
            # Only keep positively similar items
            
            sims.append((sim, other_id))

        if not sims:
            continue

        sims.sort(reverse=True, key=lambda x: x[0])
        neighbors = [iid for _, iid in sims[:topk_items_to_average]]

        total = 0.0
        cnt = 0
        for nid in neighbors:
            r = target_user_ratings.get(nid)
            if r is not None:
                total += r
                cnt += 1

        if cnt > 0:
            pred = total / cnt
            predictions.append((item_id, pred))

    # Sort by predicted rating desc, then item_id asc
    predictions.sort(key=lambda x: (-x[1], x[0]))

    return predictions[:topk_items]


if __name__ == "__main__":
    target_user_id = 600
    umatrix = get_matrix(sys.argv[1])
    ub_results = user_based(umatrix, target_user_id)

    with open('output3b_user.txt', 'w') as out:
        for item_id, score in ub_results:
            out.write(f"{item_id}\t{score}\n")
    
    ib_results = item_based(umatrix, target_user_id)
    with open('output3b_item.txt', 'w') as out:
        for item_id, score in ib_results:
            out.write(f"{item_id}\t{score}\n")
