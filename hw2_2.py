import numpy as np 

def matrix_mult(M):
    """
    INPUT: matrix M
    OUTPUT: M^T M and M M^T
    """
    MTM = np.dot(M.T, M)
    MMT = np.dot(M, M.T)
    return MTM, MMT


def power_iteration(M, delta: float = 1e-6, max_iter=10000):
    """
    INPUT: matrix M and threshold delta, max_iter
    OUTPUT: dominant eigenvector of M
    
    DESCRIPTION: Returns the dominant eigenvector of M using power iteration.
    Does not use np.linalg.norm, instead normalizes manually.
    """
    n = M.shape[0]
    # Step 1: Initialize a deterministic non-zero vector.
    b_k = np.random.rand(M.shape[1])

    # Normalize initial vector
    norm = np.sqrt(np.sum(b_k * b_k))
    if norm == 0.0:
        return b_k
    b_k = b_k / norm

    for _ in range(max_iter):
        # Step 2: Multiply by M
        b_k1 = np.dot(M, b_k)

        # Normalize
        norm = np.sqrt(np.sum(b_k1 * b_k1))
        if norm == 0.0:
            break
        b_k1 = b_k1 / norm

        # Check convergence using difference between iterations
        diff_vec = b_k1 - b_k
        diff = np.sqrt(np.sum(diff_vec * diff_vec))
        b_k = b_k1
        if diff < delta:
            break

    return b_k


def deflate_matrix(M, eigvec):
    """
    INPUT: matrix M, and dominant eigenvector eigvec
    OUTPUT: deflated matrix M after removing the contribution of eigvec
    
    DESCRIPTION: Deflates the matrix M to find subsequent eigenvectors.
    """
    v = eigvec.astype(float)
    # Ensure normalized (for safety)
    norm = np.sqrt(np.sum(v * v))
    if norm == 0.0:
        return M.copy()
    v = v / norm

    # Rayleigh quotient for eigenvalue lambda
    Mv = np.dot(M, v)
    denom = np.dot(v, v)
    if denom == 0.0:
        lam = 0.0
    else:
        lam = np.dot(v, Mv) / denom

    # Deflation: M_new = M - lambda * v v^T
    v_col = v.reshape(-1, 1)
    v_row = v.reshape(1, -1)
    vvT = np.dot(v_col, v_row)
    return M - lam * vvT


def compute_eigenvalues(M):
    """
    INPUT: matrix M
    OUTPUT: list of eigenvalues using power iteration for each eigenvector
    
    DESCRIPTION: Uses power iteration to find the dominant eigenvalue of M,
    deflates the matrix and finds subsequent eigenvalues.
    """
    A = M.astype(float).copy()
    n = A.shape[0]
    eigenvalues = []

    for _ in range(n):
        v = power_iteration(A)
        # Normalize (again, 안전하게)
        norm = np.sqrt(np.sum(v * v))
        if norm == 0.0:
            eigenvalues.append(0.0)
            break
        v = v / norm

        # Rayleigh quotient for eigenvalue
        Av = np.dot(A, v)
        lam = np.dot(v, Av) / np.dot(v, v)
        eigenvalues.append(lam)

        # Deflate for next eigenpair
        A = deflate_matrix(A, v)

    return np.array(eigenvalues, dtype=float)


def svd_manual(M):
    """
    INPUT: matrix M
    OUTPUT: matrices U, Sigma, V using SVD
    
    DESCRIPTION: Computes SVD by calculating eigenvalues and eigenvectors for M^T M.
    """
    M = M.astype(float)
    MTM, _ = matrix_mult(M)
    n = MTM.shape[0]

    # Power iteration + deflation to get eigenpairs of M^T M
    A = MTM.copy()
    eigvals = []
    eigvecs = []

    for _ in range(n):
        v = power_iteration(A)
        # Normalize
        norm = np.sqrt(np.sum(v * v))
        if norm == 0.0:
            v = np.zeros(n, dtype=float)
        else:
            v = v / norm

        # Eigenvalue w.r.t original MTM
        Av = np.dot(MTM, v)
        lam = np.dot(v, Av) / np.dot(v, v) if np.dot(v, v) != 0.0 else 0.0

        eigvals.append(lam)
        eigvecs.append(v)

        A = deflate_matrix(A, v)

    # 정렬: 큰 고유값 순서대로
    indices = list(range(n))
    indices.sort(key=lambda i: eigvals[i], reverse=True)

    eigvals_sorted = [eigvals[i] for i in indices]
    V = np.zeros((n, n), dtype=float)
    for col, idx in enumerate(indices):
        V[:, col] = eigvecs[idx]

    eigvals_arr = np.array(eigvals_sorted, dtype=float)
    # Singular values = sqrt(eigenvalues of M^T M)
    singular_vals = np.sqrt(np.maximum(eigvals_arr, 0.0))

    # Compute U from M V = U Sigma
    m = M.shape[0]
    U = np.zeros((m, n), dtype=float)
    for i in range(n):
        sigma = singular_vals[i]
        if sigma > 1e-12:
            mv = np.dot(M, V[:, i])
            U[:, i] = mv / sigma
        else:
            U[:, i] = np.zeros(m, dtype=float)

    Sigma = np.diag(singular_vals)  # 2D array of singular values
    return U, Sigma, V


def matrix_approximation(U, Sigma, V, k):
    """
    INPUT: matrices U, Sigma, V, and integer k
    OUTPUT: matrix approximation with only the top k singular values
    
    DESCRIPTION: Computes the matrix approximation using the top k singular values.
    """
    # 1) Sigma가 2D 대각행렬이면 대각선만 뽑고,
    #    1D 벡터이면 그대로 사용
    if Sigma.ndim == 2:
        singular_vals = np.diag(Sigma).astype(float)  # (n,)
    else:
        singular_vals = np.array(Sigma, dtype=float)  # (n,)

    # 2) 상위 k개의 특이값으로 k×k 대각행렬 만들기
    Sigma_k = np.diag(singular_vals[:k])   # (k, k)

    # 3) U, V에서 상위 k개 열만 사용
    U_k = U[:, :k]    # (n, k)
    V_k = V[:, :k]    # (n, k)

    # 4) M_k = U_k Σ_k V_k^T
    US = np.dot(U_k, Sigma_k)   # (n, k)
    M_k = np.dot(US, V_k.T)     # (n, n)

    return M_k

def energy_retained(Sigma, k):
    """
    INPUT: list of singular values, integer k
    OUTPUT: percentage of energy retained by the k-dimensional approximation
    
    DESCRIPTION: Computes the percentage of energy retained by the top k singular values.
    The energy is the sum of the squares of the singular values.
    """
    singular_vals = np.diag(Sigma).astype(float)

    total_energy = np.sum(singular_vals * singular_vals)
    if total_energy == 0.0:
        return 0.0

    retained_energy = np.sum(singular_vals[:k] * singular_vals[:k])
    return retained_energy / total_energy

def pca_via_svd(M, k):
    """
    INPUT: matrix M, integer k
    OUTPUT: top-k PCA projection of M
    """
    X= M.astype(float)
    mean= np.mean(X,axis=0, keepdims= True)
    X_centered = X - mean
    U, Sigma, V = svd_manual(X_centered)
    # For PCA, project rows of M onto top-k right singular vectors (columns of V)
    V_k = V[:, :k]              # (d x k)
    M_pca = np.dot(X_centered, V_k)      # (n x k)
    return M_pca


def distance_correlation(M, M_reduced):
    """
    INPUT: original matrix M and reduced matrix M_reduced
    OUTPUT: distance correlation between pairwise distances in M and M_reduced
    """
    M = np.array(M, dtype=float)
    M_reduced = np.array(M_reduced, dtype=float)

    n = M.shape[0]
    # Pairwise distance matrices
    D = np.zeros((n, n), dtype=float)
    Dk = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            diff = M[i] - M[j]
            D[i, j] = np.sqrt(np.sum(diff * diff))
            diff_k = M_reduced[i] - M_reduced[j]
            Dk[i, j] = np.sqrt(np.sum(diff_k * diff_k))

    # Means
    D_bar = np.sum(D) / (n * n)
    Dk_bar = np.sum(Dk) / (n * n)

    # Numerator and denominators
    num = 0.0
    den1 = 0.0
    den2 = 0.0
    for i in range(n):
        for j in range(n):
            a = D[i, j] - D_bar
            b = Dk[i, j] - Dk_bar
            num += a * b
            den1 += a * a
            den2 += b * b

    if den1 == 0.0 or den2 == 0.0:
        return 0.0

    dist_corr = num / np.sqrt(den1 * den2)
    return dist_corr


if __name__ == "__main__":
    # Given matrix M
    M = np.array([
        [8, 2, 1, 0, 2],
        [2, 3, 0, 6, 0],
        [1, 0, 4, 0, 0],
        [0, 6, 0, 8, 1],
        [2, 0, 0, 1, 7]
    ], dtype=float)

    # ----------------------------------------------
    #           Please do not modify below
    # ----------------------------------------------
    with open("output2.txt", "w") as f:
        # Part (a): Compute M^T M and M M^T
        print("\n(a) M^T M and M M^T:", file=f)
        MTM, MMT = matrix_mult(M)
        print("M^T M:\n", MTM, file=f)
        print("M M^T:\n", MMT, file=f)
        
        # Part (b): Eigenpairs using numpy.linalg.eig()
        print("\n(b) Power iteration vs numpy.linalg.eig():", file=f)
        eigvals_MTM, eigvecs_MTM = np.linalg.eig(MTM)
        eigvals_MMT, eigvecs_MMT = np.linalg.eig(MMT)
        print("Eigenvalues of M^T M:\n", eigvals_MTM, file=f)
        print("Eigenvectors of M^T M:\n", eigvecs_MTM, file=f)

        eigenvalues_manual_MTM = compute_eigenvalues(MTM)
        eigenvalues_manual_MMT = compute_eigenvalues(MMT)
        print("Eigenvalues of M^T M using power iteration:\n", eigenvalues_manual_MTM, file=f)
        print("Eigenvalues of M M^T using power iteration:\n", eigenvalues_manual_MMT, file=f)

        # Part (c): SVD implementation
        print("\n(c) SVD implementation:", file=f)
        U, Sigma, V = svd_manual(M)
        print("U:\n", U, file=f)
        print("Sigma:\n", Sigma, file=f)
        print("V:\n", V, file=f)

        # Part (d): Rank-k approximations
        print("\n(d) Rank-k approximations:", file=f)
        for k in range(1, len(Sigma) + 1):
            M_k = matrix_approximation(U, Sigma, V, k)
            print(f"\nRank-{k} approximation of M:\n", np.round(M_k, 3), file=f)

        # Part (e): Energy retention
        print("\n(e) Energy retention ratios:", file=f)
        for k in range(1, len(Sigma) + 1):
            energy = energy_retained(Sigma, k)
            print(f"  k = {k}: {energy * 100:.2f}% energy retained", file=f)
        
        # Part (f): PCA via SVD
        print("\n(f) PCA via SVD:", file=f)
        for k in range(1, 6):
            M_pca = pca_via_svd(M, k)
            print(f"\nTop-{k} PCA projection of M:\n", np.round(M_pca, 3), file=f)

        # Part (g): Top-k PCA projection vs. Random projection (Distance Correlation)
        print("\n(g) Top-k PCA projection vs. Random projection (Distance Correlation):", file=f)
        np.random.seed(42)  # For reproducibility
        for k in range(1, 6):
            M_pca = pca_via_svd(M, k)
            dist_corr = distance_correlation(M, M_pca)
            M_random = M @ np.random.randn(M.shape[1], k)
            dist_corr_random = distance_correlation(M, M_random)
            print(f"  k = {k}: PCA Dist Corr = {dist_corr:.4f}, Random Dist Corr = {dist_corr_random:.4f}", file=f)
