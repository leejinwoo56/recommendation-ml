import sys
import math
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

def dist(x, y):
    """
    INPUT: two points x and y
    OUTPUT: the Euclidean distance between two points x and y

    DESCRIPTION: Returns the Euclidean distance between two points.
    """
    s = 0.0
    for a, b in zip(x, y):
        d = a - b
        s += d * d
    return math.sqrt(s)


def parse_line(line):
    """
    INPUT: one line from input file
    OUTPUT: parsed line with numerical values
    
    DESCRIPTION: Parses a line to coordinates.
    """
    return tuple(float(t) for t in line.strip().split())


def pick_points(k):
    """
    INPUT: value of k for k-means algorithm
    OUTPUT: the list of initial k centroids.

    DESCRIPTION: Picks the initial cluster centroids for running k-means.
    """
    data_path = sys.argv[1]
    points = sc.textFile(data_path).map(parse_line).collect()
    k = min(k, len(points))
    centers = [points[0]]
    min_d = [dist(p, centers[0]) for p in points]
    while len(centers) < k:
        idx = max(range(len(points)), key=lambda i: min_d[i])
        c = points[idx]
        centers.append(c)
        for i, p in enumerate(points):
            d = dist(p, c)
            if d < min_d[i]:
                min_d[i] = d
    return centers


def assign_cluster(centroids, point):
    """
    INPUT: list of centorids and a point
    OUTPUT: a pair of (closest centroid, given point)

    DESCRIPTION: Assigns a point to the closest centroid.
    """
    best_i = 0
    best_d = dist(point, centroids[0])
    for i in range(1, len(centroids)):
        d = dist(point, centroids[i])
        if d < best_d:
            best_d = d
            best_i = i
    return (best_i, point)


def compute_diameter(cluster):
    """
    INPUT: cluster
    OUTPUT: diameter of the given cluster

    DESCRIPTION: Computes the diameter of a cluster.
    """
    m = len(cluster)
    if m <= 1:
        return 0.0
    max_d = 0.0
    for i in range(m):
        pi = cluster[i]
        for j in range(i + 1, m):
            d = dist(pi, cluster[j])
            if d > max_d:
                max_d = d
    return max_d


def kmeans(centroids):
    """
    INPUT: list of centroids
    OUTPUT: average diameter of the clusters
    """
    data_path = sys.argv[1]

    points_rdd = sc.textFile(data_path).map(parse_line).cache()
    k = len(centroids)

    # 1) 초기 centroids 기준으로 "한 번만" assign
    bc = sc.broadcast(centroids)
    grouped = (
        points_rdd
        .map(
            lambda p: (
                min(
                    range(k),
                    key=lambda i: dist(p, bc.value[i])
                ),
                p
            )
        )
        .groupByKey()
        .mapValues(list)
        .collect()
    )
    bc.unpersist()

    # 2) 클러스터 리스트 만들기 (비어 있는 클러스터도 포함)
    cluster_lists = [list() for _ in range(k)]
    for cid, lst in grouped:
        if 0 <= cid < k:
            cluster_lists[cid] = lst

    # 3) 각 클러스터 diameter 계산 후 평균
    diameters = [compute_diameter(lst) for lst in cluster_lists]
    if k == 0:
        return 0.0
    return sum(diameters) / float(k)




if __name__ == "__main__":
    # ----------------------------------------------
    #             Please do not modify below
    # ----------------------------------------------
    with open('output1.txt', 'w') as f:
        k_value = int(sys.argv[2])
        centroids = pick_points(k_value)
        f.write('k=%d\n' % (k_value))
        f.write('Initial centroids: %s\n' % str(centroids))
        average_diameter = kmeans(centroids)
        f.write('Average diameter: %f\n' % average_diameter)
