import random
import math
import matplotlib.pyplot as plt
import sys

class KMeans:
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.centroids = self.initialize_centroids_plusplus()

    @staticmethod
    def euclidean_distance(point1, point2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    def initialize_centroids(self):
        random.seed(0)
        return random.sample(self.data, self.k)

    def initialize_centroids_plusplus(self):
        random.seed(0)
        centroids = [random.choice(self.data)]

        while len(centroids) < self.k:
            distances = [min(self.euclidean_distance(point, c) for c in centroids) for point in self.data]
            probabilities = [dist / sum(distances) for dist in distances]
            next_centroid = random.choices(self.data, probabilities)[0]
            centroids.append(next_centroid)

        return centroids

    def assign_to_clusters(self):
        clusters = [[] for _ in range(self.k)]
        for point in self.data:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)
        return clusters

    def update_centroids(self, clusters):
        self.centroids = [self.calculate_centroid(cluster) for cluster in clusters]

    def calculate_centroid(self, cluster):
        if not cluster:
            return random.choice(self.data)
        centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
        return centroid

    def kmeans(self, max_iterations=20):
        for _ in range(max_iterations):
            clusters = self.assign_to_clusters()  
            old_centroids = self.centroids
            self.update_centroids(clusters)
            if all(self.euclidean_distance(old, new) < 0.001 for old, new in zip(old_centroids, self.centroids)):
                break

        return clusters

    def calculate_error(self, clusters):
        total_error = sum(sum(self.euclidean_distance(point, self.centroids[i]) for point in cluster) for i, cluster in enumerate(clusters))
        return total_error

    def plot_error_vs_k(self, max_k=10):
        errors = []
        for k in range(2, max_k + 1):
            self.k = k
            self.centroids = self.initialize_centroids_plusplus()
            clusters = self.kmeans()
            error = self.calculate_error(clusters)
            errors.append(error)
            print(f"For value k = {k}, After Running 20 iterations: Sum of Error = {error:.4f}")


        plt.plot(range(2, max_k + 1), errors, marker='*',linestyle='-.', color='k')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Error')
        plt.title('K-chat: Error vs K for K-means Clustering')
        plt.show()

 
def read_dataset(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            values = line.strip().split()
            features = [float(val) for val in values[:-1]]
            data.append(features)
    return data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please use the following format: python filename.py <data_file.txt>")
        sys.exit(1)

    file_name = sys.argv[1]
    dataset = read_dataset(file_name)

    kmeans_instance = KMeans(2, dataset)
    kmeans_instance.plot_error_vs_k(10)