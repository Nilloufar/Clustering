import matplotlib.pyplot as plt
from matplotlib import style
import random as random
import numpy as np
import math
import sklearn.datasets as ds
from sklearn.datasets import make_blobs


class clustering:
    def __init__(self, data, k, colors, kmeans=True, spectralKmeans=False, showPlot=False,centroids=[]):
        self.data = data
        self.originalData = data[:]
        self.k = k
        self.showPlot = showPlot
        if kmeans:
            self.title = "k-means"
        elif spectralKmeans:
            self.title = "spectral relaxed k-means"
            X = data
            X_transpose = X.transpose()
            Y = np.matmul(X, X_transpose)
            U, S, V = np.linalg.svd(Y)
            self.data = U[:, 0:k]
        self.SSE_iterations = []
        self.colors = colors
        n = self.data.shape[0]
        if centroids==[]:
            if kmeans:
                centroids = self.data[np.random.randint(0, n, k), :]
            elif spectralKmeans:
                centroids = self.data[np.random.randint(0, n, k), :]
        iterationNum = 0
        SSE = []
        while True:
            iterationNum += 1
            clusters = []
            previous_centroids = centroids[:]
            for dataPoint in self.data:
                distance = []
                for c in centroids:
                    distance.append(math.sqrt(sum((dataPoint - c) ** 2)))
                clusters.append(np.argmin(distance))
            # update centroids
            centroids = []
            for cl in range(k):
                cluster_data_index = [i for i, c in enumerate(clusters) if c == cl]
                centroids.append(np.mean(self.data[cluster_data_index, :], axis=0))
            centroids = np.vstack(centroids)
            # check whether the centroids changed
            if k == sum([center in centroids for center in previous_centroids]):
                break

            self.iterations = iterationNum
            self.clusters = clusters
            self.centroids = centroids
            self.SSE_iterations.append(self.get_SSE())
        if self.showPlot:
            self.plot_SSE()
        if self.showPlot:
            self.plot_clusters(self.iterations)

    def get_SSE(self):
        k = self.k
        clusters = self.clusters
        data = self.originalData
        centroids = self.centroids
        SSE = 0
        for cl in range(k):
            cluster_data_index = [i for i, c in enumerate(clusters) if c == cl]
            cluster_points = data[cluster_data_index, :]
            # c = centroids[cl]
            if cluster_points.shape[0]!=0:
                c=cluster_points.sum(axis=0)/cluster_points.shape[0]
            else:
                c=[0 for i in range(0,k)]
            SSE = SSE + sum(
                [sum((dataPoint - c) ** 2) for dataPoint in cluster_points])
        return math.sqrt(SSE)

    def plot_clusters(self, i=-1):
        X = self.originalData
        clusters = self.clusters
        centroids = self.centroids

        # colors = ['#4EACC5', '#FF9C34', '#4E9A06']
        colors = self.colors
        plt.figure(1)
        fig, ax = plt.subplots()
        style.use('ggplot')
        for j in range(self.k):
            col = colors[j]
            cluster_data_index = [i for i, c in enumerate(clusters) if c == j]
            plt.scatter(X[cluster_data_index, 0], X[cluster_data_index, 1],
                        c=col, marker='.', s=10)
            # plt.scatter(centroids[j, 0], centroids[j, 1], c=col, s=50)
        if i != -1:
            ax.set_title('{} iteration {}'.format(self.title, i), fontsize=15)
        plt.show()

    def plot_SSE(self):
        SSE = self.SSE_iterations
        plt.plot([i for i in range(self.iterations)], SSE)
        plt.xlabel('iteration')
        plt.ylabel('SSE')
        plt.title('Sum of Mean Squeare in each Iteration \n ({})'.format(self.title))
        plt.grid(True)
        plt.savefig("kmeans-SSE.png")
        plt.show()


# create data ponts
def colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r, g, b))
    return ret


#  main program starts:
# --------------------------
# parameters
k = 3
colors = [np.random.rand(3, ) for i in range(k)]
showPlot = True
# --------------------------
# data :
X, y = make_blobs(n_samples=300,n_features=5, centers=k, cluster_std=0.60, random_state=0)

clusters = y


# colors = ['#4EACC5', '#FF9C34', '#4E9A06']
plt.figure(1)
fig, ax = plt.subplots()
style.use('ggplot')
for j in range(k):
    col = colors[j]
    cluster_data_index = [i for i, c in enumerate(clusters) if c == j]
    plt.scatter(X[cluster_data_index, 0], X[cluster_data_index, 1],
                c=col, marker='.', s=10)
    ax.set_title('data samples in three clusters')
plt.show()





# --------------------------

clustering(X, k, colors, kmeans=True, spectralKmeans=False, showPlot=showPlot)

# spectral clustering:


clustering(X, k, colors, kmeans=False, spectralKmeans=True, showPlot=showPlot)

num = 10
mean_sse_relaxed = 0
mean_sse_kmeans = 0
mean_sse_iteration_relaxed = 0
mean_sse_iteration_kmeans = 0
for i in range(num):
    # centroids = X[np.random.randint(0, X.shape[0], k), :]
    centroids=[]
    relaxed = clustering(X, k, colors, kmeans=False, spectralKmeans=True,centroids=centroids)
    kmeans = clustering(X, k, colors, kmeans=True, spectralKmeans=False,centroids=centroids)
    mean_sse_relaxed += relaxed.SSE_iterations[relaxed.iterations - 1]
    mean_sse_kmeans += kmeans.SSE_iterations[kmeans.iterations - 1]
    mean_sse_iteration_relaxed += relaxed.iterations
    mean_sse_iteration_kmeans += kmeans.iterations

print("results: mean_sse_relaxed: {} mean_sse_kmeans: {} mean_sse_iteration_relaxed: {} mean_sse_iteration_kmeans {}"
      .format(mean_sse_relaxed / num, mean_sse_kmeans / num, mean_sse_iteration_relaxed / num,
              mean_sse_iteration_kmeans / num))
