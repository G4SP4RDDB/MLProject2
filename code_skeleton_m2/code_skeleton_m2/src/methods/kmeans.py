import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """

        ### WRITE YOUR CODE HERE
        if K <= 0: 
            raise ValueError(f"K must be a positive integer, we need at least one cluster")
        if max_iters < 0:
            raise ValueError(f"max_iters must be a non-negative integer, we cannot do negative iterations")
        
        self.K = K
        self.max_iters = max_iters
        self.centers = None
        self.cluster_center_label = None
        
        # if self.K <= 0:
        #     print(f"K was clamped to 1. It was {self.K} but we need at least one cluster")
        #     self.K = 1
        # if self.max_iters < 0:
        #     print(f"max_iters was clamped to 0. It was {self.max_iters} but we cannot do negative iterations")
        #     self.max_iters = 0

    def init_centers(self, data):
        """
        Randomly pick K data points from the data as initial cluster centers.

        Arguments:
            data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """

        ### WRITE YOUR CODE HERE
        N = data.shape[0]
        if self.K > N:
            raise ValueError(f"K cannot be larger than the number of data points. It was {self.K} but there are only {N} data points.")
            # print(f"K was clamped to the number of data points. It was {self.K} but there are only {N} data points.")
            # self.K = N
        
        indices = np.random.choice(N, self.K, replace=False)
        centers = data[indices]
        return centers

    def compute_distance(self, data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.

        Arguments:
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """

        ### WRITE YOUR CODE HERE
        N = data.shape[0]

        # distances = np.zeros((data.shape[0], self.K))
        # for n in range(N):
        #     for k in range(self.K):
        #         distances[n, k] = np.linalg.norm(data[n] - centers[k])

        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        return distances

    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.

        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """

        ### WRITE YOUR CODE HERE
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments

    def compute_centers(self, data, cluster_assignments):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments:
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """

        ### WRITE YOUR CODE HERE
        N = data.shape[0]
        centers = np.zeros((self.K, data.shape[1]))

        for k in range(self.K):
            k_cluster_points = data[cluster_assignments == k]
            if len(k_cluster_points) > 0:
                centers[k] = np.mean(k_cluster_points, axis=0)
            else: #can this even happen?
                centers[k] = data[np.random.randint(0, N)] #take random new center?

        return centers

    def k_means(self, data):
        """
        Main K-Means algorithm that performs clustering of the data.

        Arguments:
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """

        ### WRITE YOUR CODE HERE
        centers = self.init_centers(data)
        centers_prev = None
        distances = self.compute_distance(data, centers)
        cluster_assignments = self.find_closest_cluster(distances)

        for i in range(self.max_iters):
            distances = self.compute_distance(data, centers)
            cluster_assignments = self.find_closest_cluster(distances)
            centers = self.compute_centers(data, cluster_assignments)

            if (centers_prev is not None) and np.allclose(centers_prev, centers):
                break
            centers_prev = centers.copy()

        return centers, cluster_assignments

    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments:
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns:
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """

        ### WRITE YOUR CODE HERE
        cluster_center_label = np.zeros(self.K, dtype=true_labels.dtype)

        for k in range(self.K):
            kth_cluster_labels = true_labels[cluster_assignments == k]

            if len(kth_cluster_labels) > 0:
                labels, counts = np.unique(kth_cluster_labels, return_counts=True)
                cluster_center_label[k] = labels[np.argmax(counts)]
            else: #can this even happen?
                labels, counts = np.unique(true_labels, return_counts=True)
                cluster_center_label[k] = labels[np.argmax(counts)] #take most common label?
        
        return cluster_center_label

    def predict_with_centers(self, data, centers, cluster_center_label):
        """
        Predict the label for data, given the cluster center and their labels.
        To do this, it first assign points in data to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            data: array of shape (N, D)
            centers: array of shape (K, D), cluster centers
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        Returns:
            new_labels: array of shape (N,), the labels assigned to each data point after clustering, via k-means.
        """

        ### WRITE YOUR CODE HERE
        N = data.shape[0]
        # new_labels = np.zeros(N)
        # for n in range(N):
        #     distances = np.linalg.norm(data[n] - centers, axis=1)
        #     closest_cluster = np.argmin(distances)
        #     new_labels[n] = cluster_center_label[closest_cluster]

        distances = self.compute_distance(data, centers)
        closest_clusters = np.argmin(distances, axis=1)
        new_labels = cluster_center_label[closest_clusters]

        return new_labels

    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        ### WRITE YOUR CODE HERE
        self.centers, cluster_assignments = self.k_means(training_data)
        self.cluster_center_label = self.assign_labels_to_centers(self.centers, cluster_assignments, training_labels)
        pred_labels = self.predict_with_centers(training_data, self.centers, self.cluster_center_label)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        ### WRITE YOUR CODE HERE
        if self.centers is None or self.cluster_center_label is None:
            raise ValueError("Model has not been fitted yet")
        pred_labels = self.predict_with_centers(test_data, self.centers, self.cluster_center_label)
        return pred_labels