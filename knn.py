import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.train_features = features
        self.train_labels = labels

    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, I process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predict label for that testing data point.
        Thus, got N predicted label for N test data point.
        This function returns a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        test_features = features
        test_labels = []
        for i in range(len(test_features)):
            k_neighbors = Counter(self.get_k_neighbors(test_features[i]))
            majority_label = k_neighbors.most_common(1)[0][0]
            test_labels.append(majority_label)
        return test_labels

    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        We already have the k value, distance function and stored all training data in KNN class with the
        train function. This function returns a list of labels of all k neighours.
        :param point: List[float]
        :return:  List[int]
        """
        dist_list = []
        for i in range(len(self.train_features)):
            dist_list.append((self.distance_function(point, self.train_features[i]), self.train_labels[i]))
            
        distances = np.array(dist_list, dtype=({'names':('distance', 'label'), 'formats':(float, int)}))
        distances = np.sort(distances, order='distance')
        return distances['label'][:self.k]

if __name__ == '__main__':
    print(np.__version__)
