import numpy as np
from knn import KNN


def f1_score(real_labels, predicted_labels):
    """
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    
    f1_score = float(2*sum(np.multiply(real_labels, predicted_labels)))/(sum(real_labels) + sum(predicted_labels))
    return f1_score


class Distances:
    @staticmethod
    def minkowski_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        distance = np.power(sum(np.power(np.absolute(np.subtract(point1, point2)), 3)), (1/3))
        return distance
        
    @staticmethod
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        distance = np.power(np.inner(np.subtract(point1, point2), np.subtract(point1, point2)), (1/2))
        return distance
        
    @staticmethod
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        distance = np.inner(point1, point2)
        return distance
    
    @staticmethod
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        distance = 1-np.dot(point1, point2)/(np.linalg.norm(point1)*np.linalg.norm(point2))
        return distance
        
    @staticmethod
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        distance = 1/(np.exp((1/2)*np.inner(np.subtract(point1, point2), np.subtract(point1, point2))))
        return distance
    

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # Find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        Tried different distance function implemented above, and found the best k.
        Used k ranging from 1 to 30 and incremented by 2. Used f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions used to calculate the distance.
        :param x_train: List[List[int]] training data set to train KNN model
        :param y_train: List[int] train labels to train KNN model
        :param x_val:  List[List[int]] Validation data set will be used on KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Found(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, chose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, chose model which has a less k.
        """
        f1score_and_model = []
        
        for k in range(1, 30, 2):
            for d in distance_funcs:
                knn_model = KNN(k, distance_funcs[d])
                knn_model.train(x_train, y_train)
                y_val_predict = knn_model.predict(x_val)
                f1_score_value = f1_score(y_val, y_val_predict)
                f1score_and_model.append((f1_score_value, d, knn_model))
                
        f1score_and_model.sort(key= lambda x, list=['euclidean', 'minkowski', 'gaussian', 'inner_prod',
                        'cosine_dist']: (x[0], list[::-1].index(x[1]), (-1)*x[2].k), reverse=True)

        self.best_k = f1score_and_model[0][2].k
        self.best_distance_function = f1score_and_model[0][1]
        self.best_model = f1score_and_model[0][2]

    # Find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance functions used to calculate the distance.
        :param scaling_classes: dictionary of scalers used to normalized data.
        :param x_train: List[List[int]] training data set to train KNN model
        :param y_train: List[int] train labels to train KNN model
        :param x_val: List[List[int]] validation data set used on KNN predict function to produce predicted
            labels and tune k, distance function and scaler.
        :param y_val: List[int] validation labels

        Found(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, chose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, chose model which has a less k.
        """
        f1score_and_model = []
        
        for s in scaling_classes:
            scaler = scaling_classes[s]()
            x_train_scaled = scaler(x_train)
            x_val_scaled = scaler(x_val)
            for k in range(1, 30, 2):
                for d in distance_funcs:
                    knn_model = KNN(k, distance_funcs[d])
                    knn_model.train(x_train_scaled, y_train)
                    y_val_predict = knn_model.predict(x_val_scaled)
                    f1_score_value = f1_score(y_val, y_val_predict)
                    f1score_and_model.append((f1_score_value, s, d, knn_model))
                    
        f1score_and_model.sort(key= lambda x, list_s = ['min_max_scale', 'normalize'], list_d=['euclidean', 
                        'minkowski', 'gaussian', 'inner_prod','cosine_dist']: (x[0], list_s[::-1].index(x[1]), 
                        list_d[::-1].index(x[2]), (-1)*x[3].k), reverse=True)

        self.best_k = f1score_and_model[0][3].k
        self.best_distance_function = f1score_and_model[0][2]
        self.best_scaler = f1score_and_model[0][1]
        self.best_model = f1score_and_model[0][3]


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
        Normalize features for every sample
        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized_features = []
        for i in range(len(features)):
            normalized = [x/(np.power(np.inner(features[i],features[i]), 1/2)) if x!=0 else x*0 for x in features[i]]
            normalized_features.append(normalized)
        return normalized_features

class MinMaxScaler:
    """
    Note: Assumed the parameters are valid when __call__
          is being called the first time (found min and max).
    """

    def __init__(self):
        self.call_count=0
        pass

    def __call__(self, features):
        """
        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features)
        if(self.call_count==0):
            self.minimum = [min(features[:,i]) for i in range(len(features.T))]
            self.maximum = [max(features[:,i]) for i in range(len(features.T))]
            self.call_count += 1         
        
        features_scaled = []
        for i in range(len(features.T)):
            f = [(x-self.minimum[i])/(self.maximum[i]-self.minimum[i]) for x in features[:,i]]
            features_scaled.append(f)
            
        features_scaled = (np.array(features_scaled).T).tolist()
        return features_scaled
                
