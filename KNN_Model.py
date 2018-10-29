"""
Calculating the nearest neighbor with cosine similarity
from numpy array

"""
import numpy as np


class KNN:

    def __init__(self):
        self.K = 1
        self.score = 0.0
        self.predict = []

    """ Calculating algorithm score based on predicted array """
    @staticmethod
    def calc_score(arr, pred):
        sum_score = 0
        indices = arr.nonzero()
        for i in indices[0]:
            sum_score += abs(float(arr[i] - pred[i]))
            # print("arri: ", arr[i], ", predi: ", pred[i], ", calc: ", abs(float(arr[i] - pred[i])))
        return sum_score / len(indices[0]) if len(indices[0]) != 0 else 0.0

    @staticmethod
    def calc_prediction(arr, data, distances):
        indices = arr.nonzero()
        simsum = sum(i for _, i in distances)
        base = (sum([arr + (data[i[0]] * i[1]) for i in distances]) / simsum if simsum != 0 else arr)
        for i in indices[0]:
            base[i] = 0.0
            counter = 0
            for ind, sim in distances:
                if data[ind][i] == 0:
                    continue
                base[i] += (data[ind][i] * sim)
                counter += sim
            base[i] = base[i] / counter if counter != 0 else base[i]
        return base

    # TODO: add other similarity functions too
    @staticmethod
    def sim(arr1, arr2):
        dp = np.dot(arr1, arr2)
        return dp / (np.linalg.norm(arr1) * np.linalg.norm(arr2)) if dp != 0 else 0.0

    def all_similarities(self, data, target):
        out = []

        for d in range(data.shape[0]):
            out.append((d, self.sim(data[d], target)))

        y = sorted(out, key=lambda arr: arr[1], reverse=True)
        return y

    """ 
    Weighted neares n calculator, weights are not distance based but similarity based
    Also calculates score based on predicted values
        
    """
    def calculate_nearest_n(self, data, target):
        # calculate similarities and slice first K
        similarities = self.all_similarities(data, target)[:self.K]

        # calculate the predicted score with weight as similarity
        # if nonzero flag is set to false users with no vote is also included as lowest score
        pred = self.calc_prediction(target, data, similarities)
        self.score += self.calc_score(target, pred)

        return pred

    def fit(self, k, data, target):
        self.K = k
        self.predict = np.array([self.calculate_nearest_n(data, t) for t in target])
        self.score /= len(target)
