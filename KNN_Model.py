"""
Calculating the nearest neighbor with cosine similarity
from numpy array

"""
import numpy as np


class KNN:
    """ TODO: some more extensive testing required, somehow some datas are accurate while some other is not,
        One idea is to add 1 to all rating data so there is a distinction between dislike and no vote"""
    def __init__(self):
        pass

    @staticmethod
    def det(arr):
        data = np.sqrt(np.sum(np.square(arr)))
        return data if data != 0 else 0.000001

    # TODO: if possible change this cosine similarity function to adjusted one
    @staticmethod
    def sim(arr1, arr2):
        return (np.dot(arr1, arr2)) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
        # return (np.dot(arr1, np.transpose(arr2))) / (self.det(arr1) * self.det(arr2))

    def all_similarities(self, data, target):
        out = []
        for d in range(data.shape[0]):
            out.append((d, self.sim(data[d], target)))
        y = sorted(out, key=lambda arr: arr[1], reverse=True)
        return y

    def calculate_nearest_n(self, k, data, target):
        """ TODO: add weight """
        similarities = self.all_similarities(data, target)[:k]
        out = np.zeros(data.shape[1])
        for i in similarities:
            out += data[i[0]]

        return out / k

    def fit(self, k, data, target):
        return np.array([self.calculate_nearest_n(k, data, t) for t in target])
