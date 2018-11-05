from math import ceil

class CrossValidation:
    def __init__(self, min_k, max_k, data, knn):
        self.min_k = min_k
        self.max_k = max_k

        self.u_data = data[0]
        self.b_data = data[1]

        self.knn = knn

        self.kfold_score = 0
        self.kfold_nw_score = 0

    @staticmethod
    def array_split(arr, n):
        """ split an array into n clusters """
        return [arr[i: i + ceil(len(arr) / n)] for i in range(0, len(arr), ceil(len(arr) / n))]

    def dictionary_split(self, dic, n):
        """ split a dictionary into n clusters """
        return [{user: dic[user] for user in cluster} for cluster in self.array_split(list(dic.keys()), n)]

    def k_fold(self, fold, k):
        """ K-fold cross validation """
        clusters = self.dictionary_split(self.u_data, fold)

        for iteration in range(fold):
            raw_data = clusters.copy()
            test = raw_data.pop(iteration)

            data = {}
            for item in raw_data:
                data.update(item)

            self.knn.fit([data, self.b_data], [test, []], k=k)
            self.kfold_score += self.knn.score
            self.kfold_nw_score += self.knn.score_nw
            # print("Score for iteration", iteration, "-", self.knn.score)

        self.kfold_score /= fold
        self.kfold_nw_score /= fold
        print("K-fold weighted validation score for fold count", fold, "and k =", k, ": ", self.kfold_score)
        print("K-fold non-weighted validation score for fold count", fold, "and k =", k, ": ", self.kfold_nw_score)

    def validate(self, fold):
        for k_value in range(self.min_k, self.max_k):
            self.k_fold(fold, k_value)