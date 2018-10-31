from math import sqrt
from user import User
import time


class UKNN:
    def __init__(self):
        self.K = 0
        self.score = 0.0
        self.similarity_function = self.adj_cos_sim

        """ 
        u_user_list: list of user objects from training data
        t_user_list: list of user object from test data
        
        """
        self.u_user_list = {}
        self.t_user_list = {}

        """
        u_book_list: list of book objects from training data
        t_book_list: list of book objects from test data
        
        """
        self.u_book_list = {}
        self.t_book_list = {}

    def calculate_score(self, test, pred):
        score = 0
        divisor = 1
        for book in test.books.keys():
            if book not in pred.books:
                continue
            score += (abs(float(test.books[book] - pred.books[book])))
            divisor += 1
        self.score += (score / divisor)

    """ --------- CORRELATION SIMILARITY FUNCTION --------- """

    def cor_dot(self, d1, d2):
        total = 0.0
        avg_d1, avg_d2 = d1.avg, d2.avg
        for ind in set(d1.books).intersection(d2.books):
            total += ((d1.books[ind] - avg_d1) * (d2.books[ind] - avg_d2))
        return total

    def cor_norm(self, d):
        total = 0
        avg_d = d.avg
        for rating in d.books.values():
            total += ((rating - avg_d) ** 2)
        return sqrt(total)

    def cor_sim(self, darr, test):
        dp = self.cor_dot(darr, test)
        return dp / (self.cor_norm(darr) * self.cor_norm(test)) if dp != 0 else 0.0

    """ ------------------------- CORRELATION END ------------------------- """

    """ --------- COSINE SIMILARITY FUNCTION --------- """

    """ dot products two dictionary based on keys """
    def cos_dot(self, d1, d2):
        total = 0.0
        for ind in set(d1.books).intersection(d2.books):
            total += (d1.books[ind] * d2.books[ind])
        return total

    """ calculates norm from a dictionary """
    def cos_norm(self, d):
        total = 0
        for rating in d.books.values():
            total += (rating ** 2)
        return sqrt(total)

    def cos_sim(self, darr, test):
        dp = self.cos_dot(darr, test)
        return dp / (self.cos_norm(darr) * self.cos_norm(test)) if dp != 0 else 0.0

    """ ------------------------- COSINE END ------------------------- """

    """ --------- ADJUSTED COSINE SIMILARITY FUNCTION --------- """

    def adj_cos_dot(self, d1, d2):
        total = 0.0
        for ind in set(d1.books).intersection(d2.books):
            total += ((d1.books[ind] - self.u_book_list[ind].avg) * (d2.books[ind] - self.u_book_list[ind].avg))
        return total

    def adj_cos_norm(self, d):
        total = 0
        for ind, rating in d.books.items():
            total += ((rating - self.u_book_list[ind].avg) ** 2)
        return sqrt(total)

    def adj_cos_sim(self, darr, test):
        dp = self.adj_cos_dot(darr, test)
        return dp / (self.adj_cos_norm(darr) * self.adj_cos_norm(test)) if dp != 0 else 0.0

    """ ------------------------- END ------------------------- """

    def calc_similarities(self, data, test):
        out = [(user, self.similarity_function(user, test)) for user in data.values()]
        return sorted(out, key=lambda arr: arr[1], reverse=True)[:self.K]

    def calc_rating(self, sims):
        pred = User("prediction")
        sim_sum = 0

        for user, sim in sims:
            sim_sum += sim
            for book, rating in user.books.items():
                if book not in pred.books.keys():
                    pred.append_book(book, (rating * sim))
                else:
                    pred.books[book] += (rating * sim)
        for book in pred.books.keys():
            pred.books[book] /= (sim_sum if sim_sum != 0 else 1)
        return pred

    def calc_nearest_n(self, data, test):
        sims = self.calc_similarities(data, test)
        pred = self.calc_rating(sims)
        self.calculate_score(test, pred)

    def fit(self, k, data, test):
        self.K = k
        self.u_user_list = data[0]
        self.t_user_list = test[0]

        self.u_book_list = data[1]
        self.t_book_list = test[1]

        for t in test[0].values():
            self.calc_nearest_n(data[0], t)

        self.score /= len(test[0])
