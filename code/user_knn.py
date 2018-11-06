"""
KNN calculation with dictionary based approach with 3 different similarity algorithm option

"""

from code.user import User


class UKNN:
    def __init__(self):
        self.K = 0
        self.score = 0.0
        self.score_nw = 0.0
        self.similarity_function = self.cos_sim
        self.threshold = 0

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

    def calculate_score(self, test, pred, pred_nw):
        score, score_nw = 0, 0
        divisor = 1
        for book in test.books.keys():
            if book not in pred.books:
                continue
            score += (abs(float(test.books[book] - pred.books[book])))
            score_nw += (abs(float(test.books[book] - pred_nw.books[book])))
            divisor += 1
        self.score += (score / divisor)
        self.score_nw += (score_nw / divisor)

    """ --------- CORRELATION SIMILARITY FUNCTION --------- """
    def cor_dot(self, d1, d2):
        total = 0.0
        avg_d1, avg_d2 = d1.avg, d2.avg
        for ind in set(d1.books).intersection(d2.books):
            total += ((d1.books[ind] - avg_d1) * (d2.books[ind] - avg_d2))
        return total

    def cor_sim(self, s_user, test):
        norm = s_user.corr_norm * test.corr_norm
        return self.cor_dot(s_user, test) / norm if norm != 0 else 0.0
    """ ------------------------- CORRELATION END ------------------------- """

    """ --------- COSINE SIMILARITY FUNCTION --------- """
    def cos_dot(self, d1, d2):
        total = 0.0
        for ind in set(d1.books).intersection(d2.books):
            total += (d1.books[ind] * d2.books[ind])
        return total

    def cos_sim(self, s_user, test):
        norm = s_user.norm * test.norm
        return self.cos_dot(s_user, test) / norm if norm != 0 else 0.0
    """ ------------------------- COSINE END ------------------------- """

    """ --------- ADJUSTED COSINE SIMILARITY FUNCTION --------- """
    def adj_cos_dot(self, d1, d2):
        total = 0.0
        for ind in set(d1.books).intersection(d2.books):
            total += ((d1.books[ind] - self.u_book_list[ind].avg) * (d2.books[ind] - self.u_book_list[ind].avg))
        return total

    def adj_cos_sim(self, s_user, test):
        norm = s_user.adj_norm * test.adj_norm
        return self.adj_cos_dot(s_user, test) / norm if norm != 0 else 0.0
    """ ------------------------- END ------------------------- """

    def calc_similarities(self, test):
        similar_users = []
        for book in test.books:
            similar_users.extend(self.u_book_list[book].users.keys())

        out = [(self.u_user_list.get(user), self.similarity_function(self.u_user_list.get(user), test)) for user in similar_users
               if self.u_user_list.get(user) is not None]
        return sorted(out, key=lambda arr: arr[1], reverse=True)[:self.K]

    def calc_rating(self, sims, test):
        pred = User("prediction")
        pred_nw = User('non weight prediction')
        sim_sum = 0

        for user, sim in sims:
            sim_sum += abs(sim)

            for book, rating in user.books.items():

                # filter out books rated by user count less than threshold
                if self.u_book_list[book].user_count < self.threshold:
                    rating = test.avg

                # sum ratings
                if book not in pred.books.keys():
                    pred_nw.append_book(book, rating)
                    pred.append_book(book, (rating * abs(sim)))
                else:
                    pred_nw.books[book] += rating
                    pred.books[book] += (rating * abs(sim))

        # mean
        for book in pred.books.keys():
            pred_nw.books[book] /= self.K
            pred.books[book] /= (sim_sum if sim_sum != 0 else self.K)

        return pred, pred_nw

    def calc_nearest_n(self, test):
        sims = self.calc_similarities(test)
        pred, pred_nw = self.calc_rating(sims, test)
        self.calculate_score(test, pred, pred_nw)

    def fit(self, data, test, threshold=0, k=3):
        self.K = 0
        self.score = 0.0
        self.score_nw = 0.0

        self.K = k
        self.threshold = threshold

        self.u_user_list = data[0]
        self.t_user_list = test[0]

        self.u_book_list = data[1]
        self.t_book_list = test[1]

        for test_user in test[0].values():
            self.calc_nearest_n(test_user)

        self.score /= len(test[0])
        self.score_nw /= len(test[0])
