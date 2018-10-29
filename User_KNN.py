from math import sqrt


class UKNN:
    def __init__(self):
        self.K = 0
        self.score = 0.0

    """ finds common elements in two dictionary keys """
    @staticmethod
    def find_common(d1, d2):
        return set(d1).intersection(d2)

    """ dot products two dictionary based on keys """
    def ddot(self, d1, d2):
        total = 0.0
        for ind in self.find_common(d1.books, d2.books):
            total += (d1.books[ind] * d2.books[ind])
        return total

    """ calculates norm from a dictionary """
    def dnorm(self, d):
        total = 0
        for rating in d.books.values():
            total += (rating ** 2)
        return sqrt(total)

    def sim(self, darr, test):
        dp = self.ddot(darr, test)
        return dp / (self.dnorm(darr) * self.dnorm(test)) if dp != 0 else 0.0

    def calc_similarities(self, data, test):
        out = []
        for user in data.values():
            out.append((user, self.sim(user, test)))

        return sorted(out, key=lambda arr: arr[1], reverse=True)[:self.K]

    def calc_rating(self, data, sims):

        pass

    def calc_nearest_n(self, data, test):
        sims = self.calc_similarities(data, test)

        pred = self.calc_rating(data, sims)

    def fit(self, k, data, test):
        self.K = k
        self.score /= len(test)
        for t in test.values():
            self.calc_nearest_n(data, t)
