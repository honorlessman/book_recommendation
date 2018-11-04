from math import sqrt


class User:
    def __init__(self, user_id):
        self.books = {}
        self.book_count = 0
        self.uid = user_id
        self.avg = 0

        self.norm = 0.000000001
        self.adj_norm = 0.000000001
        self.corr_norm = 0.000000001

        self.products = {}

    def append_book(self, book, rating):
        self.books[book] = rating
        self.book_count += 1

    def remove_book(self, book):
        return self.books.pop(book)[1]

    def calc_avg(self):
        self.avg = sum(self.books.values()) / len(self.books.keys())

    def calc_norm(self, books):
        for book in self.books:
            self.norm += self.books[book] ** 2
            self.adj_norm += (self.books[book] - books[book].avg) ** 2
            self.corr_norm += (self.books[book] - self.avg) ** 2

        self.norm = sqrt(self.norm)
        self.adj_norm = sqrt(self.adj_norm)
        self.corr_norm = sqrt(self.corr_norm)


    def update(self):
        self.calc_avg()
        self.book_count = len(self.books)
        return self
