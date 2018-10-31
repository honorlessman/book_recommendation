class User:
    def __init__(self, user_id):
        self.books = {}
        self.book_count = 0
        self.uid = user_id
        self.avg = 0

    def append_book(self, book, rating):
        self.books[book] = rating
        self.book_count += 1

    def remove_book(self, book):
        return self.books.pop(book)[1]

    def calc_avg(self):
        self.avg = sum(self.books.values()) / len(self.books.keys())

    def update(self):
        self.calc_avg()
        self.book_count = len(self.books)
        return self
