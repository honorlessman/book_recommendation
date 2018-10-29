class User:
    def __init__(self, user_id):
        self.books = {}
        self.book_count = 0
        self.uid = user_id

    def append_book(self, book, rating):
        self.books[book] = rating
        self.book_count += 1

    def remove_book(self, book):
        return self.books.pop(book)[1]
