class Book:
    def __init__(self, isbn):
        self.isbn = isbn
        self.users = {}
        self.user_count = 0
        self.avg = 0

    def append_user(self, uid, score):
        self.users[uid] = score
        self.user_count += 1

    def remove_user(self, uid):
        return self.users.pop(uid)[1]

    def calc_avg(self):
        self.avg = sum(self.users.values()) / len(self.users.keys())

    def update(self):
        self.calc_avg()
        self.user_count = len(self.users)
        return self
