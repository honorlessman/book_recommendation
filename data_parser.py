"""
Parser and toolkit for CSV data

"""

import pandas
import numpy as np
from user import User
from book import Book


class CSVParser:
    def __init__(self, fpath):
        self.DF = pandas.DataFrame()
        self.path = fpath

        if fpath != "":
            self.__read()

    def load(self, fpath):
        self.path = fpath
        self.__read()

    def __read(self):
        self.DF = pandas.read_csv(self.path, delimiter=';', header=0, encoding="ISO-8859-1", error_bad_lines=False,
                                  warn_bad_lines=False)

    def merge(self, d2, key, jtype="inner"):
        """ Merges DFs by a key """
        self.DF = self.DF.merge(d2.DF, on=[key], how=jtype)
        return self.DF

    def filter(self, key, value):
        """ Filters out a Dataframe by given key/value pairs """
        self.DF = self.DF.loc[self.DF[key].str.contains(value)]

    def indexify(self, key):
        """ Enumerates given key to a an index starting from 0 """
        arr = self.DF[key].drop_duplicates().real
        return {arr[data]: data for data in range(len(arr))}

    def distinct(self, key):
        return self.DF[key].drop_duplicates()

    def get_columns(self, column_names):
        if type(column_names) == list:
            return self.DF[column_names].values
        return self.DF[[column_names]].values

    """ Transforms dataframe into zero padded ndarray matrix """
    def to_matrix(self, columns):

        book_index = self.indexify('ISBN')
        user_index = self.indexify('User-ID')

        n_arr = np.zeros((len(user_index), len(book_index)))
        data = self.get_columns(columns)

        for row in data:
            n_arr[user_index[row[1]]][book_index[row[0]]] = float(row[2] + 1)

        return n_arr

    """ Transforms the dataframe into user and book object dictionary """
    def to_dict(self, columns):

        # get distinct keys
        book_index = self.indexify('ISBN')
        user_index = self.indexify('User-ID')

        # get data array
        m_arr = self.get_columns(columns)

        # base dicts
        users = {uid: User(uid) for uid in user_index.keys()}
        books = {isbn: Book(isbn) for isbn in book_index.keys()}

        # fill dict above with objects
        for user in m_arr:
            users[user[1]].append_book(user[0], user[2])
            users[user[1]].update()

            books[user[0]].append_user(user[1], user[2])
            books[user[0]].update()

        return users, books
