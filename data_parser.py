"""
Parser and toolkit for CSV data

"""

import pandas


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

    def merge(self, d2: pandas.DataFrame(), key, jtype="inner"):
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
