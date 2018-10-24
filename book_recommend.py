"""
Book recommendation system using K-Nearest Neighbor

"""
import numpy as np


class BookRecommend:
    the_library = []

    def __init__(self):
        pass

    # recommend book to the given user
    def recommend(self):
        pass

    """Assign consecutive index numbers for each book so it is easier to make a data matrix"""
    def indexify(self, b_data):
        self.the_library = [(b_data[b_index]["ISBN"], b_index) for b_index in range(len(b_data))]

    """Parse data in KNN ready format"""
    def parse(self, r_data, b_data):
        self.indexify(b_data)
        parsed = np.zeros((len(r_data), len(b_data)))
        print(self.the_library)

    # Load and parse the data
    # TODO: delete or find a use for this
    def load(self, data):
        pass
