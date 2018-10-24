from KNN_Model import KNN
from data_parser import CSVParser
import numpy as np

if __name__ == "__main__":
    model = KNN()

    file_arr = ["data/BX-Users.csv", "data/BX-Book-Ratings-Train.csv", "data/BX-Books.csv"]
    data_arr = [CSVParser(f) for f in file_arr]

    data_arr[0].filter('Location', ", usa|, canada")

    merged = data_arr[1]
    merged.merge(data_arr[0], "User-ID")
    merged.merge(data_arr[2], "ISBN")

    index1 = merged.indexify('ISBN')
    index2 = merged.indexify('User-ID')

    m_arr = merged.DF[['ISBN', 'User-ID', 'Book-Rating']].values
    n_arr = np.zeros((len(index2), len(index1)))
    for row in m_arr:
        n_arr[index2[row[1]]][index1[row[0]]] = (row[2] + 1)

    data_arr, test_arr = n_arr[10:], n_arr[:10]
    test = model.fit(3, data_arr, test_arr)

    print("end")
