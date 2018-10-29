from KNN_Model import KNN
from data_parser import CSVParser
import numpy as np
from User import User
from User_KNN import UKNN

if __name__ == "__main__":
    # TODO: add a function that reads files and transforms them into numpy array or custom sparse dataframes
    model = KNN()

    file_arr = ["data/BX-Users.csv", "data/BX-Book-Ratings-Train.csv", "data/BX-Books.csv",
                "data/Test/Test-User_Rating300.csv"]
    data_arr = [CSVParser(f) for f in file_arr]

    data_arr[0].filter('Location', ", usa|, canada")

    test_in = CSVParser(file_arr[3])
    test_in.merge(data_arr[2], "ISBN")
    t_arr = test_in.DF[['ISBN', 'User-ID', 'Book-Rating']].values

    merged = data_arr[1]
    merged.merge(data_arr[0], "User-ID")
    merged.merge(data_arr[2], "ISBN")

    index1 = merged.indexify('ISBN')
    index2 = merged.indexify('User-ID')

    tindex = test_in.indexify('User-ID')

    m_arr = merged.DF[['ISBN', 'User-ID', 'Book-Rating']].values

    users = {uid: User(uid) for uid in index2.keys()}
    for user in m_arr:
        users[user[1]].append_book(user[0], user[2])

    test = {uid: User(uid) for uid in tindex.keys()}
    for user in t_arr:
        test[user[1]].append_book(user[0], user[2])

    mod = UKNN()
    mod.fit(3, users, test)

    """n_arr = np.zeros((len(index2), len(index1)))
    tt_arr = np.zeros((len(tindex), len(index1)))
    for row in t_arr:
        try:
            tt_arr[tindex[row[1]]][index1[row[0]]] = float(row[2] + 1)
        except KeyError:
            print("passing")
            continue

    for row in m_arr:
        n_arr[index2[row[1]]][index1[row[0]]] = float(row[2] + 1)

    # data_arr, test_arr = n_arr[150:], n_arr[:150]
    data_arr, test_arr = n_arr, tt_arr[:50]
    model.fit(3, data_arr, test_arr)
    predict = model.predict
    print("Weighted score: ", model.score)"""

    print("end")
