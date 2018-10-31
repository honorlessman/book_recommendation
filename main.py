from knn_model import KNN
from data_parser import CSVParser
from user_knn import UKNN

# matrix one is bugged so do not switch this off
SPARSE = False

if __name__ == "__main__":
    file_arr = ["data/BX-Users.csv", "data/BX-Book-Ratings-Train.csv", "data/BX-Books.csv",
                "data/Test/Test-User_Rating300.csv"]
    data_arr = [CSVParser(f) for f in file_arr]

    # filter out usa and canada
    data_arr[0].filter('Location', ", usa|, canada")

    # merge test file with book file to equalize book count
    test_in = CSVParser(file_arr[3])
    test_in.merge(data_arr[2], "ISBN")

    # merge all training files
    merged = data_arr[1]
    merged.merge(data_arr[0], "User-ID")
    merged.merge(data_arr[2], "ISBN")

    if SPARSE:
        """ data is sparse so using dictionary will speed this up """

        # parse data to dicts
        users, u_books = merged.to_dict(['ISBN', 'User-ID', 'Book-Rating'])
        test, t_books = test_in.to_dict(['ISBN', 'User-ID', 'Book-Rating'])
        print("data parsed")

        # train model and predict
        model = UKNN()
        model.fit(3, [users, u_books], [test, t_books])
        print("Weighted score: ", model.score)

    else:
        """ data is dense so using matrix will speed this up """

        # fill a zero filled matrix with data
        test_arr = test_in.to_matrix(['ISBN', 'User-ID', 'Book-Rating'])
        data_arr = merged.to_matrix(['ISBN', 'User-ID', 'Book-Rating'])
        print("data parsed")

        # fit data to model and make a prediction
        model = KNN()
        model.fit(3, data_arr, test_arr)
        predict = model.predict
        print("Weighted score: ", model.score)

    print("end")
