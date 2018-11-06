from code.knn_model import KNN
from code.data_parser import CSVParser
from code.user_knn import UKNN

# matrix one is not very accurate since it uses cosine similarity also not very optimized so do not use if you can
# switches between matrix mode and dictionary mode depending
# FALSE =   Matrix      (faster on dense data(even then immensely slow) but less accurate)
# TRUE  =   Dictionary  (faster on sparse data, more accurate)
SPARSE = True

# Test file path
TEST_FILE = "code/data/BXBookRatingsTest.csv"
# TEST_FILE = "code/data/Test/Test-User_Rating300.csv"

# Fix column names if they are not similar to data
FIX_COLUMN_NAMES = True

if __name__ == "__main__":
    file_arr = ["code/data/BX-Users.csv", "code/data/BX-Book-Ratings-Train.csv", "code/data/BX-Books.csv"]
    data_arr = [CSVParser(f) for f in file_arr]

    # filter out usa and canada
    data_arr[0].filter('Location', ", usa|, canada")

    # merge all training files
    merged = data_arr[1]
    merged.merge(data_arr[0], "User-ID")
    merged.merge(data_arr[2], "ISBN")

    # merge test file with book file to equalize book count
    test_in = CSVParser(TEST_FILE)
    if FIX_COLUMN_NAMES:
        test_in.DF.columns = ["User-ID", "ISBN", "Book-Rating"]

    test_in.filter_by_series(merged.DF['ISBN'], 'ISBN')

    if SPARSE:
        """ data is sparse so using dictionary will speed this up """

        # parse data to dicts
        users, u_books = merged.to_dict(['ISBN', 'User-ID', 'Book-Rating'])
        test, t_books = test_in.to_dict(['ISBN', 'User-ID', 'Book-Rating'])

        CSVParser.normals(users, u_books)
        CSVParser.normals(test, t_books)

        print("data parsed")

        # train model and predict
        model = UKNN()

        # Cross Validation for model, pretty slow depending on k values(up to 500seconds) so think carefully
        # cross_val = CrossValidation(1, 20, [users, u_books], UKNN())
        # cross_val.validate(10)

        model.fit([users, u_books], [test, t_books], k=2, threshold=20)
        print("Weighted test score: ", model.score)
        print("Non-Weighted test score: ", model.score_nw)

    else:
        """ data is dense so using matrix will give us a constant runtime compared to dictionary """

        # since data might have different books we eliminate that chance by using common index
        book_index = merged.indexify('ISBN')

        # fill a zero filled matrix with data
        test_arr = test_in.to_matrix(['ISBN', 'User-ID', 'Book-Rating'], book_index)
        data_arr = merged.to_matrix(['ISBN', 'User-ID', 'Book-Rating'], book_index)
        print("data parsed")

        # fit data to model and make a prediction
        # since matrix method is extremely slow test count is limited
        model = KNN()
        model.fit(3, data_arr, test_arr[:200])
        predict = model.predict
        print("Weighted score: ", model.score)

    print("end")
