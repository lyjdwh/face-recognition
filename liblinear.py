import os
import time

import joblib
from data import Data
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def compute_accuracy(test_x, test_y, model):
    predict_y = model.predict(test_x)
    accuracy = sum(predict_y == test_y) / len(test_y)
    return accuracy


if __name__ == "__main__":

    ps = [7, 13, 20]
    for p in ps:
        # load data
        data = Data(p=p)
        data.preprocess()
        train_x, train_y = data.get_train_data()

        # cross validation , para choosing
        param_grid = {"C": [1e-2, 1e-1, 1, 10, 100, 1000, 10000]}
        fivefolds = GridSearchCV(
            svm.LinearSVC(
                class_weight="balanced", penalty="l2", loss="hinge", max_iter=10000
            ),
            param_grid,
            cv=5,
            verbose=0,
        )

        training = fivefolds.fit(train_x, train_y.ravel())
        print("The best estimator found by GridSearch:")
        print(fivefolds.best_estimator_)

        # test
        test_x, test_y = data.get_test_data()

        starttime = time.time()
        predict_y = fivefolds.predict(test_x)
        endtime = time.time()
        print("test speed %f s/image : ", (endtime - starttime) / test_y.shape[0])
        accuracy = sum(predict_y == test_y) / len(test_y)
        print("accuracy: ", accuracy)

        # save models
        if not os.path.exists("./models"):
            os.makedirs("./models")
        joblib.dump(fivefolds.best_estimator_, "./models/libsvm-%d" % p)
