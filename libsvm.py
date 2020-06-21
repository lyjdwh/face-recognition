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
        param_grid = {
            "kernel": ["linear"],
            "C": [0.01, 0.1, 1, 10, 100],
            "gamma": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        }
        fivefolds = GridSearchCV(
            svm.SVC(class_weight="balanced"), param_grid, cv=5, verbose=0
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
