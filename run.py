"""
Run experiments

Usage:
    run.py <file_name.csv> [options]
    run.py -h | --help

Options:
  -h --help                     Show this screen.
  --frompickle                  Use pickle data if available (dump it to pickle if not)
  --model=<xgb|logreg>          Model to use. [default: xgboost]. Options are
                                    - xgb to use XGBoost
                                    - logreg to use LogisticRegression
"""

import csv
import logging
import os
import pickle
import zipfile


from docopt import docopt


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union, make_pipeline

import xgboost as xgb

from features import (
    AmountOfApps, DeviceBrand, DeviceModel, DeviceBrandFrecuency, AppLabel,
    BagOfApps, BagOfAppsVec, DeviceModelFrecuency)

from talking import main, PICKLE_TEST_NAME, PICKLE_TRAIN_NAME


logger = logging.getLogger(__name__)


def get_y_test(frompickle=True):
    """
    Load pickled data.
    """
    if frompickle:
        y = pickle.load(open(PICKLE_TEST_NAME, "rb"))
    else:
        y = main(train=False)
    return y


def get_xy_train(frompickle=True):
    """
    Load pickled data.
    """
    if frompickle:
        X = pickle.load(open(PICKLE_TRAIN_NAME, "rb"))
    else:
        X = main(train=True)
    y = [x["group"] for x in X]
    return X, y


def get_cols():
    """
    Returns cols for csv submition file.
    """
    # Kaggle requirement
    cols = ["device_id", "F23-", "F24-26", "F27-28", "F29-32", "F33-42", "F43+", "M22-",
            "M23-26", "M27-28", "M29-31", "M32-38", "M39+"]
    return cols


def generate_file(y_test, y_hat, file_name):
    """
    Generate submition csv.zip
    """
    logger.info("Creating csv file...")
    y_test_hat = zip(y_test, y_hat)
    cols = get_cols()
    with open(os.path.join("./Submitions", file_name), "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(cols)
        for test, hat in y_test_hat:
            extra = []
            extra.append(test["device_id"])
            for z in hat:
                extra.append(z)
            writer.writerow(extra)
    logger.info("Creating zip file...")
    zf = zipfile.ZipFile(os.path.join("./Submitions", file_name + ".zip"), mode='w')
    zf.write(os.path.join("./Submitions", file_name))
    zf.close()


if __name__ == '__main__':
    opts = docopt(__doc__)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info('Requesting data')
    frompickle = False
    if opts["--frompickle"]:
        frompickle = True
    X, y = get_xy_train(frompickle)

    classifier = None
    if opts["--model"] == "xgb":
        # XGB Params
        # https://www.kaggle.com/glebvasiliev/talkingdata-mobile-user-demographics/sparse-xgboost-starter-2-26857/run/331057
        # params, num_boost_round=1000, watchlist, early_stopping_rounds=25
        params = {}
        params['objective'] = "multi:softprob"
        params['eval_metric'] = 'mlogloss'
        params['eta'] = 0.005
        params['num_class'] = 12
        params['lambda'] = 3
        params['alpha'] = 2
        params["ntree"] = 330

        classifier = xgb.XGBClassifier(
            objective='multi:softprob',
            max_depth=4,
            learning_rate=0.10,
            n_estimators=600,
            reg_alpha=2,
            reg_lambda=3,
        )
    elif opts["--model"] == "logreg":
        classifier = LogisticRegression()
    else:
        exit('Undetermined model to check')

    model = make_pipeline(
        make_union(
            AmountOfApps(),
            DeviceBrand(),
            DeviceModel(),
            AppLabel(),
            DeviceBrandFrecuency(),
            BagOfAppsVec(),
            # DeviceModelFrecuency(),
        ),
        classifier
    )
    logger.info('Starting experiment with %s samples' % len(y))
    model.fit(X, y)
    y_test = get_y_test(frompickle)
    logger.info('Prediction...')
    y_hat = model.predict_proba(y_test)
    generate_file(y_test, y_hat, opts['<file_name.csv>'])
