"""Time cost model based on collected data"""
import argparse
from itertools import cycle
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

import parsedata


TOP_TFIDF = 41562
SHUFFLE_SPLIT = ShuffleSplit(n_splits=100, test_size=0.25)
LINECYCLER = cycle(['-', '--', ':', '-.'])


def _parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Build time cost model')
    parser.add_argument(
        'userdata',
        help='directory where data is stored; assuming the data files end ' + \
            'in ".data"')
    parser.add_argument(
        'filedict',
        help='file path to pickle dictionary of corpus information')
    return parser.parse_args()


def _build_tfidfer(filedict, sorted_titles):
    """Build TfidfVectorizer"""
    result = TfidfVectorizer()
    mat = result.fit_transform(
        [filedict[title]['text'] for title in sorted_titles])
    return result, mat


def _build_title_index(sorted_titles):
    """Build title index"""
    result = {}
    for i, title in enumerate(sorted_titles):
        result[title] = i
    return result


def _build_learning_data(userdata, filedict, title_index, tfidf_mat):
    """Build features list with corresponding label list"""
    featureses = []
    labels = []
    for _, data in userdata.items():
        last_topic = -1
        count = 0
        for datum in data:
            featureses.append([])
            if last_topic != datum[2]:
                last_topic = datum[2]
                count = 0
            else:
                count += 1
            featureses[-1].append(count)
            featureses[-1].append(
                # get document length
                len(filedict[str(datum[3])]['text'].split()))
            featureses[-1].extend(
                # magical incantation to make returned sparse row matrix into 1D
                # numpy array and then grab only the first TOP_TFIDF
                tfidf_mat.getrow(title_index[str(datum[3])]).todense().A1[:TOP_TFIDF])
            labels.append(float(datum[1] - datum[0]) / 1000)
    return np.array(featureses), np.array(labels)


FEATURES_SIZES = [0, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]


def _crossval_model(model, model_id, featureses, labels):
    """Cross validate time cost model"""
    print(model_id)
    for tfidf_count in FEATURES_SIZES:
        limit = tfidf_count + 2
        start_time = time.time()
        model = Ridge()
        scores = cross_val_score(
            model,
            featureses[:, :limit],
            labels,
            cv=SHUFFLE_SPLIT)
        compute_time = time.time() - start_time
        print(
            limit,
            np.mean(scores),
            np.median(scores),
            np.std(scores),
            compute_time)


def _check_ridge_model(featureses, labels):
    """Plot ridge regression predictions"""
    for tfidf_count in FEATURES_SIZES:
        test_points = []
        for i in range(16):
            tmp = [i, 100]
            tmptmp = [0] * tfidf_count
            if tmptmp:
                tmp.extend(tmptmp)
            test_points.append(tmp)
        test_points = np.array(test_points)
        limit = tfidf_count + 2
        model = Ridge()
        model.fit(featureses[:, :limit], labels)
        predictions = model.predict(test_points)
        plt.plot(
            predictions,
            label=str(tfidf_count),
            linestyle=next(LINECYCLER),
            linewidth=3)
        # plt.text(test_points[-1, 0], predictions[-1], str(tfidf_count))
    plt.legend()
    plt.xlabel('Document order')
    plt.ylabel('Time (seconds)')
    plt.savefig('ridge_predictions.pdf')


def _run(args):
    """Build time cost model"""
    userdata = parsedata.get_data(args.userdata)
    filedict = parsedata.grab_pickle(args.filedict)
    sorted_titles = sorted([title for title in filedict])
    title_index = _build_title_index(sorted_titles)
    _, tfidf_mat = _build_tfidfer(filedict, sorted_titles)
    print(tfidf_mat.shape)
    featureses, labels = _build_learning_data(
        userdata,
        filedict,
        title_index,
        tfidf_mat)
    print(featureses.shape)
    """
    _crossval_model(
        MLPRegressor(
            hidden_layer_sizes=10,
            solver='adam',
            max_iter=100000),
        '# Neural Network',
        featureses,
        labels)
    _crossval_model(
        Ridge(),
        '# Ridge Regression',
        featureses,
        labels)
    """
    _check_ridge_model(featureses, labels)


if __name__ == '__main__':
    _run(_parse_args())

