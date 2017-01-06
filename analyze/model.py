"""Time cost model based on collected data"""
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor

import parsedata


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
                # numpy array and then grab only the first 100
                tfidf_mat.getrow(title_index[str(datum[3])]).todense().A1[:200])
            labels.append(float(datum[1] - datum[0]) / 1000)
    return np.array(featureses), np.array(labels)


def _train_model(featureses, labels):
    """Train time cost model"""
    scores = []
    for _ in range(100):
        model = MLPRegressor(
            hidden_layer_sizes=10,
            solver='adam',
            max_iter=100000)
        model.fit(featureses, labels)
        scores.append(model.score(featureses, labels))
    print(scores)
    print(np.mean(scores), np.median(scores), np.std(scores))


def _run(args):
    """Build time cost model"""
    userdata = parsedata.get_data(args.userdata)
    filedict = parsedata.grab_pickle(args.filedict)
    sorted_titles = sorted([title for title in filedict])
    title_index = _build_title_index(sorted_titles)
    _, tfidf_mat = _build_tfidfer(filedict, sorted_titles)
    featureses, labels = _build_learning_data(
        userdata,
        filedict,
        title_index,
        tfidf_mat)
    print(featureses.shape)
    _train_model(featureses, labels)


if __name__ == '__main__':
    _run(_parse_args())

