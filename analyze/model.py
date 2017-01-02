"""Time cost model based on collected data"""
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

import analyze
import parsedata


def _parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Build time cost model')
    parser.add_argument(
        'userdata',
        help='directory where data is stored; assuming the data files end ' + \
            'in ".data"')
    parser.add_argument(
        'corpus',
        help='file path to pickle containing corpus information')
    parser.add_argument(
        'titles',
        help='file path to pickle containing ordered titles')
    return parser.parse_args()


def _build_tfidfer(corpus, sorted_titles):
    """Build TfidfVectorizer"""
    result = TfidfVectorizer()
    mat = result.fit_transform(
        [corpus[title]['text'] for title in sorted_titles])
    return result, mat


def _build_title_index(sorted_titles):
    """Build title index"""
    result = {}
    for i, title in enumerate(sorted_titles):
        result[title] = i
    return result


def _build_learning_data(userdata, title_index, tfidf_mat):
    """Build features list with corresponding label list"""
    featureses = []
    labels = []
    for user, data in userdata.items():
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
            featureses[-1].extend(
                # magical incantation to make returned sparse row matrix into 1D
                # numpy array and then grab only the first 100
                tfidf_mat.getrow(title_index[str(datum[3])]).todense().A1[:100])
            labels.append(float(datum[1] - datum[0]) / 1000)
    return np.array(featureses), np.array(labels)


def _train_model(featureses, labels):
    """Train time cost model"""
    model = MLPRegressor(
        hidden_layer_sizes=(int(len(featureses[0] * 2 / 3))),
        solver='adam',
        max_iter=5000)
    scores = cross_val_score(model, featureses, labels, cv=5)
    print(scores)
    print(scores.mean(), scores.std())


def _run(userdata, corpus):
    """Build time cost model"""
    sorted_titles = sorted([title for title in corpus])
    title_index = _build_title_index(sorted_titles)
    tfidfer, tfidf_mat = _build_tfidfer(corpus, sorted_titles)
    featureses, labels = _build_learning_data(userdata, title_index, tfidf_mat)
    print(featureses.shape)
    _train_model(featureses, labels)


if __name__ == '__main__':
    args = _parse_args()
    _run(
        parsedata.get_data(args.userdata),
        parsedata.grab_pickle(args.corpus))

