"""Builds order.pickle for server to serve from"""

import argparse
import os.path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

from activetm.active.selectors.utils import distance
import ankura.anchor
import ankura.pipeline
import ankura.topic


RNG_SEED = 431
NUM_TOPICS = 80
NUM_TRAIN = 1
EXPGRAD_EPS = 1e-4
TRAINED = 'trained.pickle'
TOPIC_MIXTURES = 'topicmix.pickle'

CAND_SIZE = 500


def _get_trained_model(dataset):
    """Get trained model"""
    if os.path.exists(TRAINED):
        with open(TRAINED, 'rb') as ifh:
            return pickle.load(ifh)
    pdim = 1000 if dataset.vocab_size > 1000 else dataset.vocab_size
    anchors, anchor_indices = \
        ankura.anchor.gramschmidt_anchors(dataset,
                                          NUM_TOPICS,
                                          0.015*len(dataset.titles),
                                          project_dim=pdim,
                                          return_indices=True)
    topics = ankura.topic.recover_topics(dataset,
                                         anchors,
                                         EXPGRAD_EPS)
    model = {'anchors': anchors,
             'anchor_indices': anchor_indices,
             'topics': topics,
             'numtopics': NUM_TOPICS}
    with open(TRAINED, 'wb') as ofh:
        pickle.dump(model, ofh)
    return model


SAMPLES_PER_PREDICT = 5
def _predict_topics(model, docwords):
    """Predict topic assignments for docwords"""
    if len(docwords) == 0:
        return np.array([1.0/NUM_TOPICS]*NUM_TOPICS)
    result = np.zeros(NUM_TOPICS)
    for _ in range(SAMPLES_PER_PREDICT):
        counts, _ = ankura.topic.predict_topics(model['topics'],
                                                docwords)
        result += counts
    return result / (len(docwords)*SAMPLES_PER_PREDICT)


def _get_doc_topic_mixes(model, dataset):
    """Get topic mixtures for each document"""
    if os.path.exists(TOPIC_MIXTURES):
        with open(TOPIC_MIXTURES, 'rb') as ifh:
            return pickle.load(ifh)
    doc_topic_mixes = []
    for num in range(len(dataset.titles)):
        doc_topic_mixes.append(
            _predict_topics(
                model, dataset.doc_tokens(num)))
    with open(TOPIC_MIXTURES, 'wb') as ofh:
        pickle.dump(doc_topic_mixes, ofh)
    return doc_topic_mixes


def _get_pos_before_target(increasing, target):
    """Get position in increasing that is the closest to but less than target"""
    pos = 0
    while pos < len(increasing)-1:
        if target < increasing[pos]:
            break
        pos += 1
    return pos


def _build_simil_matrix(doc_topic_mixes, titles):
    """Build similarity matrix of documents"""
    simil = np.zeros((len(titles), len(titles)))
    for i in range(len(titles)):
        for j in range(i+1, len(titles)):
            simil[i, j] = distance.js_divergence(doc_topic_mixes[i],
                                                 doc_topic_mixes[j])
            simil[j, i] = simil[i, j]
    return simil


def _run(corpus):
    """Save out order.pickle"""
    with open(corpus, 'rb') as ifh:
        labeleddataset = pickle.load(ifh)
        dataset = ankura.pipeline.Dataset(labeleddataset.docwords,
                                          labeleddataset.vocab,
                                          labeleddataset.titles)
        labeleddataset = None
    model = _get_trained_model(dataset)
    doc_topic_mixes = _get_doc_topic_mixes(model, dataset)
    simil = _build_simil_matrix(doc_topic_mixes, dataset.titles)
    with open('simil.pickle', 'wb') as ofh:
        pickle.dump(simil, ofh)
    plt.hist(simil.ravel)
    plt.savefig('hist.pdf')


def _get_corpus():
    """Get corpus from argument parsing"""
    parser = argparse.ArgumentParser(description='Builder for order.pickle')
    parser.add_argument(
        'corpus',
        help='path to corpus pickle (activetm.labeled.LabeledDataset)')
    args = parser.parse_args()
    return args.corpus


if __name__ == '__main__':
    _run(_get_corpus())

