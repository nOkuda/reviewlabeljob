"""Builds order.pickle for server to serve from"""

import argparse
import math
import os.path
import pickle
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from activetm.active import select
from activetm.active.selectors.utils import distance
from activetm.tech import anchor


RNG_SEED = 431
NUM_TOPICS = 80
NUM_TRAIN = 1
TRAINED = 'trained.pickle'
TOPIC_MIXTURES = 'topicmix.pickle'

CAND_SIZE = 500


def _construct_model():
    """Make model"""
    rng = random.Random(RNG_SEED)
    return anchor.RidgeAnchor(rng, NUM_TOPICS, NUM_TRAIN)


def _get_labels(dataset):
    """Get labels for documents"""
    known_labels = []
    for title in dataset.titles:
        known_labels.append(dataset.labels[title])
    return known_labels


def _get_trained_model(dataset):
    """Get trained model"""
    if os.path.exists(TRAINED):
        with open(TRAINED, 'rb') as ifh:
            return pickle.load(ifh)
    model = _construct_model()
    known_labels = _get_labels(dataset)
    model.train(dataset, list(range(len(dataset.titles))), known_labels)
    with open(TRAINED, 'wb') as ofh:
        pickle.dump(model, ofh)
    return model


def _get_doc_topic_mixes(model, dataset):
    """Get topic mixtures for each document"""
    if os.path.exists(TOPIC_MIXTURES):
        with open(TOPIC_MIXTURES, 'rb') as ifh:
            return pickle.load(ifh)
    doc_topic_mixes = []
    for num in range(len(dataset.titles)):
        # pylint:disable=protected-access
        doc_topic_mixes.append(
            model._predict_topics(
                0,
                model._convert_vocab_space(dataset.doc_tokens(num))))
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


FIRSTS = []
def _get_next_candidate(candidates, prev_num, doc_topic_mixes, target):
    """Get next candidate"""
    pairs = []
    total = 0.0
    for cand in candidates:
        cur_cand_score = distance.js_divergence(
            doc_topic_mixes[prev_num],
            doc_topic_mixes[cand])
        pairs.append((cur_cand_score, cand))
        total += cur_cand_score
    pairs.sort()
    FIRSTS.append(pairs[0][0])
    pos = _get_pos_before_target([pair[0] for pair in pairs], target)
    jsd, cand = pairs[pos]
    return cand, jsd


# LN2 is upper bound on JSD
LN2 = math.log(2)
# Lower Bound (prevents oversampling due to scarcity of small distances)
LB = 0
RANGE = LN2 - LB
def _get_order(model, dataset):
    """Get ordering documents"""
    rng = random.Random(RNG_SEED)
    not_yet_chosen = set(list(range(len(dataset.titles))))
    prev_num = rng.randrange(len(dataset.titles))
    not_yet_chosen.remove(prev_num)
    doc_topic_mixes = _get_doc_topic_mixes(model, dataset)
    order = [(dataset.titles[prev_num], float('nan'))]
    targets = [rng.random()*RANGE + LB for _ in range(len(dataset.titles)-1)]
    pos = 0
    while len(not_yet_chosen) > 0:
        candidates = select.reservoir(list(not_yet_chosen), rng, CAND_SIZE)
        best_cand, best_cand_score = _get_next_candidate(
            candidates,
            prev_num,
            doc_topic_mixes,
            targets[pos])
        order.append((dataset.titles[best_cand], best_cand_score))
        not_yet_chosen.remove(best_cand)
        prev_num = best_cand
        pos += 1
    return order


def _run(corpus):
    """Save out order.pickle"""
    with open(corpus, 'rb') as ifh:
        dataset = pickle.load(ifh)
    model = _get_trained_model(dataset)
    order = _get_order(model, dataset)
    plt.style.use('ggplot')
    plt.hist([a[1] for a in order[1:]])
    plt.savefig('hist.pdf')
    plt.clf()
    plt.hist(FIRSTS)
    plt.savefig('firsts.pdf')
    with open('order.pickle', 'wb') as ofh:
        pickle.dump(order, ofh)


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

