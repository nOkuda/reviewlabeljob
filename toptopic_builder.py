"""Build toptopic.pickle"""
import numpy as np
import pickle

from activetm.active.selectors.utils import distance
import simil_builder
#pylint:disable=protected-access


def _get_topic_relations(model):
    """JSD between topics"""
    relations = np.zeros((model['numtopics'], model['numtopics']))
    for i in range(model['numtopics']):
        for j in range(i+1, model['numtopics']):
            relations[i][j] = distance.js_divergence(
                model['topics'][:, i],
                model['topics'][:, j])
            relations[j][i] = relations[i][j]
    return relations


def _get_toptopic(model, dataset):
    """Organize documents by top topic"""
    toptopic = []
    for _ in range(model['numtopics']):
        toptopic.append([])
    doc_topic_mixes = simil_builder._get_doc_topic_mixes(model, dataset)
    for i in range(len(dataset.titles)):
        topic = np.argmax(doc_topic_mixes[i])
        # prevent numpy dependency in toptopic.pickle
        toptopic[topic].append(
            (float(doc_topic_mixes[i][topic]), str(dataset.titles[i])))
    for top in toptopic:
        top.sort(reverse=True)
    return toptopic


def _run(corpus):
    """Save out toptopic.pickle"""
    with open(corpus, 'rb') as ifh:
        dataset = pickle.load(ifh)
    model = simil_builder._get_trained_model(dataset)
    toptopic = _get_toptopic(model, dataset)
    with open('toptopic.pickle', 'wb') as ofh:
        pickle.dump(toptopic, ofh)
    relations = _get_topic_relations(model)
    with open('topicdistances.pickle', 'wb') as ofh:
        pickle.dump(relations, ofh)


if __name__ == '__main__':
    _run(simil_builder._get_corpus())
