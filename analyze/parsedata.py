"""Parse data"""
import os
import pickle

import numpy as np


def grab_pickle(filename):
    """Load pickle"""
    with open(filename, 'rb') as ifh:
        return pickle.load(ifh)


def _get_true_labels(corpus, titles):
    """Get true labels of documents"""
    result = []
    for title in titles:
        result.append(corpus[title]['label'])
    return np.array(result)


def get_true_labels_by_user(userdata, corpus):
    """Get true labels by user"""
    true_labels_by_user = {}
    for user in userdata:
        true_labels_by_user[user] = _get_true_labels(
            corpus,
            [str(a) for a in userdata[user][:, 3]])
    return true_labels_by_user


def get_data(datadir):
    """Parse data files

    For each data file, there is a matrix.
        * the 0th column is the start time (in milliseconds)
        * the 1st column is the end time (in milliseconds)
        * the 2nd column is the topic number
        * the 3rd column is the document id
        * the 4th column is the user's label
    All values in the matrix are integers.
    """
    data = {}
    for filename in os.listdir(datadir):
        filename_name, filename_ext = os.path.splitext(filename)
        if filename_ext == '.data':
            parsed_data = []
            with open(os.path.join(datadir, filename)) as ifh:
                for line in ifh:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parsed_data.append([int(a) for a in line.split()])
            data[filename_name] = np.array(parsed_data)
    return data

