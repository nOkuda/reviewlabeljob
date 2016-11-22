"""Script to analyze use study data"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

import GPy


def _parse_args():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(description='Analyze user study data')
    parser.add_argument(
        '-o',
        default='.',
        help='directory where output is to be placed; default is current ' +\
            'working directory')
    parser.add_argument(
        'userdata',
        help='directory where data is stored; assuming the data files end ' +\
            'in ".data"')
    parser.add_argument(
        'corpus',
        help='file path to pickle containing corpus information')
    parser.add_argument(
        'topicdata',
        help='file path to pickle containing topic information')
    return parser.parse_args()


def _get_data(datadir):
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


def _get_true_labels_by_user(userdata, corpus):
    """Get true labels by user"""
    true_labels_by_user = {}
    for user in userdata:
        true_labels_by_user[user] = _get_true_labels(
            corpus,
            [str(a) for a in userdata[user][:, 3]])
    return true_labels_by_user


#pylint:disable-msg=no-member,too-many-locals
def _plot2d(filename, xdata, ydata, **kwargs):
    """Plot data as scatter with regression"""
    fig, axis = plt.subplots(1, 1)
    baseline = axis.scatter(xdata, ydata, **kwargs)
    # regression via Gaussian process
    sortedx = np.sort(xdata)
    maxdist = np.max(sortedx[1:] - sortedx[:-2])
    # TODO lengthscale needs to be tuned...not sure how to do that yet
    # lengthscale is set to 3*maxdist because points that far away along the x
    # axis seem like they should affect a given point
    kernel = GPy.kern.RBF(input_dim=1, lengthscale=3*maxdist)
    gpx = xdata.reshape((xdata.size, 1))
    gpy = ydata.reshape((ydata.size, 1))
    gpr = GPy.models.GPRegression(gpx, gpy, kernel)
    gpr.optimize()
    plotx = np.linspace(xdata.min(), xdata.max(), 200)
    plotx = plotx.reshape((plotx.size, 1))
    pred_mean, _ = gpr.predict(plotx)
    # pred_quants = gpr.predict_quantiles(plotx, quantiles=(25., 75.))
    axis.plot(plotx, pred_mean, color=baseline.get_facecolors()[0])
    fig.savefig(filename, bbox_inches='tight')


def _accuracy(guesses, true_labels):
    """Return accuracy scores"""
    return float(np.sum(guesses == true_labels)) / len(guesses)


def _mean_absolute_error(guesses, true_labels):
    """Return error scores"""
    return float(np.sum(abs(guesses - true_labels))) / len(guesses)


def _score_eval_helper(scorer):
    """Returns a function that evaluates score according to scorer"""
    def _inner(guesseses, true_labelses):
        """Return scores"""
        result = []
        for guesses, true_labels in zip(guesseses, true_labelses):
            result.append(scorer(guesses, true_labels))
        return np.array(result)
    return _inner


def _totaltime_vs_finalscore(
        userdata,
        true_labels_by_user,
        filename,
        score_eval):
    """Analyzes data to plot total time vs. final score, as per score_eval"""
    users = sorted(userdata.keys())
    # note that there are 60000 milliseconds per minute
    xdata = np.array(
        [
            float(userdata[user][-1, 1] - userdata[user][0, 0]) / 60000 \
            for user in users])
    ydata = score_eval(
        [userdata[user][:, -1] for user in users],
        [true_labels_by_user[user] for user in users])
    _plot2d(filename, xdata, ydata)


def _analyze_data(userdata, corpus, topicdata, outdir):
    """Analyze data"""
    true_labels_by_user = _get_true_labels_by_user(userdata, corpus)
    # total time spent vs. final accuracy
    _totaltime_vs_finalscore(
        userdata,
        true_labels_by_user,
        os.path.join(outdir, 'totaltime_finalaccuracy.pdf'),
        _score_eval_helper(_accuracy))
    # total time spent vs. final average absolute error
    _totaltime_vs_finalscore(
        userdata,
        true_labels_by_user,
        os.path.join(outdir, 'totaltime_finalmae.pdf'),
        _score_eval_helper(_mean_absolute_error))
    # document length vs. time spent
    # JS divergence of switch topics vs. time spent
    # box plot:  relative time spent on switch, relative time spent not on switch


def _run():
    """Run analysis"""
    args = _parse_args()
    userdata = _get_data(args.userdata)
    corpus = grab_pickle(args.corpus)
    topicdata = grab_pickle(args.topicdata)
    _analyze_data(userdata, corpus, topicdata, args.o)


if __name__ == '__main__':
    _run()
