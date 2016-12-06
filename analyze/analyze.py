"""Script to analyze user study data"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
from sklearn.linear_model import LinearRegression

import GPy


#pylint:disable-msg=too-few-public-methods
class DivergenceChecker():
    """Class for looking up document divergences"""

    def __init__(self, divergence, titles):
        """

            * divergence :: 2D numpy matrix of floats
                DxD, where D is number of documents; should line up in order as
                given by titles; each cell contains of JS divergence of topic
                mixtures of the documents compared
            * titles :: 1D numpy matrix of strings
                There should be D entries
        """
        self.divergence = divergence
        self.indexfinder = {}
        for i, title in enumerate(titles):
            self.indexfinder[title] = i

    def find_div(self, title1, title2):
        """Get divergence score between title1 and title2"""
        return self.divergence[
            self.indexfinder[title1],
            self.indexfinder[title2]]


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
        'divergence',
        help='file path to pickle containing divergence matrix of documents')
    parser.add_argument(
        'titles',
        help='file path to pickle containing ordered titles')
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
    """Plot data as scatter with regression

    Opacity settings are dealt with in this function, so don't include alpha in
    kwargs.
    """
    fig, axis = plt.subplots(1, 1)
    axis.scatter(xdata, ydata, alpha=0.5, **kwargs)
    # regression via Gaussian process
    sortedx = np.sort(xdata)
    meandist = np.mean(sortedx[1:] - sortedx[:-1])
    # TODO lengthscale needs to be tuned...not sure how to do that yet
    # lengthscale is set to 3*maxdist because points that far away along the x
    # axis seem like they should affect a given point
    kernel = GPy.kern.RBF(input_dim=1, lengthscale=meandist)
    gpx = xdata.reshape((xdata.size, 1))
    gpy = ydata.reshape((ydata.size, 1))
    gpr = GPy.models.GPRegression(gpx, gpy, kernel)
    gpr.optimize()
    plotx = np.linspace(xdata.min(), xdata.max(), 200)
    plotx = plotx.reshape((plotx.size, 1))
    pred_mean, _ = gpr.predict(plotx)
    pred_line = axis.plot(
        plotx,
        pred_mean,
        alpha=0.8,
        linewidth=3)[0]
    pred_quants = gpr.predict_quantiles(plotx, quantiles=(25., 75.))
    axis.plot(
        plotx,
        pred_quants[0],
        color=pred_line.get_color(),
        alpha=0.8,
        linestyle='dashed',
        linewidth=1.5)
    axis.plot(
        plotx,
        pred_quants[1],
        color=pred_line.get_color(),
        alpha=0.8,
        linestyle='dashed',
        linewidth=1.5)
    fig.savefig(filename, bbox_inches='tight')


def _accuracy(guesses, true_labels):
    """Return accuracy scores"""
    return float(np.sum(guesses == true_labels)) / len(guesses)


def _mean_absolute_error(guesses, true_labels):
    """Return error scores"""
    return float(np.sum(abs(guesses - true_labels))) / len(guesses)


def _score_eval_helper(scorer):
    """Return a function that evaluates score according to scorer"""
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
    """Analyze data and plot total time vs. final score, as per score_eval"""
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


def _doclength_vs_time(userdata, corpus, filename):
    """Analyze data and plot document length vs. time spent labeling document"""
    xdata = []
    ydata = []
    for user in userdata:
        docids = userdata[user][:, 3]
        doclengths = [len(corpus[str(a)]['text'].split()) for a in docids]
        xdata.extend(doclengths)
        # 1000 milliseconds per second
        times = [
            float(a) / 1000 \
            for a in (userdata[user][:, 1] - userdata[user][:, 0])]
        ydata.extend(times)
    _plot2d(filename, np.array(xdata), np.array(ydata))


def _extract_time(_, usermatrix):
    """Extract time information for all labels except first"""
    # there are 1000 milliseconds per second
    return (usermatrix[1:, 1] - usermatrix[1:, 0]) / 1000


def _extract_runningaccdiff(true_labels_by_user):
    """Return a function that extracts running accuracy data"""
    def _inner(user, usermatrix):
        """Extract running accuracy data"""
        truelabels = true_labels_by_user[user]
        accuracies = truelabels == usermatrix[:, 4]
        runningacc = np.array([
            float(np.sum(accuracies[:a])) / float(a + 1) \
            for a in range(len(accuracies))])
        return runningacc[1:] - runningacc[:-1]
    return _inner


def _extract_runningtotalaccdiff(true_labels_by_user):
    """Return a function that extracts running total accuracy data"""
    def _inner(user, usermatrix):
        """Extract running total accuracy data"""
        truelabels = true_labels_by_user[user]
        accuracies = truelabels == usermatrix[:, 4]
        runningtotalacc = np.array([
            float(np.sum(accuracies[:a])) / float(len(accuracies)) \
            for a in range(len(accuracies))])
        return runningtotalacc[1:] - runningtotalacc[:-1]
    return _inner


def _extract_runningmeanerrdiff(true_labels_by_user):
    """Return a function that extracts running mean error data"""
    def _inner(user, usermatrix):
        """Extract running mean error data"""
        truelabels = true_labels_by_user[user]
        errors = np.abs(truelabels - usermatrix[:, 4])
        runningmeanerr = np.array([
            float(np.sum(errors[:a])) / float(a + 1) \
            for a in range(len(errors))])
        return runningmeanerr[1:] - runningmeanerr[:-1]
    return _inner


def _extract_runningtotalerrdiff(true_labels_by_user):
    """Return a function that extracts running total error data"""
    def _inner(user, usermatrix):
        """Extract running total error data"""
        truelabels = true_labels_by_user[user]
        errors = np.abs(truelabels - usermatrix[:, 4])
        result = [errors[0]]
        for error in errors[1:]:
            result.append(result[-1] + error)
        runningtotalerr = np.array(result)
        return runningtotalerr[1:] - runningtotalerr[:-1]
    return _inner


def _other_eval_helper(extractor):
    """Return a function that extracts data"""
    def _inner(user, usermatrix):
        """Return data"""
        return extractor(user, usermatrix)
    return _inner


def _docdiv_vs_other(userdata, s_checker, filename, other_eval):
    """Analyze data and plot document divergence vs. time spent labeling"""
    xdata = []
    ydata = []
    for user in userdata:
        docids = userdata[user][:, 3]
        divs = [
            s_checker.find_div(str(a), str(b)) \
            for a, b in zip(docids[:-1], docids[1:])]
        xdata.extend(divs)
        others = other_eval(user, userdata[user])
        ydata.extend(others)
    _plot2d(filename, np.array(xdata), np.array(ydata))


def _get_relative_times(userdata):
    """Calculate relative times by user

    Relative time here means the proportion to the max time spent labeling for a
    given user.
    """
    result = {}
    for user in userdata:
        times = userdata[user][:, 1] - userdata[user][:, 0]
        maxtime = float(np.max(times))
        portions = []
        for time in times:
            portions.append(float(time) / maxtime)
        result[user] = np.array(portions)
    return result


def _make_boxplot(data, labels, filename):
    """Plot boxplots"""
    fig, axis = plt.subplots(1, 1)
    axis.boxplot(data, labels=labels)
    fig.savefig(filename, bbox_inches='tight')


def _switch_vs_not(userdata, relative_times_by_user, filename):
    """Analyze data and make box plots of relative times"""
    switch = []
    notswitch = []
    for user in userdata:
        reltimes = relative_times_by_user[user]
        not_switch_indices = userdata[user][:-1, 2] == userdata[user][1:, 2]
        # the first item is considered a switch
        not_switch_indices = np.insert(not_switch_indices, 0, False)
        notswitch.append(reltimes[not_switch_indices])
        switch.append(reltimes[np.logical_not(not_switch_indices)])
    _make_boxplot([switch, notswitch], ['switch', 'not switch'], filename)


def _numlabeled_vs_reltime(userdata, relative_times_by_user, filename):
    """Analyze data and plot number of documents labeled vs. relative time"""
    bins = []
    for user in userdata:
        bins.append(relative_times_by_user[user])
    # every row is now the ith relative time for each user
    bins = np.array(bins).T

    user = next(iter(userdata.keys()))
    not_switch_indices = \
        userdata[user][:-1, 2] == \
        userdata[user][1:, 2]
    # the first item is considered a switch
    not_switch_indices = np.insert(not_switch_indices, 0, False)
    switch_indices = np.nonzero(np.logical_not(not_switch_indices))[0]
    major_locator = FixedLocator(switch_indices+1)
    major_formatter = FixedFormatter([str(i+1) for i in switch_indices])

    fig, axis = plt.subplots(1, 1)
    axis.boxplot([b for b in bins])
    axis.xaxis.set_major_locator(major_locator)
    axis.xaxis.set_major_formatter(major_formatter)
    fig.savefig(filename, bbox_inches='tight')


def _multiregression_helper(
        userdata,
        s_checker,
        corpus,
        relative_times_by_user):
    """Get data necessary for multiple regression cases"""
    doclengths = []
    reltimes = []
    final_doclengths = []
    final_reltimes = []
    doc_divs = []
    for user, data in userdata.items():
        docids = data[:, 3]
        curdoclengths = [len(corpus[str(a)]['text'].split()) for a in docids]
        doclengths.extend(curdoclengths)
        final_doclengths.extend(curdoclengths[1:])
        reltimes.extend(relative_times_by_user[user])
        final_reltimes.extend(relative_times_by_user[user][1:])
        divs = [
            s_checker.find_div(str(a), str(b)) \
            for a, b in zip(docids[:-1], docids[1:])]
        doc_divs.extend(divs)
    doclengths = np.array(doclengths).reshape((len(doclengths), 1))
    reltimes = np.array(reltimes)
    final_doclengths = np.array(final_doclengths).reshape((
        len(final_doclengths), 1))
    final_reltimes = np.array(final_reltimes)
    doc_divs = np.array(doc_divs)
    return doclengths, reltimes, final_doclengths, final_reltimes, doc_divs


#pylint:disable-msg=invalid-name
def _docdiv_vs_doclength_reltime_residuals(
        userdata,
        s_checker,
        corpus,
        relative_times_by_user,
        filename):
    """Analyze data and plot document length residuals and document divergence
    vs. relative time

    Note that the first document's length is not considered because we don't
    know what its JS divergences is comparative to nothing"""
    doclengths, reltimes, final_doclengths, final_reltimes, doc_divs = \
        _multiregression_helper(
            userdata,
            s_checker,
            corpus,
            relative_times_by_user)

    doclength_vs_reltime = LinearRegression()
    doclength_vs_reltime.fit(doclengths, reltimes)
    predictions = doclength_vs_reltime.predict(final_doclengths)
    residuals = np.abs(final_reltimes - predictions)

    _plot2d(filename, doc_divs, residuals)
    fig, axis = plt.subplots(1, 1)
    axis.scatter(doclengths, reltimes)
    axis.plot(doclengths, doclength_vs_reltime.predict(doclengths))
    fig.savefig('regression.pdf', bbox_inches='tight')


def _doclength_docdiv_vs_reltime(
        userdata,
        s_checker,
        corpus,
        relative_times_by_user,
        filename):
    """Analyze data and plot document length residuals and document divergence
    vs. relative time

    Note that the first document's length is not considered because we don't
    know what its JS divergences is comparative to nothing"""
    _, _, final_doclengths, final_reltimes, doc_divs = \
        _multiregression_helper(
            userdata,
            s_checker,
            corpus,
            relative_times_by_user)

    regr = LinearRegression()
    regr.fit(
        np.array(
            [[a, b] for a, b in zip(final_doclengths.ravel(), doc_divs)]),
        final_reltimes)
    xcoords = np.linspace(0, np.max(final_doclengths), 100)
    ycoords = np.linspace(0, np.max(doc_divs), 100)
    xgrid, ygrid = np.meshgrid(xcoords, ycoords)
    predictions = []
    for ycoord in ycoords:
        predictions.append(
            regr.predict(
                np.array(
                    [
                        xcoords,
                        np.array([ycoord] * len(ycoords))]).T))
    fig, axis = plt.subplots(1, 1)
    plot = axis.pcolormesh(
        xgrid,
        ygrid,
        np.array(predictions),
        cmap=plt.cm.Reds,
        vmin=0.0,
        vmax=1.0)
    plt.colorbar(plot)
    fig.savefig(filename, bbox_inches='tight')


def _firsts_vs_lasts_reltimes(userdata, relative_times_by_user, filename):
    """Analyze data and make plots of relative times"""
    firsts = []
    lasts = []
    for user, data in userdata.items():
        reltimes = relative_times_by_user[user]
        switches = data[:-1, 2] != data[1:, 2]
        switches = np.insert(switches, 0, True)
        switch_indices = np.nonzero(switches)[0]
        for i in range(3):
            firsts_indices = switch_indices+i
            firsts.extend(reltimes[firsts_indices])
            lasts_indices = switch_indices-(i+1)
            lasts.extend(reltimes[lasts_indices])
    _make_boxplot([firsts, lasts], ['firsts', 'lasts'], filename)


def _analyze_data(userdata, corpus, divergence, titles, outdir):
    """Analyze data"""
    true_labels_by_user = _get_true_labels_by_user(userdata, corpus)
    s_checker = DivergenceChecker(divergence, titles)
    relative_times_by_user = _get_relative_times(userdata)
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
    _doclength_vs_time(
        userdata,
        corpus,
        os.path.join(outdir, 'doclength_time.pdf'))
    # JS divergence of switch topics vs. time spent
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_time.pdf'),
        _other_eval_helper(_extract_time))
    # JS divergence of switch topics vs. running accuracy
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningaccdiff.pdf'),
        _other_eval_helper(_extract_runningaccdiff(true_labels_by_user)))
    # JS divergence of switch topics vs. running total accuracy
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningtotalaccdiff.pdf'),
        _other_eval_helper(_extract_runningtotalaccdiff(true_labels_by_user)))
    # JS divergence of switch topics vs. running mean error
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningmeanerrdiff.pdf'),
        _other_eval_helper(_extract_runningmeanerrdiff(true_labels_by_user)))
    # JS divergence of switch topics vs. running total error
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningtotalerrdiff.pdf'),
        _other_eval_helper(_extract_runningtotalerrdiff(true_labels_by_user)))
    # box plot:  relative time spent on switch, relative time spent not on switch
    _switch_vs_not(
        userdata,
        relative_times_by_user,
        os.path.join(outdir, 'switch_relativetimes.pdf'))
    # number of docs labeled vs. relative time spent
    _numlabeled_vs_reltime(
        userdata,
        relative_times_by_user,
        os.path.join(outdir, 'numlabeled_relativetimes.pdf'))
    # residual on length of docs and JS divergences of doc vs. relative time
    # spent
    _docdiv_vs_doclength_reltime_residuals(
        userdata,
        s_checker,
        corpus,
        relative_times_by_user,
        os.path.join(outdir, 'docdiv_doclengthreltimeresiduals.pdf'))
    # heatmap of doc length and divergence vs relative time
    _doclength_docdiv_vs_reltime(
        userdata,
        s_checker,
        corpus,
        relative_times_by_user,
        os.path.join(outdir, 'doclengthdiv_reltime.pdf'))
    # box plot:  relative time spent after switch, relative time spent before
    # switch
    _firsts_vs_lasts_reltimes(
        userdata,
        relative_times_by_user,
        os.path.join(outdir, 'firsts_lasts_relativetimes.pdf'))

def _run():
    """Run analysis"""
    args = _parse_args()
    userdata = _get_data(args.userdata)
    corpus = grab_pickle(args.corpus)
    divergence = np.load(args.divergence)
    titles = grab_pickle(args.titles)
    _analyze_data(userdata, corpus, divergence, titles, args.o)


if __name__ == '__main__':
    _run()
