"""Script to analyze user study data"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
from scipy.stats import kstest, ks_2samp, gamma, mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import GPy
import parsedata


# pylint:disable-msg=too-few-public-methods
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
        help='directory where output is to be placed; default is current ' +
             'working directory')
    parser.add_argument(
        'userdata',
        help='directory where data is stored; assuming the data files end ' +
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


# pylint:disable-msg=no-member
def _plot2d_linear(xdata, ydata, _, axis, line_color):
    """Fits a linear regression to the data and plots regression line"""
    linreg = LinearRegression()
    regx = xdata.reshape((len(xdata), 1))
    regy = ydata.reshape((len(ydata), 1))
    linreg.fit(regx, regy)
    corr = linreg.score(regx, regy)
    plotx = np.linspace(xdata.min(), xdata.max())
    axis.plot(
        plotx,
        linreg.predict(plotx.reshape(len(plotx), 1)),
        color=line_color,
        solid_capstyle='round',
        linewidth=3)
    axis.set_title('$R^2$ = ' + str(corr))


# pylint:disable-msg=no-member
def _plot2d_gaussian(xdata, ydata, _, axis, line_color):
    """Fit a Gaussian process to the data and plots the regression"""
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
    axis.plot(
        plotx,
        pred_mean,
        color=line_color,
        alpha=0.8,
        linewidth=3)
    pred_quants = gpr.predict_quantiles(plotx, quantiles=(25., 75.))
    axis.plot(
        plotx,
        pred_quants[0],
        color=line_color,
        alpha=0.8,
        linestyle='dashed',
        linewidth=1.5)
    axis.plot(
        plotx,
        pred_quants[1],
        color=line_color,
        alpha=0.8,
        linestyle='dashed',
        linewidth=1.5)
    axis.set_title('Gaussian Process')


def _plot2d(filename, xdata, ydata, plot_helper, xlabel, ylabel):
    """Plot data as scatter with regression

    Opacity settings are dealt with in this function, so don't include alpha in
    kwargs.
    """
    fig, axis = plt.subplots(1, 1)
    color_list = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    axis.plot(
        xdata,
        ydata,
        alpha=0.5,
        linestyle='None',
        marker='o',
        markersize=3,
        markeredgewidth=0,
        markerfacecolor=color_list[0])
    plot_helper(xdata, ydata, fig, axis, color_list[1])
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


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
        score_eval,
        ylabel):
    """Analyze data and plot total time vs. final score, as per score_eval"""
    users = sorted(userdata.keys())
    # note that there are 60000 milliseconds per minute
    xdata = np.array(
        [
            float(userdata[user][-1, 1] - userdata[user][0, 0]) / 60000
            for user in users])
    ydata = score_eval(
        [userdata[user][:, -1] for user in users],
        [true_labels_by_user[user] for user in users])
    _plot2d(
        filename,
        xdata,
        ydata,
        _plot2d_linear,
        'Time (minutes)',
        ylabel)


def _doclength_vs_time(userdata, corpus, filename):
    """Analyze data and plot document length vs. time spent labeling document
    """
    xdata = []
    ydata = []
    for user in userdata:
        docids = userdata[user][:, 3]
        doclengths = [len(corpus[str(a)]['text'].split()) for a in docids]
        xdata.extend(doclengths)
        # 1000 milliseconds per second
        times = [
            float(a) / 1000
            for a in (userdata[user][:, 1] - userdata[user][:, 0])]
        ydata.extend(times)
    _plot2d(
        filename,
        np.array(xdata),
        np.array(ydata),
        _plot2d_linear,
        'Document length (number of tokens)',
        'Time (seconds)')


def _extract_time(_, usermatrix):
    """Extract time information for all labels except first"""
    # there are 1000 milliseconds per second
    return (usermatrix[1:, 1] - usermatrix[1:, 0]) / 1000


def _extract_reltime(_, usermatrix):
    """Extract relative time information for all labels except first"""
    times = usermatrix[1:, 1] - usermatrix[1:, 0]
    return times / times.max()


def _extract_runningaccdiff(true_labels_by_user):
    """Return a function that extracts running accuracy data"""
    def _inner(user, usermatrix):
        """Extract running accuracy data"""
        truelabels = true_labels_by_user[user]
        accuracies = truelabels == usermatrix[:, 4]
        runningacc = np.array([
            float(np.sum(accuracies[:a])) / float(a + 1)
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
            float(np.sum(accuracies[:a])) / float(len(accuracies))
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
            float(np.sum(errors[:a])) / float(a + 1)
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


def _docdiv_vs_other(userdata, s_checker, filename, other_eval, ylabel):
    """Analyze data and plot document divergence vs. time spent labeling"""
    xdata = []
    ydata = []
    for user in userdata:
        docids = userdata[user][:, 3]
        divs = [
            s_checker.find_div(str(a), str(b))
            for a, b in zip(docids[:-1], docids[1:])]
        xdata.extend(divs)
        others = other_eval(user, userdata[user])
        ydata.extend(others)
    _plot2d(
        filename,
        np.array(xdata),
        np.array(ydata),
        _plot2d_linear,
        'Document divergence',
        ylabel)


def _get_relative_times(userdata):
    """Calculate relative times by user

    Relative time here means the proportion to the max time spent labeling for
    a given user.
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


def _make_boxplot(data, labels, filename, xlabel, ylabel):
    """Plot boxplots"""
    fig, axis = plt.subplots(1, 1)
    axis.boxplot(data, labels=labels)
    if len(data) == 2:
        test_stat, pval = ks_2samp(data[0], data[1])
        axis.set_title(
            'KS: test_stat=' + str(test_stat) + '; pval=' + str(pval))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _switch_vs_not(userdata, filename):
    """Analyze data and make box plots of times"""
    switch = []
    notswitch = []
    for user in userdata:
        times = (userdata[user][:, 1] - userdata[user][:, 0]) / 1000
        not_switch_indices = userdata[user][:-1, 2] == userdata[user][1:, 2]
        # the first item is considered a switch
        not_switch_indices = np.insert(not_switch_indices, 0, False)
        notswitch.extend(times[not_switch_indices])
        switch.extend(times[np.logical_not(not_switch_indices)])
    _make_boxplot(
        [switch, notswitch],
        ['switch', 'not switch'],
        filename,
        '',
        'Time (seconds)')


def _switch_vs_not_relative(userdata, relative_times_by_user, filename):
    """Analyze data and make box plots of relative times"""
    switch = []
    notswitch = []
    for user in userdata:
        reltimes = relative_times_by_user[user]
        not_switch_indices = userdata[user][:-1, 2] == userdata[user][1:, 2]
        # the first item is considered a switch
        not_switch_indices = np.insert(not_switch_indices, 0, False)
        notswitch.extend(reltimes[not_switch_indices])
        switch.extend(reltimes[np.logical_not(not_switch_indices)])
    _make_boxplot(
        [switch, notswitch],
        ['switch', 'not switch'],
        filename,
        '',
        'Relative time (% of participant\'s longest labeling time)')


def _get_not_switch_indices(userdata):
    """Get array of booleans indicating whether there was a topic switch"""
    not_switch_indices = np.ones(1)
    for _, data in userdata.items():
        cur_not_switch_indices = \
            data[:-1, 2] == \
            data[1:, 2]
        if len(not_switch_indices) < len(cur_not_switch_indices):
            not_switch_indices = cur_not_switch_indices
    # the first item is considered a switch
    not_switch_indices = np.insert(not_switch_indices, 0, False)
    return not_switch_indices


def _numlabeled_vs_time(userdata, filename):
    """Analyze data and plot number of documents labeled vs. time"""
    bins = []
    for _, data in userdata.items():
        bins.append((data[:, 1] - data[:, 0]) / 1000)
    # every row is now the ith time for each user
    bins = np.array(bins).T

    not_switch_indices = _get_not_switch_indices(userdata)
    switch_indices = np.nonzero(np.logical_not(not_switch_indices))[0]
    major_locator = FixedLocator(switch_indices+1)
    major_formatter = FixedFormatter([str(i+1) for i in switch_indices])

    fig, axis = plt.subplots(1, 1)
    axis.boxplot([b for b in bins])
    axis.xaxis.set_major_locator(major_locator)
    axis.xaxis.set_major_formatter(major_formatter)
    axis.set_xlabel('Number of documents labeled')
    axis.set_ylabel('Time (seconds)')
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _numlabeled_vs_reltime(userdata, relative_times_by_user, filename):
    """Analyze data and plot number of documents labeled vs. relative time"""
    bins = []
    for user in userdata:
        bins.append(relative_times_by_user[user])
    # every row is now the ith relative time for each user
    bins = np.array(bins).T

    not_switch_indices = _get_not_switch_indices(userdata)
    switch_indices = np.nonzero(np.logical_not(not_switch_indices))[0]
    major_locator = FixedLocator(switch_indices+1)
    major_formatter = FixedFormatter([str(i+1) for i in switch_indices])

    fig, axis = plt.subplots(1, 1)
    axis.boxplot([b for b in bins])
    axis.xaxis.set_major_locator(major_locator)
    axis.xaxis.set_major_formatter(major_formatter)
    axis.set_xlabel('Number of documents labeled')
    axis.set_ylabel(
        'Relative time (% of participant\'s longest labeling time)')
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _get_doclengths_for_user(userdata, user, corpus):
    """Get doclengths for documents labeled by user"""
    docids = userdata[user][:, 3]
    return np.array([len(corpus[str(a)]['text'].split()) for a in docids])


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
        curdoclengths = _get_doclengths_for_user(userdata, user, corpus)
        doclengths.extend(curdoclengths)
        final_doclengths.extend(curdoclengths[1:])
        reltimes.extend(relative_times_by_user[user])
        final_reltimes.extend(relative_times_by_user[user][1:])
        docids = data[:, 3]
        divs = [
            s_checker.find_div(str(a), str(b))
            for a, b in zip(docids[:-1], docids[1:])]
        doc_divs.extend(divs)
    doclengths = np.array(doclengths).reshape((len(doclengths), 1))
    reltimes = np.array(reltimes)
    final_doclengths = np.array(final_doclengths).reshape((
        len(final_doclengths), 1))
    final_reltimes = np.array(final_reltimes)
    doc_divs = np.array(doc_divs)
    return doclengths, reltimes, final_doclengths, final_reltimes, doc_divs


# pylint:disable-msg=invalid-name
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

    _plot2d(
        filename,
        doc_divs,
        residuals,
        _plot2d_linear,
        'Document length vs. document divergence residual',
        'Relative time (% of participant\'s longest labeling time)')


# pylint:disable-msg=too-many-locals
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
    inputs = np.array(
        [[a, b] for a, b in zip(final_doclengths.ravel(), doc_divs)])
    regr.fit(
        inputs,
        final_reltimes)
    _plot_heatmap(
        'doclengths',
        0,
        np.max(final_doclengths),
        'docdivs',
        0,
        np.max(doc_divs),
        regr,
        regr.score(inputs, final_reltimes),
        filename)


def make_format_function(lmin, lmax, intervals):
    """Return function that scales input tick value to correct label"""
    lrange = lmax - lmin

    def _inner(xval, _):
        """Return correct label"""
        return (lrange * xval / intervals) + lmin
    return _inner


def _plot_heatmap(
        xlabel,
        xmin,
        xmax,
        ylabel,
        ymin,
        ymax,
        predictor,
        r2,
        filename):
    """Plot heatmap"""
    xcoords = np.linspace(xmin, xmax, 100)
    ycoords = np.linspace(ymin, ymax, 100)
    predictions = []
    for ycoord in ycoords:
        predictions.append(
            predictor.predict(
                np.array(
                    [
                        xcoords,
                        np.array([ycoord] * len(ycoords))]).T))
    fig, axis = plt.subplots(1, 1)
    plot = axis.matshow(
        np.array(predictions),
        cmap=plt.cm.YlGn)
    axis.grid()
    axis.set_title("$R^2$ = "+str(r2))
    axis.set_xlabel(xlabel)
    axis.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            make_format_function(xmin, xmax, 100)))
    axis.set_ylabel(ylabel)
    axis.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            make_format_function(ymin, ymax, 100)))
    plt.colorbar(plot)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


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
    _make_boxplot(
        [firsts, lasts],
        ['firsts', 'lasts'],
        filename,
        '',
        'Relative time (% of participant\'s longest labeling time)')


DECIMAL_PLACES = 5
def _plot_table(data, colorer, highlighter, filename):
    """Plot a table

     * data :: 2-D np.array
        the data to be tabulated
     * colorer :: float -> hsva
        function that determines the color of the cell
     * highlighter :: float -> boolean
        function that determines whether the numbers shown are highlighted or
        not
     * filename :: str
        ouptut file name
    """
    # http://stackoverflow.com/questions/10194482/custom-matplotlib-plot-chess-board-like-table-with-colored-cells
    fig, axis = plt.subplots(1, 1)
    axis.set_axis_off()
    table = matplotlib.table.Table(axis, bbox=[0, 0, 1, 1])
    nrows, ncols = data.shape
    width, height = 1.0/ncols, 1.0/nrows
    for (i, j), datum in np.ndenumerate(data):
        text = str(round(datum, DECIMAL_PLACES))
        table.add_cell(
            i,
            j,
            width,
            height,
            text=text,
            loc='center',
            facecolor=colorer(datum))
        color = (0.0, 0.0, 0.0, 0.75)
        weight = 'normal'
        if highlighter(datum):
            color = (0.0, 0.0, 0.0, 1.0)
            weight = 'bold'
        cell = table.get_celld()[(i, j)]
        cell.get_text().set_color(color)
        cell.get_text().set_weight(weight)
    for i in range(data.shape[0]):
        table.add_cell(
            i,
            -1,
            width,
            height,
            text=str(i+1),
            loc='right',
            edgecolor='none',
            facecolor='none')
    for j in range(data.shape[1]):
        table.add_cell(
            -1,
            j,
            width,
            height,
            text=str(j+1),
            loc='center',
            edgecolor='none',
            facecolor='none')
    axis.add_table(table)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _plot_matrix(data, cmap, colorbar, filename):
    """Plot a matrix

     * data :: 2-D np.array
        that data to be matricized
     * cmap :: matplotlib.colors.Colormap
        the colormap to use
     * colorbar :: boolean
        whether to display color bar
     * filename :: str
        output file name
    """
    fig, axis = plt.subplots(1, 1)
    plot = axis.matshow(data, cmap=cmap)
    if colorbar:
        plt.colorbar(plot)
    axis.grid(b=False)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _pval_colorer(pval_highlighter, cmap):
    """Returns function to color p-value cells"""
    def _inner(value):
        """Returns hsva color based on the value"""
        if pval_highlighter(value):
            return cmap(0.0)
        return cmap(1.0)
    return _inner


def _stat_colorer(stats, cmap):
    """Returns function to color statistic cells"""
    smin = np.min(stats)
    smax = np.max(stats)
    sdiff = smax - smin

    def _inner(value):
        """Returns hsva color based on the value"""
        return cmap((value - smin) / (sdiff))
    return _inner


def _compare_test(sampleses, comparer, filename):
    """Plots statistical test results

    Assuming that filename ends with ".pdf"
    """
    test_stats = []
    test_pvals = []
    for samples in sampleses:
        test_stats.append([])
        test_pvals.append([])
        for other in sampleses:
            stat, pval = comparer.stat_test(samples, other)
            test_stats[-1].append(stat)
            test_pvals[-1].append(pval)
    test_pvals = np.array(test_pvals)
    np.savetxt('test_pvals.txt', test_pvals)

    _plot_table(
        test_pvals,
        _pval_colorer(comparer.pval_highlighter, plt.cm.YlGn),
        comparer.pval_highlighter,
        filename[:-4]+'_'+comparer.test_name+'_pvals.pdf')

    sig_pvals = []
    for row in test_pvals:
        sig_pvals.append([])
        for val in row:
            sig_pvals[-1].append(1.0 if val < 0.05 else 0.0)
    _plot_table(
        np.array(test_stats),
        _stat_colorer(np.array(test_stats), plt.cm.YlGn_r),
        comparer.stat_highlighter,
        filename[:-4]+'_'+comparer.test_name+'_stats.pdf')


def _pad_num(i):
    """Pad number to make two characters long

    Assuming that i is an integer never greater than 99
    """
    if i < 10:
        return '0' + str(i)
    return str(i)


def _fit_gamma(sampleses, filename):
    """Fits a gamma distribution to the first 16 samples and plots the results

    Assuming that filename ends with ".pdf"
    """
    for i, samples in enumerate(sampleses[:16]):
        sample_mean = np.mean(samples)
        sample_var = np.var(samples)
        sample_median = np.median(samples)
        shape, loc, scale = gamma.fit(samples)
        stat, pval = kstest(
            samples,
            'gamma',
            args=(shape, loc, scale))
        fig, axis = plt.subplots(1, 1)
        axis.hist(samples, normed=True)
        if i == 15:
            fig.savefig('last.pdf')
        plotx = np.linspace(np.min(samples), np.max(samples))
        axis.plot(
            plotx,
            gamma.pdf(plotx, shape, loc=loc, scale=scale),
            linewidth=3)
        axis.set_title(
            'shape='+str(shape)+'; loc='+str(loc) +
            '; scale='+str(scale)+'\n' +
            'stat='+str(stat)+'; pval='+str(pval)+'\n' +
            'mean='+str(shape*scale)+'; var='+str(shape*scale*scale)+'\n' +
            's_mean='+str(sample_mean)+'; s_var='+str(sample_var)+'\n' +
            's_median='+str(sample_median))
        fig.savefig(
            filename[:-4]+'_fit_'+_pad_num(i+1)+'.pdf',
            bbox_inches='tight')
        plt.close()


def _plot_stats(sampleses, filename, xlabel, ylabel):
    """Plot sample means, medians, and variances for the first 16 samples

    Assuming that filename ends with ".pdf"
    Also spits out text file with data

        * sampleses :: [[float]]
        * filename :: str
    """
    means = []
    std_devs = []
    medians = []
    first_quartile = []
    third_quartile = []
    variances = []
    for samples in sampleses:
        means.append(np.mean(samples))
        std_devs.append(np.std(samples))
        medians.append(np.median(samples))
        first_quartile.append(np.percentile(samples, 25))
        third_quartile.append(np.percentile(samples, 75))
        variances.append(np.var(samples))
    ind = np.arange(len(sampleses)) + 1
    first_quartile_distance = np.array(medians) - np.array(first_quartile)
    third_quartile_distance = np.array(third_quartile) - np.array(medians)

    barwidth = 0.8
    xlim = [0.5, len(sampleses) + 0.5]

    fig, axis = plt.subplots(1, 1)
    axis.bar(
        ind,
        means,
        width=barwidth,
        alpha=0.9,
        yerr=std_devs,
        capsize=barwidth*2,
        align='center')
    axis.set_xlim(xlim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    fig.savefig(filename[:-4]+'_means.pdf', bbox_inches='tight')
    plt.close()

    fig, axis = plt.subplots(1, 1)
    axis.bar(
        ind,
        medians,
        width=barwidth,
        alpha=0.9,
        yerr=[first_quartile_distance, third_quartile_distance],
        capsize=barwidth*2,
        align='center')
    axis.set_xlim(xlim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    fig.savefig(filename[:-4]+'_medians.pdf', bbox_inches='tight')
    plt.close()

    fig, axis = plt.subplots(1, 1)
    axis.bar(
        ind,
        variances,
        width=barwidth,
        capsize=barwidth*2,
        alpha=0.9,
        align='center')
    axis.set_xlim(xlim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel('Variance')
    fig.savefig(filename[:-4]+'_vars.pdf', bbox_inches='tight')
    plt.close()

    with open(filename[:-4]+'_stats.txt', 'w') as ofh:
        ofh.write('# mean median variance\n')
        for men, med, vari in zip(means, medians, variances):
            ofh.write(str(men)+' '+str(med)+' '+str(vari)+'\n')


def _get_switch_indiceses(userdata):
    """Calculate switch indices for each user"""
    result = {}
    for user, data in userdata.items():
        switches = data[:-1, 2] != data[1:, 2]
        switches = np.insert(switches, 0, True)
        # get indices where switches occur
        result[user] = np.nonzero(switches)[0]
    return result


def _get_max_same(userdata, switch_indiceses):
    """Get the maxium number of documents with the same top topic in a row"""
    max_same = 0
    for user, data in userdata.items():
        switch_indices = switch_indiceses[user]
        same_counts = switch_indices[1:] - switch_indices[:-1]
        same_counts = np.append(
            same_counts,
            data[:, 2].shape[0] - switch_indices[-1])
        for count in same_counts:
            if count > max_same:
                max_same = count
    return max_same


def _get_max_topics(switch_indiceses):
    """Get the maximum number of topical groups"""
    max_topics = 0
    for _, data in switch_indiceses.items():
        if len(data) > max_topics:
            max_topics = len(data)
    return max_topics


def _build_data_times(_, data):
    """Get time data (in seconds)"""
    return (data[:, 1] - data[:, 0]) / 1000


def _build_data_reltimes(relative_times_by_user):
    """Build function for build relative times data"""
    def _inner(user, _):
        """Get relative times data"""
        return relative_times_by_user[user]
    return _inner


def _build_data_doclength(corpus):
    """Build function for build document length data"""
    def _inner(_, data):
        """Get document length data"""
        docids = data[:, 3]
        doclengths = [len(corpus[str(a)]['text'].split()) for a in docids]
        return doclengths
    return _inner


def _aggregate_order_data(userdata, switch_indiceses, max_same, build_data):
    """Aggregate data for _order_vs_* functions

    We want to collect data into bins such that the first bin has data with
    respect to the first documents in a topical group, etc.
    """
    result = [[] for _ in range(max_same)]
    for user, data in userdata.items():
        resultdata = build_data(user, data)
        switch_indices = switch_indiceses[user]
        same_counts = switch_indices[1:] - switch_indices[:-1]
        same_counts = np.append(
            same_counts,
            data[:, 2].shape[0] - switch_indices[-1])
        for i, switch in enumerate(switch_indices):
            for j in range(same_counts[i]):
                result[j].append(resultdata[switch+j])
    return result


def _aggregate_firsts_data(userdata, switch_indiceses, max_topics, build_data):
    """Aggregate data for _firsts_vs_* functions

    We want to collect data into bins such that the first bin has data with
    respect to the first document in the first topical groups, etc.
    """
    result = [[] for _ in range(max_topics)]
    for user, data in userdata.items():
        resultdata = build_data(user, data)
        switch_indices = switch_indiceses[user]
        for i, switch in enumerate(switch_indices):
            result[i].append(resultdata[switch])
    return result


def _order_vs_times(userdata, switch_indiceses, max_same, comparers, filename):
    """Analyze data and make plots of document number within topic vs. times

    Also plot statistical tests of first 16 in order against each other
    """
    result = _aggregate_order_data(
        userdata,
        switch_indiceses,
        max_same,
        _build_data_times)
    # I want information for the first 16 only
    _make_boxplot(
        result[:16],
        [str(i+1) for i in range(16)],
        filename,
        'Document order',
        'Time (seconds)')
    for comparer in comparers:
        _compare_test(result[:16], comparer, filename)
    _plot_stats(result[:16], filename, 'Document order', 'Time (seconds)')


def _firsts_vs_times(
        userdata,
        switch_indiceses,
        max_topics,
        comparers,
        filename):
    """Analyze data and make plots of first documents within topic vs. times

    Also plot statistical tests of first 5 in order against each other
    """
    result = _aggregate_firsts_data(
        userdata,
        switch_indiceses,
        max_topics,
        _build_data_times)
    # I want information for 5
    _make_boxplot(
        result[:5],
        [str(i+1) for i in range(5)],
        filename,
        'Topical group',
        'Time (seconds)')
    for comparer in comparers:
        _compare_test(result[:5], comparer, filename)
    _plot_stats(result[:5], filename, 'Topical group', 'Time (seconds)')


def _order_vs_doclength(
        userdata,
        switch_indiceses,
        max_same,
        corpus,
        comparers,
        filename):
    """Analyze data and make plots of document number within topic vs. document
    lengths

    Also plot statistical tests of first 16 in order against each other
    """
    result = _aggregate_order_data(
        userdata,
        switch_indiceses,
        max_same,
        _build_data_doclength(corpus))
    for comparer in comparers:
        _compare_test(result[:16], comparer, filename)


def _order_vs_reltimes(
        userdata,
        switch_indiceses,
        max_same,
        relative_times_by_user,
        comparers,
        filename):
    """Analyze data and make plots of document number within topic vs. relative
    times

    Also plot statistical tests of first 16 in order against each other
    """
    result = _aggregate_order_data(
        userdata,
        switch_indiceses,
        max_same,
        _build_data_reltimes(relative_times_by_user))
    # I want information for the first 16 only
    _make_boxplot(
        result[:16],
        [str(i+1) for i in range(16)],
        filename,
        'Document order',
        'Relative time (% of participant\'s longest labeling time)')
    for comparer in comparers:
        _compare_test(result[:16], comparer, filename)
    _plot_stats(
        result[:16],
        filename,
        'Document order',
        'Relative time (% of participant\'s longest labeling time)')


def _add_one(xval, _):
    """Add one"""
    return int(xval + 1)


def _regression_surface(
        userdata,
        switch_indiceses,
        corpus,
        filename):
    """Analyze data and make plot of document position and length vs. labeling
    time.
    """
    doclengths = []
    positions = []
    times = []
    for user, data in userdata.items():
        curdoclengths = _get_doclengths_for_user(userdata, user, corpus)
        switch_indices = switch_indiceses[user]
        user_times = _build_data_times(user, data)
        for i in range(1, len(switch_indices)):
            if switch_indices[i] - switch_indices[i-1] == 16:
                doclengths.extend(
                    curdoclengths[switch_indices[i-1]:switch_indices[i]])
                positions.extend(np.arange(1, 17))
                times.extend(
                    user_times[switch_indices[i-1]:switch_indices[i]])
    doclengths = np.array(doclengths)
    positions = np.array(positions)
    times = np.array(times)
    model_inputs = np.stack((doclengths, positions), axis=-1)
    ridge_model = Ridge()
    ridge_model.fit(model_inputs, times)
    r2 = ridge_model.score(model_inputs, times)
    fig, axis = plt.subplots(1, 1)
    xdata = np.arange(1, 17)
    for doclength in [30, 50, 100, 200, 500, 1000]:
        inputs = np.stack((np.array([doclength]*len(xdata)), xdata), axis=-1)
        ydata = ridge_model.predict(inputs)
        axis.plot(
            xdata,
            ydata,
            linewidth=2,
            label=str(doclength))
        # apparently, all of the lines go down by 6.02762577314 from first
        # labeling time to 16th
        # axis.annotate(str(ydata[0] - ydata[-1]), (xdata[-1], ydata[-1]))
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    legend = axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    legend.set_title('Document length (in tokens)')
    axis.set_title('$R^2=$'+str(r2))
    axis.set_xlabel('Document order')
    axis.set_ylabel('Time (seconds)')
    fig.savefig(filename, bbox_inches='tight')


class Comparer:
    """Class to encapsulate functions and data for statistical tests"""

    def __init__(
            self,
            test_name,
            stat_test,
            pval_highlighter,
            stat_highlighter):
        """Constructor"""
        self.test_name = test_name
        self.stat_test = stat_test
        self.pval_highlighter = pval_highlighter
        self.stat_highlighter = stat_highlighter


def _mannwhitneyu_helper(x, y):
    """Return common language effect size and p-value"""
    _, pval = mannwhitneyu(x, y, alternative='greater')
    count = 0
    for first in x:
        for second in y:
            if first > second:
                count += 1
    total = len(x) * len(y)
    return float(count) / float(total), pval


def _pval_256(val):
    """Check whether val < 0.05 / (16*16)"""
    return val < 0.0001953125


def _pval_25(val):
    """Check whether val < 0.05 / (5*5)"""
    return val < 0.002


def _abs_stat(val):
    """Check whether abs(val) > 0.6"""
    return abs(val) > 0.6


def _analyze_data(userdata, corpus, divergence, titles, outdir):
    """Analyze data"""
    true_labels_by_user = parsedata.get_true_labels_by_user(userdata, corpus)
    s_checker = DivergenceChecker(divergence, titles)
    relative_times_by_user = _get_relative_times(userdata)
    switch_indiceses = _get_switch_indiceses(userdata)
    max_same = _get_max_same(userdata, switch_indiceses)
    max_topics = _get_max_topics(switch_indiceses)
    # total time spent vs. final accuracy
    _totaltime_vs_finalscore(
        userdata,
        true_labels_by_user,
        os.path.join(outdir, 'totaltime_finalaccuracy.pdf'),
        _score_eval_helper(_accuracy),
        'Accuracy (% correct)')
    # total time spent vs. final average absolute error
    _totaltime_vs_finalscore(
        userdata,
        true_labels_by_user,
        os.path.join(outdir, 'totaltime_finalmae.pdf'),
        _score_eval_helper(_mean_absolute_error),
        'Mean Absolute Error')
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
        _other_eval_helper(_extract_time),
        'Time (seconds)')
    # JS divergence of switch topics vs. relative time spent
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_reltime.pdf'),
        _other_eval_helper(_extract_reltime),
        'Relative time (% of participant\'s longest labeling time)')
    # JS divergence of switch topics vs. running accuracy
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningaccdiff.pdf'),
        _other_eval_helper(_extract_runningaccdiff(true_labels_by_user)),
        'Change in running accuracy (% correct)')
    # JS divergence of switch topics vs. running total accuracy
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningtotalaccdiff.pdf'),
        _other_eval_helper(_extract_runningtotalaccdiff(true_labels_by_user)),
        'Change in total accuracy (% correct)')
    # JS divergence of switch topics vs. running mean error
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningmeanerrdiff.pdf'),
        _other_eval_helper(_extract_runningmeanerrdiff(true_labels_by_user)),
        'Change in running mean error')
    # JS divergence of switch topics vs. running total error
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningtotalerrdiff.pdf'),
        _other_eval_helper(_extract_runningtotalerrdiff(true_labels_by_user)),
        'Change in total mean error')
    # box plot:  relative time spent on switch, relative time spent not on
    # switch
    _switch_vs_not_relative(
        userdata,
        relative_times_by_user,
        os.path.join(outdir, 'switch_relativetimes.pdf'))
    # box plot:  time spent on switch/not on switch
    _switch_vs_not(
        userdata,
        os.path.join(outdir, 'switch_times.pdf'))
    # number of docs labeled vs. relative time spent
    _numlabeled_vs_reltime(
        userdata,
        relative_times_by_user,
        os.path.join(outdir, 'numlabeled_relativetimes.pdf'))
    # number of docs labeled vs. time spent
    _numlabeled_vs_time(
        userdata,
        os.path.join(outdir, 'numlabeled_times.pdf'))
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
    stat_tests = [
        Comparer('ks', ks_2samp, _pval_256, _pval_256),
        Comparer('mwu', _mannwhitneyu_helper, _pval_256, _abs_stat)]
    # box plot:  relative times per document within group
    _order_vs_reltimes(
        userdata,
        switch_indiceses,
        max_same,
        relative_times_by_user,
        stat_tests,
        os.path.join(outdir, 'order_reltime.pdf'))
    # box plot:  times per document within group
    _order_vs_times(
        userdata,
        switch_indiceses,
        max_same,
        stat_tests,
        os.path.join(outdir, 'order_time.pdf'))
    # statistical tests:  lengths per document within group
    _order_vs_doclength(
        userdata,
        switch_indiceses,
        max_same,
        corpus,
        stat_tests,
        os.path.join(outdir, 'order_doclength.pdf'))
    # box plot:  times per first documents after topic switch
    stat_tests_25 = [
        Comparer('ks', ks_2samp, _pval_256, _pval_256),
        Comparer('mwu', _mannwhitneyu_helper, _pval_256, _abs_stat)]
    _firsts_vs_times(
        userdata,
        switch_indiceses,
        max_topics,
        stat_tests_25,
        os.path.join(outdir, 'firsts_time.pdf'))
    # regression surface
    _regression_surface(
        userdata,
        switch_indiceses,
        corpus,
        os.path.join(outdir, 'regression_surface.pdf'))


def _run():
    """Run analysis"""
    args = _parse_args()
    userdata = parsedata.get_data(args.userdata)
    corpus = parsedata.grab_pickle(args.corpus)
    divergence = np.load(args.divergence)
    titles = parsedata.grab_pickle(args.titles)
    _analyze_data(userdata, corpus, divergence, titles, args.o)


if __name__ == '__main__':
    _run()
