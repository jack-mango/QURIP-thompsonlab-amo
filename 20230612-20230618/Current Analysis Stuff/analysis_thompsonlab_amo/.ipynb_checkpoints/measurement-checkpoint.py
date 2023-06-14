import numpy as np
from ipywidgets import FloatProgress
from IPython.display import clear_output


class measurement:
    """ a measurement takes a dataset and does some operation on it. """

    def __init__(self, datasets, xvars, includefn=lambda x: True):
        """

        """
        try:
            iter(datasets)
        except TypeError:
            self.datasets = [datasets]
        else:
            self.datasets = datasets

        self.files = []
        self.fIdx_list = []
        self.xvars = xvars
        self.data = []

        self.checkIfFilesMatch()  # If they don't match, here it will raise TypeError.
        self.xVals = []
        for i, f0 in enumerate(self.datasets[0].files):
            if includefn(f0):  # only include files in the measurement that pass includefn
                self.xVals.append(list(map(lambda x: f0.params[x], self.xvars)))  # Construct list of values of xvars
                # for each set of files.
                self.fIdx_list.append(f0.params['fIdx'])
                if len(self.datasets) == 1:
                    self.files.append(f0)
                else:
                    self.files.append([d_set.files[i] for d_set in self.datasets])

        # group into occurrences to average, and make list of files corresponding to each unique value
        # if we want to treat files individually, could always add fIndex to xvars
        self.xVals_unique = np.unique(self.xVals, axis=0)
        self.x_raw = np.squeeze(self.xVals_unique)  # Used to plot out the figures.
        self.file_groups = []
        for y in self.xVals_unique:
            idx_y = filter(lambda idx: np.all(self.xVals[idx] == y), range(len(self.xVals)))
            self.file_groups.append(list(map(lambda idx: self.files[idx], idx_y)))

    def checkIfFilesMatch(self):
        fIdx_glob = [[f.params['fIdx'] for f in d_set.files] for d_set in self.datasets]
        for i, fIdx_list in enumerate(fIdx_glob[1:], start=1):
            if fIdx_list != fIdx_glob[0]:
                redundancy = list(set(fIdx_list) - set(fIdx_glob[0]))
                missing = list(set(fIdx_glob[0]) - set(fIdx_list))
                raise RuntimeError('The dataset #{:.0f} (length={:.0f}) and doesn\'t have the same files as dataset '
                                   '#0!\n'.format(i, len(fIdx_list), len(fIdx_glob[0])) +
                                   'fIdx\'s found in dataset #{:.0f} but not in dataset #0: '.format(i) +
                                   str(redundancy) + '\n' +
                                   'fIdx\'s found in dataset #0 but not in dataset #{:.0f}: '.format(i) + str(missing)
                                   )
        return True

    def getParam(self, params, array=True):
        res = []
        for i in self.datasets.allParamsDict:
            res.append(i[params][0])
        return res

    def measure(self, fn, fn_args, resultType=None, clear=True):
        """ apply a measurement function on file_groups, averaging result across all files in group """
        n = len(self.file_groups)
        status = FloatProgress(min=0, max=n)
        result = []
        fn_args.update({'raw': True})
        for fg in self.file_groups:
            if clear:
                clear_output(wait=True)
            meas_func = lambda x: fn(x, fn_args)
            measResult = list(map(meas_func, fg))  # [shape of meas_func's return] * (#files with the specific params)
            # resultPerX = np.transpose(measResult)
            result.append(measResult)

        if resultType is None:
            resultType = fn('resultType')

        if resultType == 'raw':
            return result
        elif resultType == 'average':  # Elements of "result" should be like: [x1, x2, ... xm].
            y = np.array(mapFn(lambda x: np.mean(x, axis=0), result))  # Averaged counts.
            yerr = np.array(mapFn(mean_value_err, result))  # Uncertainty based on theory of repetitive measurements.
            return y, yerr  # Return the average.
        elif resultType == 'sum':  # Elements of "result" should be like: [x1, x2, ... xm].
            y = np.array(mapFn(lambda x: np.sum(x, axis=0), result))  # Summed counts.
            yerr = np.array(mapFn(lambda x: mean_value_err(x) * len(x), result))  # Uncertainty based on theory of
            # repetitive measurements.
            return y, yerr  # Return the sum.
        elif resultType == 'survival':  # Elements of "result" should be like: [[x1, y1], [x2, y2], ... [xm, ym]].
            # [y1+y2+...+ym] * nParams:
            n_0 = np.array(mapFn(lambda z1: np.sum(mapFn(lambda z2: z2[1], z1), axis=0), result))
            # [x1+x2+...+xm] * nParams:
            n_1 = np.array(mapFn(lambda z1: np.sum(mapFn(lambda z2: z2[0], z1), axis=0), result))
            # The stupid division below is to circumvent the rules in numpy for the division of two numpy arrays.
            p = np.array([y / x for x, y in zip(n_0, n_1)])  # Survival rate.
            perr = np.sqrt([y / x for x, y in zip(n_0, p * (1 - p))])  # One-sigma error bar of a binary distribution.
            return p, perr
        else:
            raise ValueError('Unknown result type: {:s}!'.format(resultType))

    def xVals_fine(self, step, idx=0):
        """ generate a list covering min(xvals_unique[idx]) to max(xvals_unique[idx]) in steps of step, for plotting
        theory curves"""
        return np.arange(min(self.xVals_unique[:, idx]), max(self.xVals_unique[:, idx]), step)

    def load(self):
        """ load the data, but only if we haven't already """
        if self.rawdata is None:
            self.rawdata = np.fromfile(self.fullname, dtype='uint64')

    def matchParams(self, keys, vals):
        """ check if params specified in keys match vals """
        for k, v in zip(keys, vals):
            if self.params[k] != v:
                return False
        return True


def mapFn(fn, arg):
    return list(map(fn, arg))


def mean_value_err(x):
    if len(x) == 1:
        return np.nan * np.ones_like(x[0])
    else:
        return np.std(x, axis=0) / np.sqrt(len(x) - 1)