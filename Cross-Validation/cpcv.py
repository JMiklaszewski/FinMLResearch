#from itertools import combinations
import numbers
import itertools as itt
from abc import abstractmethod
from typing import Iterable, Tuple, List
import numpy as np
from sklearn.model_selection._split import indexable, _BaseKFold, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
import pandas as pd


class BaseTimeSeriesCrossValidator:
    """
    Abstract class for time series cross-validation.
    Time series cross-validation requires each sample has a prediction time pred_time, at which the features are used to
    predict the response, and an evaluation time eval_time, at which the response is known and the error can be
    computed. Importantly, it means that unlike in standard sklearn cross-validation, the samples X, response y,
    pred_times and eval_times must all be pandas dataframe/series having the same index. It is also assumed that the
    samples are time-ordered with respect to the prediction time (i.e. pred_times is non-decreasing).
    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.
    """
    def __init__(self, n_splits=10):


        '''This (abstract) method initializes the CV object with specified number of splits'''

        if not isinstance(n_splits, int): # Changed the integral type to int in this check
            raise ValueError(f"The number of folds must be of Integral type. {n_splits} of type {type(n_splits)}"
                             f" was passed.")
        n_splits = int(n_splits)
        if n_splits <= 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting n_splits = 2 "
                             f"or more, got n_splits = {n_splits}.")
        self.n_splits = n_splits
        self.pred_times = None
        self.eval_times = None
        self.indices = None

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series = None,
              pred_times: pd.Series = None, eval_times: pd.Series = None):

        ''' This is an abstract method for splitting X and y (optional) to prediction and evalutation
        indexes. Keep in mind this is an abstract (ie. blueprint) function which sets up the structure
        of methods wich will be inherited, but is not to be used by itself'''


        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series):
            raise ValueError('X should be a pandas DataFrame/Series.')
        if not isinstance(y, pd.Series) and y is not None:
            raise ValueError('y should be a pandas Series.')
        if not isinstance(pred_times, pd.Series):
            raise ValueError('pred_times should be a pandas Series.')
        if not isinstance(eval_times, pd.Series):
            raise ValueError('eval_times should be a pandas Series.')
        if y is not None and (X.index == y.index).sum() != len(y):
            raise ValueError('X and y must have the same index')
        if (X.index == pred_times.index).sum() != len(pred_times):
            raise ValueError('X and pred_times must have the same index')
        if (X.index == eval_times.index).sum() != len(eval_times):
            raise ValueError('X and eval_times must have the same index')

        if not pred_times.equals(pred_times.sort_values()):
            raise ValueError('pred_times should be sorted')
        if not eval_times.equals(eval_times.sort_values()):
            raise ValueError('eval_times should be sorted')

        self.pred_times = pred_times
        self.eval_times = eval_times
        self.indices = np.arange(X.shape[0])

class CombPurgedKFoldCVLocal(BaseTimeSeriesCrossValidator):
    """
    Purged and embargoed combinatorial cross-validation
    As described in Advances in financial machine learning, Marcos Lopez de Prado, 2018.
    The samples are decomposed into n_splits folds containing equal numbers of samples, without shuffling. In each cross
    validation round, n_test_splits folds are used as the test set, while the other folds are used as the train set.
    There are as many rounds as n_test_splits folds among the n_splits folds.
    Each sample should be tagged with a prediction time pred_time and an evaluation time eval_time. The split is such
    that the intervals [pred_times, eval_times] associated to samples in the train and test set do not overlap. (The
    overlapping samples are dropped.) In addition, an "embargo" period is defined, giving the minimal time between an
    evaluation time in the test set and a prediction time in the training set. This is to avoid, in the presence of
    temporal correlation, a contamination of the test set by the train set.
    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.
    n_test_splits : int, default=2
        Number of folds used in the test set. Must be at least 1.
    embargo_td : pd.Timedelta, default=0
        Embargo period (see explanations above).
    """
    def __init__(self, n_splits=10, n_test_splits=2, embargo_td=pd.Timedelta(minutes=0)):

        ''' This function checks the integrity and values of number of folds, number of folds
        used for testing and embargo period (ie. time shift before end of test batch and start of another
        train batch)'''

        super().__init__(n_splits) # Inherit from parent Abstract Base class (super)

        # As confirmed by ChatGPT, this check is not needed as it's present in the parent class
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(f"The number of test folds must be of Integral type. {n_test_splits} of type "
                             f"{type(n_test_splits)} was passed.")

        n_test_splits = int(n_test_splits)
        if n_test_splits <= 0 or n_test_splits > self.n_splits - 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"n_test_splits between 1 and n_splits - 1, got n_test_splits = {n_test_splits}.")
        self.n_test_splits = n_test_splits

        if not isinstance(embargo_td, pd.Timedelta):
            raise ValueError(f"The embargo time should be of type Pandas Timedelta. {embargo_td} of type "
                             f"{type(embargo_td)} was passed.")
        if embargo_td < pd.Timedelta(minutes=0):
            raise ValueError(f"The embargo time should be positive, got embargo = {embargo_td}.")
        self.embargo_td = embargo_td

    def split(self, X: pd.DataFrame, y: pd.Series = None,
              pred_times: pd.Series = None, eval_times: pd.Series = None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield the indices of the train and test sets.
        Although the samples are passed in the form of a pandas dataframe, the indices returned are position indices,
        not labels.
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Samples. Only used to extract n_samples.
        y : pd.Series, not used, inherited from _BaseKFold
        pred_times : pd.Series, shape (n_samples,), required
            Times at which predictions are made. pred_times.index has to coincide with X.index.
        eval_times : pd.Series, shape (n_samples,), required
            Times at which the response becomes available and the error can be computed. eval_times.index has to
            coincide with X.index.
        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.
        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.
        """
        super().split(X, y, pred_times, eval_times) # Inherit from parent Abstract Base class (super)

        # Fold boundaries

        '''
        # np.array_split splits array into n sub-arrays, the only difference between this and simple np.split, is
        # that in this case, numpy allows using n-splits that doesn't divide array equally
        # For more details, please see: https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        '''

        fold_bounds = [(fold[0], fold[-1] + 1) for fold in np.array_split(self.indices, self.n_splits)]
        # List of all combinations of n_test_splits folds selected to become test sets


        '''
        Since itertools.combinations will give you all the possible combinations of the iterable,
        in our case it equates to producing all possible combinations of fold boundaries (ie. start/end of
        each fold)
        '''
        selected_fold_bounds = list(itt.combinations(fold_bounds, self.n_test_splits))
        # In order for the first round to have its whole test set at the end of the dataset
        selected_fold_bounds.reverse()

        for fold_bound_list in selected_fold_bounds:
            # Computes the bounds of the test set, and the corresponding indices
            test_fold_bounds, test_indices = self.compute_test_set(fold_bound_list)
            # Computes the train set indices
            train_indices = self.compute_train_set(test_fold_bounds, test_indices)

            yield train_indices, test_indices


    def compute_test_set(self, fold_bound_list: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Compute the indices of the samples in the test set.
        Parameterst
        ----------
        fold_bound_list: List of tuples of position indices
            Each tuple records the bounds of the folds belonging to the test set.
        Returns
        -------
        test_fold_bounds: List of tuples of position indices
            Like fold_bound_list, but witest_fold_boundsth the neighboring folds in the test set merged.
        test_indices: np.ndarray
            A numpy array containing the test indices.
        """
        test_indices = np.empty(0) # empty (zero-shaped) numpy array
        test_fold_bounds = []
        for fold_start, fold_end in fold_bound_list:
            # Records the boundaries of the current test split
            if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]: # if it's the first fold or if it's not after a prevoius fold
                test_fold_bounds.append((fold_start, fold_end)) # update initialized list with start and end indexes of a test set
            # If the current test split is contiguous to the previous one, simply updates the endpoint
            elif fold_start == test_fold_bounds[-1][-1]:
                test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end) # if it's the next one, update the endpoint of previous test fold in list
            test_indices = np.union1d(test_indices, self.indices[fold_start:fold_end]).astype(int) # merge all the test indices with making sure that there are no duplicates in the provess
        return test_fold_bounds, test_indices


    def compute_train_set(self, test_fold_bounds: List[Tuple[int, int]], test_indices: np.ndarray) -> np.ndarray:
        """
        Compute the position indices of samples in the train set.
        Parameters
        ----------
        test_fold_bounds : List of tuples of position indices
            Each tuple records the bounds of a block of indices in the test set.
        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.
        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.
        """


        '''
        In order to filter for indices that are not present in the test set
        we will use np.setdiff1d that returns 'unmatched' part from first set:

        >>> a = np.array([1, 2, 3, 2, 4, 1])
        >>> b = np.array([3, 4, 5, 6])
        >>> np.setdiff1d(a, b)

        array([1, 2])

        '''

        # As a first approximation, the train set is the complement of the test set
        train_indices = np.setdiff1d(self.indices, test_indices)
        # But we now have to purge and embargo
        for test_fold_start, test_fold_end in test_fold_bounds:
            # Purge
            train_indices = self.purge(self, train_indices, test_fold_start, test_fold_end)
            # Embargo
            train_indices = self.embargo(self, train_indices, test_indices, test_fold_end)

        return train_indices


    '''
    I have checked both the function as well as the original ColabNotebook with implementation
    and it appears that following compute_fold_bounds function is never used neither in class nor
    in later parts of code
    '''

    def compute_fold_bounds(cv: BaseTimeSeriesCrossValidator, split_by_time: bool) -> List[int]:
        """
        Compute a list containing the fold (left) boundaries.
        Parameters
        ----------
        cv: BaseTimeSeriesCrossValidator
            Cross-validation object for which the bounds need to be computed.
        split_by_time: bool
            If False, the folds contain an (approximately) equal number of samples. If True, the folds span identical
            time intervals.
        """
        if split_by_time:
            full_time_span = cv.pred_times.max() - cv.pred_times.min()
            fold_time_span = full_time_span / cv.n_splits
            fold_bounds_times = [cv.pred_times.iloc[0] + fold_time_span * n for n in range(cv.n_splits)]

            return cv.pred_times.searchsorted(fold_bounds_times)
        else:
            return [fold[0] for fold in np.array_split(cv.indices, cv.n_splits)]

    def purge(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray,
              test_fold_start: int, test_fold_end: int) -> np.ndarray:
        """
        Purge part of the train set.
        Given a left boundary index test_fold_start of the test set, this method removes from the train set all the
        samples whose evaluation time is posterior to the prediction time of the first test sample after the boundary.
        Parameters
        ----------combinatorial purged k fold
        cv: Cross-validation class
            Needs to have the attributes cv.pred_times, cv.eval_times and cv.indices.
        train_indices: np.ndarray
            A numpy array containing all the indices of the samples currently included in the train set.
        test_fold_start : int
            Index corresponding to the start of a test set block.
        test_fold_end : int
            Index corresponding to the end of the same test set block.
        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing the train indices purged at test_fold_start.
        """

        # Identify the start of the test block
        time_test_fold_start = cv.pred_times.iloc[test_fold_start]
        # The train indices before the start of the test fold, purged.
        # In this line we cut of all observations from test set which would be evaluated on ground-truth data that lies in the test set
        train_indices_1 = np.intersect1d(train_indices, cv.indices[cv.eval_times < time_test_fold_start])
        # The train indices after the end of the test fold.
        # Now we joined purged training data before test set with all training samples that happend after training finishes
        train_indices_2 = np.intersect1d(train_indices, cv.indices[test_fold_end:])

        # Once done, we can now merge training indices from before and after trainig set after pruning
        return np.concatenate((train_indices_1, train_indices_2))

    def embargo(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray,
                test_indices: np.ndarray, test_fold_end: int) -> np.ndarray:
        """
        Apply the embargo procedure to part of the train set.
        This amounts to dropping the train set samples whose prediction time occurs within self.embargo_dt of the test
        set sample evaluation times. This method applies the embargo only to the part of the training set immediately
        following the end of the test set determined by test_fold_end.
        Parameters
        -------mestamps of p[t-1] values
      df0 = prices.inde---
        cv: Cross-validation class
            Needs to have the attributes cv.pred_times, cv.eval_times, cv.embargo_dt and cv.indices.
        train_indices: np.ndarray
            A numpy array containing all the indices of the samples currently included in the train set.
        test_indices : np.ndarray
            A numpy array containing all the indices of the samples in the test set.
        test_fold_end : int
            Index corresponding to the end of a test set block.
        Returns
        -------
        train_indices: np.ndarray
            The same array, with the indices subject to embargo removed.
        """
        if not hasattr(cv, 'embargo_td'): # Check if currently used object instance (cv) already has an embargo_td (embargo time delay) argument
            raise ValueError("The passed cross-validation object should have a member cv.embargo_td defining the embargo"
                             "time.")

        # Get the last time of evaluation in test sample (ie. start of embargo period)
        last_test_eval_time = cv.eval_times.iloc[test_indices[test_indices <= test_fold_end]].max()
        # Get the first index of train set after embargo period is imposed (ie. end of embargo period)
        min_train_index = len(cv.pred_times[cv.pred_times <= last_test_eval_time + cv.embargo_td])
        # Check if there are any train index left after embargo period
        if min_train_index < cv.indices.shape[0]:
            # Filter out embargoed indexes by concatenating data until end of training batch and start of train again after emabrgo period
            allowed_indices = np.concatenate((cv.indices[:test_fold_end], cv.indices[min_train_index:]))
            # With that information filter out embargoed index in train set by takin intersercion between set of all training indices and set embargoed indices
            train_indices = np.intersect1d(train_indices, allowed_indices)

        return train_indices
