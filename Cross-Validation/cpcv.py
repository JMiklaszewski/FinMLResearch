import numpy as np
from scipy.special import comb
from itertools import combinations

class CombinatorialPurgedGroupKFold:
    """
    Implements Purged and Embargoed Combinatorial Cross-Validation as described in
    "Advances in Financial Machine Learning" by Marcos Lopez de Prado, 2018.

    This class decomposes samples into n_splits folds with equal numbers of samples,
    without shuffling. In each cross-validation round, n_test_splits folds are used
    as the test set, while the other folds form the train set. This approach helps
    prevent contamination of the test set by the train set in the presence of
    temporal correlation.

    Parameters:
    -----------
    n_splits : int, default=6
        Total number of folds. Must be at least 2.
    n_test_splits : int, default=2
        Number of folds used in the test set. Must be at least 1 and less than n_splits.
    purge : int, default=1
        Number of groups to purge from the beginning and end of each fold.
    pctEmbargo : float, default=0.01
        Percentage of groups to embargo after each test set.

    Attributes:
    -----------
    n_splits : int
        Total number of folds.
    n_test_splits : int
        Number of folds used in the test set.
    purge : int
        Number of groups to purge.
    pctEmbargo : float
        Percentage of groups to embargo.
    """

    def __init__(self, n_splits=6, n_test_splits=2, purge=1, pctEmbargo=0.01, **kwargs):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge = purge
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters:
        -----------
        X : pd.DataFrame, shape (n_samples, n_features)
            The input samples. Used only to extract n_samples.
        y : pd.Series, optional
            The target variable. Not used, present for API consistency.
        groups : array-like, shape (n_samples,)
            Group labels for the samples. Used to ensure that the same group is not
            in both training and test sets.

        Yields:
        -------
        train_indices : np.ndarray
            The training set indices for that split.
        test_indices : np.ndarray
            The testing set indices for that split.

        Raises:
        -------
        ValueError
            If groups is None, n_folds > n_groups, or if embargo results in negative groups.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")

        # Identify unique groups and their order
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_groups = len(unique_groups)

        # Create a dictionary mapping each group to its sample indices
        group_dict = {}
        for idx in range(len(X)):
            if groups[idx] in group_dict:
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]

        # Calculate the number of folds and validate against number of groups
        n_folds = comb(self.n_splits, self.n_test_splits, exact=True)
        if n_folds > n_groups:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater than"
                f" the number of groups={n_groups}")

        # Calculate embargo size and validate
        mbrg = int(n_groups * self.pctEmbargo)
        if mbrg < 0:
            raise ValueError("The number of 'embargoed' groups should not be negative")

        # Split groups into folds
        # Initialize an empty dictionary to store the groups for each split
        split_dict = {}

        # Calculate the number of groups that should be in each split
        # This ensures an approximately equal distribution of groups across splits
        group_test_size = n_groups // self.n_splits

        # Iterate through each split
        for split in range(self.n_splits):
            if split == self.n_splits - 1:
                # For the last split, include all remaining groups
                # This handles cases where n_groups is not perfectly divisible by n_splits
                split_dict[split] = unique_groups[int(split * group_test_size):].tolist()
            else:
                # For all other splits, take a slice of unique_groups
                # The slice starts at the current split's starting index
                # and ends at the next split's starting index
                split_dict[split] = unique_groups[int(split * group_test_size):int((split + 1) * group_test_size)].tolist()
        
        # Generate train and test indices for each combination of test splits
        for test_splits in combinations(range(self.n_splits), self.n_test_splits):
            # Initialize lists to store test and banned groups for the current combination
            test_groups = []
            banned_groups = []
    
            # Iterate over each split in the current combination of test splits
            for split in test_splits:
                # Add the groups corresponding to the current test split to the test_groups list
                test_groups += split_dict[split]
        
                # Calculate and add the groups to be purged before the current test split
                banned_groups += unique_groups[split_dict[split][0] - self.purge:split_dict[split][0]].tolist()
        
                # Calculate and add the groups to be embargoed after the current test split
                banned_groups += unique_groups[split_dict[split][-1] + 1:split_dict[split][-1] + self.purge + mbrg + 1].tolist()
    
            # Determine the train groups by excluding test and banned groups from all unique groups
            train_groups = [i for i in unique_groups if (i not in banned_groups) and (i not in test_groups)]

            # Initialize lists to store train and test indices
            train_idx = []
            test_idx = []
    
            # Collect indices for all samples in the train groups
            for train_group in train_groups:
                train_idx += group_dict[train_group]
    
            # Collect indices for all samples in the test groups
            for test_group in test_groups:
                test_idx += group_dict[test_group]
    
            # Yield the train and test indices for the current combination of test splits
            yield train_idx, test_idx