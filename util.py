from typing import Tuple, List

import numpy as np
import pandas as pd

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

class CVManager:
    def __init__(self, reviews: List[str], labels: np.ndarray, n_folds: int=10) -> None:
        self.n_folds = n_folds

        # Shuffle data
        perm = np.random.permutation(reviews.shape[0])
        reviews = np.array(reviews)[perm]
        labels = labels[perm]

        # Create folds using equal slices from shuffled data
        idx_folds = np.array_split(np.arange(reviews.shape[0]), n_folds)
        self.X_slices = [reviews[idx] for idx in idx_folds]
        self.y_slices = [labels[idx] for idx in idx_folds]

        self.cur_index = 0

    def get_fold_data(self) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray]]:
        if self.cur_index >= self.n_folds:
            raise ValueError(f"CVManager.get_fold_data() called more times than specified number of folds (n_folds={self.n_folds})")

        train_X = np.concatenate([slice for i, slice in enumerate(self.X_slices) if i != self.cur_index], axis=0)
        train_y = np.concatenate([slice for i, slice in enumerate(self.y_slices) if i != self.cur_index], axis=0)

        val_X = self.X_slices[self.cur_index]
        val_y = self.y_slices[self.cur_index]

        self.cur_index += 1

        return (train_X, train_y), (val_X, val_y)
    
def update_index_token_list(original: List[str], indices: List) -> List[str]:
    token_array = np.array(original)
    return list(token_array[indices])

def perform_q_test(models: List) -> None:
    names = [model.name for model in models]
    # shape: (n_samples, n_models)
    prediction_results = np.vstack([
        model.correctness_vector for model in models
    ]).T

    n_models = len(models)
    pvals = np.ones((n_models, n_models))
    pair_indices = []

    # Compute all unique pairwise p-values
    raw_pvals = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            a, b = prediction_results[:, i], prediction_results[:, j]
            n01 = np.sum((a == 0) & (b == 1))
            n10 = np.sum((a == 1) & (b == 0))
            table = [[0, n01], [n10, 0]]
            p = mcnemar(table, exact=True, correction=True).pvalue
            raw_pvals.append(p)
            pair_indices.append((i, j))

    # Holm–Bonferroni correction
    corrected = multipletests(raw_pvals, method='holm')
    corrected_pvals = corrected[1]  # second element = adjusted p-values

    # Fill symmetric matrix
    for (i, j), p_corr in zip(pair_indices, corrected_pvals):
        pvals[i, j] = p_corr
        pvals[j, i] = p_corr

    # Format as DataFrame
    pval_df = pd.DataFrame(pvals, index=names, columns=names)
    print(pval_df.round(4))