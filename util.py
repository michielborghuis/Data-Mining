from typing import Dict, Tuple, List

import numpy as np

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
    
def update_index_token_dict(original: Dict[int,str], indices: List) -> List[str]:
    token_list = ["PLACEHOLDER"] * (max(original.keys())+1)
    for i in range(len(token_list)):
        token_list[i] = original[i]
    
    token_array = np.array(token_list)
    return list(token_array[indices])