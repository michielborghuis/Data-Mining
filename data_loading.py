from typing import Tuple
import os

import numpy as np

class ReviewLoader:
    def __init__(self, base_dir: str="data/negative_polarity") -> None:
        self.base_dir = base_dir

    def _load_reviews_from_fold(self, fold_nr: int) -> Tuple[np.ndarray]:
        reviews = []
        labels = []

        for filename in os.listdir(f"{self.base_dir}/truthful_from_Web/fold{fold_nr}"):
            with open(f"{self.base_dir}/truthful_from_Web/fold{fold_nr}/{filename}", 'r') as f:
                reviews.append(f.read())
                labels.append(1)

        for filename in os.listdir(f"{self.base_dir}/deceptive_from_MTurk/fold{fold_nr}"):
            with open(f"{self.base_dir}/deceptive_from_MTurk/fold{fold_nr}/{filename}", 'r') as f:
                reviews.append(f.read())
                labels.append(0)

        return np.array(reviews), np.array(labels)

    def load_train_reviews(self) -> Tuple[np.ndarray]:
        reviews = np.array([], dtype=str)
        labels = np.array([], dtype=int)

        for fold_nr in range(1, 5):
            fold_reviews, fold_labels = self._load_reviews_from_fold(fold_nr=fold_nr)
            reviews = np.concatenate([reviews, fold_reviews])
            labels = np.concatenate([labels, fold_labels])

        return np.array(reviews), np.array(labels)
    
    def load_test_reviews(self) -> Tuple[np.ndarray]:
        return self._load_reviews_from_fold(fold_nr=5)
