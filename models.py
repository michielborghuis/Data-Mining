from typing import Dict

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from util import CVManager

class Classifier:
    def __init__(self, name: str) -> None:
        self.name = name

    def _initialize_model(self):
        # Model not defined for 'vanilla' classifier object
        self.model = None

    def _get_performace_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str,float]:
        predictions = self.model.predict(X)

        performance_dict = {}
        performance_dict['accuracy'] = accuracy_score(y, predictions)
        performance_dict['precision'] = precision_score(y, predictions)
        performance_dict['recall'] = recall_score(y, predictions)
        performance_dict['f1'] = f1_score(y, predictions)

        return performance_dict

    def get_validation_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str,float]:
        cv_manager = CVManager(X=X, y=y)

        performance_dict = {
            "accuracy":[],
            "precision":[],
            "recall":[],
            "f1":[]
        }

        for fold_nr in range(cv_manager.n_folds):
            (train_X, train_y), (val_X, val_y) = cv_manager.get_fold_data()

            self._initialize_model()
            self.model.fit(train_X, train_y)

            fold_metrics = self._get_performace_metrics(val_X, val_y)

            performance_dict['accuracy'].append(fold_metrics['accuracy'])
            performance_dict['precision'].append(fold_metrics['precision'])
            performance_dict['recall'].append(fold_metrics['recall'])
            performance_dict['f1'].append(fold_metrics['f1'])

        performance_dict['accuracy'] = np.mean(performance_dict['accuracy'])
        performance_dict['precision'] = np.mean(performance_dict['precision'])
        performance_dict['recall'] = np.mean(performance_dict['recall'])
        performance_dict['f1'] = np.mean(performance_dict['f1'])

        return performance_dict

class NaiveBayesClassifier(Classifier):
    def __init__(self, smoothing_alpha: float=1.0, name: str="NaiveBayes") -> None:
        super().__init__(name)
        self.alpha = smoothing_alpha

    def _initialize_model(self) -> None:
        self.model = MultinomialNB(alpha=self.alpha, force_alpha=True)

    def analyse_feature_importances(self, index_to_word_mapping: Dict[int,str]) -> None:
        log_probs = self.model.feature_log_prob_
        probs = np.exp(log_probs)
        prob_ratios = probs[1]/probs[0]
        indices = np.argsort(prob_ratios)
        max_len = np.max([len(token) for token in index_to_word_mapping.values()])

        print('MOST IMPORTANT FEATURES FOR NEGATIVE CLASSIFICATIONS:\n')
        print('Token'.ljust(max_len + 2)+'|\tP(x|y=1)/P(x|y=0)')
        print('-'*35)
        for i in indices[:20]:
            token = index_to_word_mapping[i]
            print(f"{token}".ljust(max_len + 2)+f"|\t{prob_ratios[i]:.3f}")

        print("")
        print('#'*35)
        print("")

        print('MOST IMPORTANT FEATURES FOR POSITIVE CLASSIFICATIONS:\n')
        print('Token'.ljust(max_len + 2)+'|\tP(x|y=1)/P(x|y=0)')
        print('-'*35)
        for i in indices[-20:]:
            token = index_to_word_mapping[i]
            print(f"{token}".ljust(max_len + 2)+f"|\t{prob_ratios[i]:.3f}")

        print("")
        print('#'*35)
        print("")

        low_ratios = prob_ratios[np.where(prob_ratios < 1.25)]
        moderate_ratios = low_ratios[np.where(low_ratios > .8)]

        print(f"{len(moderate_ratios)}/{len(prob_ratios)} FEATURES APPEAR NON-DISCRIMINATIVE (0.8 < PROB RATIO < 1.25)")
