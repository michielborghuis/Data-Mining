from typing import Dict

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier as SklearnRF, GradientBoostingClassifier as SklearnGB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mutual_info_score

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

    def get_validation_performance(self, X: np.ndarray, y: np.ndarray, n_features: int=None) -> Dict[str,float]:
        cv_manager = CVManager(X=X, y=y, n_folds=10)

        performance_dict = {
            "accuracy":[],
            "precision":[],
            "recall":[],
            "f1":[]
        }

        for fold_nr in range(cv_manager.n_folds):
            (train_X, train_y), (val_X, val_y) = cv_manager.get_fold_data()

            ###     START FEATURE SELECTION SECTION         ###
            
            ### CURRENT FEATURE UTILITY: MUTUAL INFORMATION ###

            if n_features not in [train_X.shape[1], None]:
                binary_X = (train_X > 0)*1

                #doc_freqs = np.sum(binary_X/binary_X.shape[0], axis=0)

                mi_scores = np.zeros(binary_X.shape[1])
                for feature_idx in range(binary_X.shape[1]):
                    mi_scores[feature_idx] = mutual_info_score(train_X[:,feature_idx], train_y)

                feature_score = mi_scores#*doc_freqs
                good_feature_ids = np.argpartition(feature_score, -n_features)[-n_features:]

                train_X = train_X[:,good_feature_ids]
                val_X = val_X[:,good_feature_ids]

            ### END FEATURE SELECTION SECTION ###

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

class RandomForestClassifier(Classifier):
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float | str = "sqrt",
        name: str = "RandomForest"
    ) -> None:
        super().__init__(name)
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features

    def _initialize_model(self) -> None:
        self.model = SklearnRF(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=42
        )

    def analyse_feature_importances(self, index_to_word_mapping: Dict[int,str], top_n: int = 20) -> None:
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        max_len = np.max([len(token) for token in index_to_word_mapping.values()])

        print(f"Top {top_n} Important Features (Random Forest):\n")
        print('Token'.ljust(max_len + 2)+'|\tImportance')
        print('-'*35)
        for i in indices[:top_n]:
            token = index_to_word_mapping.get(i, str(i))
            print(f"{token}".ljust(max_len + 2)+f"|\t{importances[i]:.5f}")
        print("")

class GradientBoostingClassifier(Classifier):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        max_features: str | None = None,
        name: str = "GradientBoosting"
    ) -> None:
        super().__init__(name)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_features = max_features

    def _initialize_model(self) -> None:
        self.model = SklearnGB(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            max_features=self.max_features,
            random_state=42
        )

    def analyse_feature_importances(self, index_to_word_mapping: Dict[int,str], top_n: int = 20) -> None:
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        max_len = max(len(str(token)) for token in index_to_word_mapping.values())

        print(f"Top {top_n} Important Features (Gradient Boosting):\n")
        print('Token'.ljust(max_len + 2)+'|\tImportance')
        print('-'*35)
        for i in indices[:top_n]:
            token = index_to_word_mapping.get(i, str(i))
            print(f"{token}".ljust(max_len + 2)+f"|\t{importances[i]:.5f}")
        print("")
