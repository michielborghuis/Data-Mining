from typing import Dict

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mutual_info_score

from data_loading import ReviewLoader
from preprocessing import ReviewProcessor
from util import CVManager, update_index_token_dict

class Classifier:
    def __init__(self, min_df: float=.005, include_bigrams: bool=False, name: str="ReviewClassifier") -> None:
        self.min_df = min_df
        self.include_bigrams = include_bigrams
        self.name = name

        self.loader = ReviewLoader()
        self.processor = ReviewProcessor()

    def _initialize_model(self):
        # Model not defined for 'vanilla' classifier object
        self.model = None

    def get_performance_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str,float]:
        predictions = self.model.predict(X)

        performance_dict = {}
        performance_dict['accuracy'] = accuracy_score(y, predictions)
        performance_dict['precision'] = precision_score(y, predictions)
        performance_dict['recall'] = recall_score(y, predictions)
        performance_dict['f1'] = f1_score(y, predictions)

        return performance_dict

    def get_validation_performance(self, n_folds: int=10, n_repeats: int=1) -> Dict[str,float]:
        # Load reviews from files
        reviews, labels = self.loader.load_train_reviews()

        performance_dict = {
            "accuracy": np.zeros((n_repeats, n_folds)),
            "precision": np.zeros((n_repeats, n_folds)),
            "recall": np.zeros((n_repeats, n_folds)),
            "f1": np.zeros((n_repeats, n_folds))
        }

        for rep_i in range(n_repeats):
            cv_manager = CVManager(reviews=reviews, labels=labels, n_folds=n_folds)

            for fold_nr in range(cv_manager.n_folds):
                (train_X, train_y), (val_X, val_y) = cv_manager.get_fold_data()

                # Preprocess reviews to convert to count matrix
                train_X = self.processor.process_train_reviews(train_X, include_bigrams=self.include_bigrams)

                # Remove features (uni-/bigrams) that occur in very few documents
                train_X = self.processor.filter_rare_terms(train_X, min_review_freq=self.min_df)

                # Load test reviews
                val_X = self.processor.process_test_reviews(val_X, include_bigrams=self.include_bigrams)

                self._initialize_model()
                self.model.fit(train_X, train_y)

                fold_metrics = self.get_performance_metrics(val_X, val_y)

                performance_dict['accuracy'][rep_i][fold_nr] = fold_metrics['accuracy']
                performance_dict['precision'][rep_i][fold_nr] = fold_metrics['precision']
                performance_dict['recall'][rep_i][fold_nr] = fold_metrics['recall']
                performance_dict['f1'][rep_i][fold_nr] = fold_metrics['f1']

        return performance_dict
    
    def get_test_performance(self) -> Dict[str,float]:
        train_reviews, train_y = self.loader.load_train_reviews()

        train_X = self.processor.process_train_reviews(train_reviews, include_bigrams=self.include_bigrams)
        train_X = self.processor.filter_rare_terms(train_X, min_review_freq=self.min_df)

        test_reviews, test_y = self.loader.load_test_reviews()
        test_X = self.processor.process_test_reviews(test_reviews, include_bigrams=self.include_bigrams)

        self._initialize_model()
        self.model.fit(train_X, train_y)

        return self.get_performance_metrics(test_X, test_y)

class NaiveBayesClassifier(Classifier):
    def __init__(self, smoothing_alpha: float=1.0, drop_features: int=0, min_df: float=.005, include_bigrams: bool=False, name: str="NaiveBayes") -> None:
        super().__init__(min_df=min_df, include_bigrams=include_bigrams, name=name)
        self.alpha = smoothing_alpha
        self.min_df = min_df
        self.drop_features = drop_features
        self.include_bigrams = include_bigrams

    def _initialize_model(self) -> None:
        self.model = MultinomialNB(alpha=self.alpha, force_alpha=True)

    def select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Return indices of features that should be used
        if self.drop_features in [None, 0]:
            self.feature_indices = np.arange(X.shape[1])
        elif self.drop_features > X.shape[1]:
            raise ValueError(
                f"Total number of possible features {X.shape[1]} exceeds " + \
                f"specified number of features to drop {self.drop_features}"
            )
        else:
            binary_X = (X > 0)*1

            mi_scores = np.zeros(binary_X.shape[1])
            for feature_idx in range(binary_X.shape[1]):
                mi_scores[feature_idx] = mutual_info_score(binary_X[:,feature_idx], y)

            feature_score = mi_scores
            n_features = len(feature_score) - self.drop_features

            self.feature_indices = np.argpartition(feature_score, -n_features)[-n_features:]
        
    def get_validation_performance(self, n_folds: int=10, n_repeats: int=1) -> Dict[str,float]:
        # Load reviews from files
        reviews, labels = self.loader.load_train_reviews()

        performance_dict = {
            "accuracy": np.zeros((n_repeats, n_folds)),
            "precision": np.zeros((n_repeats, n_folds)),
            "recall": np.zeros((n_repeats, n_folds)),
            "f1": np.zeros((n_repeats, n_folds))
        }

        for rep_i in range(n_repeats):
            cv_manager = CVManager(reviews=reviews, labels=labels, n_folds=n_folds)

            for fold_nr in range(cv_manager.n_folds):
                (train_X, train_y), (val_X, val_y) = cv_manager.get_fold_data()

                # Preprocess reviews to convert to count matrix
                train_X = self.processor.process_train_reviews(train_X, include_bigrams=self.include_bigrams)

                # Remove features (uni-/bigrams) that occur in very few documents
                train_X = self.processor.filter_rare_terms(train_X, min_review_freq=self.min_df)

                # Load test reviews
                val_X = self.processor.process_test_reviews(val_X, include_bigrams=self.include_bigrams)

                self.select_features(train_X, train_y)

                train_X = train_X[:,self.feature_indices]
                val_X = val_X[:,self.feature_indices]

                self._initialize_model()
                self.model.fit(train_X, train_y)

                fold_metrics = self.get_performance_metrics(val_X, val_y)

                performance_dict['accuracy'][rep_i][fold_nr] = fold_metrics['accuracy']
                performance_dict['precision'][rep_i][fold_nr] = fold_metrics['precision']
                performance_dict['recall'][rep_i][fold_nr] = fold_metrics['recall']
                performance_dict['f1'][rep_i][fold_nr] = fold_metrics['f1']

        return performance_dict
    
    def get_test_performance(self) -> Dict[str,float]:
        train_reviews, train_y = self.loader.load_train_reviews()

        train_X = self.processor.process_train_reviews(train_reviews, include_bigrams=self.include_bigrams)
        train_X = self.processor.filter_rare_terms(train_X, min_review_freq=self.min_df)

        test_reviews, test_y = self.loader.load_test_reviews()
        test_X = self.processor.process_test_reviews(test_reviews, include_bigrams=self.include_bigrams)

        self.select_features(train_X, train_y)

        train_X = train_X[:,self.feature_indices]
        test_X = test_X[:,self.feature_indices]

        self._initialize_model()
        self.model.fit(train_X, train_y)

        return self.get_performance_metrics(test_X, test_y)

    def analyse_feature_importances(self) -> None:
        feature_mapping = self.processor.index_token_dict
        feature_mapping = update_index_token_dict(feature_mapping, self.feature_indices)

        log_probs = self.model.feature_log_prob_
        log_odds = log_probs[1] - log_probs[0]

        # Top for true reviews
        true_indices = np.argsort(log_odds)[::-1][:10]

        print(f"\nTop 10 features for true reviews:")
        for j in true_indices:
            fn = feature_mapping[j]
            lo = log_odds[j] 
            print(f"  {fn:<25} log-odds={lo:+.4f}  ")

        # Top for fake reviews
        fake_indices = np.argsort(log_odds)[:10]
        
        print(f"\nTop 10 features for fake reviews:")
        for j in fake_indices:
            fn = feature_mapping[j]
            lo = log_odds[j] 
            print(f"  {fn:<25} log-odds={lo:+.4f}  ")

class RandomForestClassifier(Classifier):
    def __init__(self, n_estimators: int = 100, name: str = "RandomForest") -> None:
        super().__init__(name)
        self.n_estimators = n_estimators

    def _initialize_model(self) -> None:
        self.model = SklearnRF(n_estimators=self.n_estimators, random_state=42)

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
