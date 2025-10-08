from typing import Dict, Literal

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier as SklearnRF, GradientBoostingClassifier as SklearnGB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mutual_info_score

from data_loading import ReviewLoader
from preprocessing import ReviewProcessor
from util import CVManager, update_index_token_dict

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

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
    def __init__(self, 
                 smoothing_alpha: float=1.0, 
                 features: float=None, 
                 feature_mode: Literal["include", "drop", "fraction"]="include",
                 min_df: float=.005, 
                 include_bigrams: bool=False, 
                 name: str="NaiveBayes") -> None:
        
        super().__init__(min_df=min_df, include_bigrams=include_bigrams, name=name)
        self.alpha = smoothing_alpha
        self.min_df = min_df
        self.feature_param = features
        self.feature_mode = feature_mode
        if self.feature_mode not in ["include", "drop", "fraction"]:
            raise ValueError(f"Feature mode {self.feature_mode} invalid (choose from ['include', 'drop', 'fraction')")
        self.include_bigrams = include_bigrams

    def _initialize_model(self) -> None:
        self.model = MultinomialNB(alpha=self.alpha, force_alpha=True)

    def _select_features_include(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Return indices of features that should be used
        if self.feature_param in [None, X.shape[1]]:
            self.feature_indices = np.arange(X.shape[1])
        elif self.feature_param > X.shape[1]:
            raise ValueError(
                f"Total number of possible features {X.shape[1]} lower than " + \
                f"specified number of features to use {self.feature_param}"
            )
        else:
            binary_X = (X > 0)*1

            mi_scores = np.zeros(binary_X.shape[1])
            for feature_idx in range(binary_X.shape[1]):
                mi_scores[feature_idx] = mutual_info_score(binary_X[:,feature_idx], y)

            feature_score = mi_scores

            self.feature_indices = np.argpartition(feature_score, -self.feature_param)[-self.feature_param:]

    def _select_features_drop(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Return indices of features that should be used
        if self.feature_param in [None, 0]:
            self.feature_indices = np.arange(X.shape[1])
        elif self.feature_param > X.shape[1]:
            raise ValueError(
                f"Total number of possible features {X.shape[1]} exceeds " + \
                f"specified number of features to drop {self.feature_param}"
            )
        else:
            binary_X = (X > 0)*1

            mi_scores = np.zeros(binary_X.shape[1])
            for feature_idx in range(binary_X.shape[1]):
                mi_scores[feature_idx] = mutual_info_score(binary_X[:,feature_idx], y)

            feature_score = mi_scores
            n_features = len(feature_score) - self.feature_param

            self.feature_indices = np.argpartition(feature_score, -n_features)[-n_features:]

    def _select_features_fraction(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Return indices of features that should be used
        if self.feature_param in [None, 1]:
            self.feature_indices = np.arange(X.shape[1])
        elif self.feature_param > 1 or self.feature_param <= 0:
            raise ValueError(f"Fraction of features to use not in valid range <0, 1]")
        else:
            binary_X = (X > 0)*1

            mi_scores = np.zeros(binary_X.shape[1])
            for feature_idx in range(binary_X.shape[1]):
                mi_scores[feature_idx] = mutual_info_score(binary_X[:,feature_idx], y)

            feature_score = mi_scores
            n_features = int(len(feature_score) * self.feature_param)

            self.feature_indices = np.argpartition(feature_score, -n_features)[-n_features:]

    def select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Return indices of features that should be used
        if self.feature_mode == 'include':
            self._select_features_include(X, y)
        elif self.feature_mode == 'drop':
            self._select_features_drop(X, y)
        elif self.feature_mode == 'fraction':
            self._select_features_fraction(X, y)
        
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



class SingleClassificationTree(Classifier):
    def __init__(self,
                name: str="SingleClassificationTree",
                ccp_alpha: float=1.0 ) -> None:
        super().__init__(name)
        self.ccp_alpha = ccp_alpha
    
    def _initialize_model(self):
        self.model = DecisionTreeClassifier( 
                                ccp_alpha=self.ccp_alpha,
                                
        )
        
    
    def analyse_feature_importances(self,index_to_word_mapping: Dict[int,str]) -> None:
        #impurity reductie
        importances = self.model.feature_importances_
        scores = []
        print(len(importances)) #1274 features
        print(importances)
        
        for i  in range(len(importances)):
            token = index_to_word_mapping[i]
            score = importances[i]
            scores.append((token,score))
            #print(token,score)
        
        sorted_pairs_desc = sorted(scores, key=lambda x: x[1], reverse=True)
        print(sorted_pairs_desc)

class Logisticregression(Classifier):
    def __init__(self, name: str="LogisticRegression", c:float = 0.1) -> None:
        super().__init__(name)
        self.c = c
        
    def _initialize_model(self):
        self.model = LogisticRegression(C = self.c 
        )

    def analyse_feature_importances(self,index_to_word_mapping: Dict[int,str]) -> None:
        #grote van coëfficient.
        coefs = self.model.coef_[0]
        importance = np.abs(coefs)
        
        # in /word
        scores = [(index_to_word_mapping[i], importance[i]) for i in range(len(importance))]

        #sort 
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # 10 most important features.
        for word, score in sorted_scores[:10]:
            print(f"{word}: {score:.4f}")
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
