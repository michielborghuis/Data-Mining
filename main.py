import numpy as np
import random

from models.models import (
    NaiveBayesClassifier,
    LRClassifier,
    ClassificationTree,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from utils.util import perform_mcnemar_tests

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    # -------------------------------------------
    #       1. SPECIFY MODEL CONFIGURATIONS
    # -------------------------------------------

    models = [
        # ---------------------------------------
        #       FINAL UNIGRAM CLASSIFIERS
        # ---------------------------------------

        NaiveBayesClassifier(
            smoothing_alpha=1.0,
            min_df=.005,
            features=565,
            feature_mode="drop",
            include_bigrams=False,
            name="FINAL NB UNIGRAM"
        ),
        LRClassifier(
            min_df=0, 
            c=1263.46, 
            include_bigrams=False, 
            name="FINAL LR UNIGRAM"
        ),
        ClassificationTree(
            min_df=.0,
            ccp_alpha=.01043,
            criterion='gini',
            include_bigrams=False,
            name="FINAL CT UNIGRAM"
        ),
        RandomForestClassifier(
            include_bigrams=False,
            min_df=.01,
            criterion='entropy',
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            name="FINAL RF UNIGRAM"
        ),
        GradientBoostingClassifier(
            include_bigrams=False,
            min_df=.01,
            n_estimators=300,
            learning_rate=.01,
            max_depth=20,
            subsample=.75,
            max_features='log2',
            name="FINAL GB UNIGRAM"
        ),


        # ---------------------------------------
        #       FINAL BIGRAM CLASSIFIERS
        # ---------------------------------------

        NaiveBayesClassifier(
            smoothing_alpha=0.5,
            min_df=.010,
            features=102,
            feature_mode="drop",
            include_bigrams=True,
            name="FINAL NB BIGRAM"
        ),
        LRClassifier(
            min_df=0, 
            c=3141.02, 
            include_bigrams=True, 
            name="FINAL LR BIGRAM"
        ),
        ClassificationTree(
            min_df=.01,
            ccp_alpha=.00840,
            criterion='gini',
            include_bigrams=True,
            name="FINAL CT BIGRAM"
        ),
        RandomForestClassifier(
            include_bigrams=True,
            min_df=.005,
            criterion='entropy',
            n_estimators=400,
            max_depth=None,
            min_samples_split=8,
            min_samples_leaf=1,
            max_features='log2',
            name="FINAL RF BIGRAM"
        ),
        GradientBoostingClassifier(
            include_bigrams=True,
            min_df=.01,
            n_estimators=300,
            learning_rate=.1,
            max_depth=10,
            subsample=.5,
            max_features='log2',
            name="FINAL GB BIGRAM"
        )
    ]

    # -------------------------------------------
    #       2. GET MODEL PERFORMANCES
    # -------------------------------------------

    print("")
    print("TEST SET PERFORMANCES:")
    print("")
    print("-"*85)
    print(f"{'MODEL':<30}|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*85)
    for model in models:
        test_set_performance = model.get_test_performance()

        print(
            f"{model.name:<30}|\t" + \
            f"{100*test_set_performance['accuracy']:.2f}%\t|\t" + \
            f"{test_set_performance['precision']:.3f}\t|\t" + \
            f"{test_set_performance['recall']:.3f}\t|\t" + \
            f"{test_set_performance['f1']:.3f}"
        )
    print("-"*85)
    print('\n'+'#'*85+'\n')

    # -------------------------------------------
    #       3. PERFORM STATISTICAL TEST
    # -------------------------------------------

    print("")
    print("MCNEMAR TEST P-VALUES:")
    print("")
    perform_mcnemar_tests(models)
    print('\n'+'#'*85+'\n')

    # -------------------------------------------
    #       4. PERFORM FEATURE ANALYSIS
    # -------------------------------------------

    print("")
    print("FEATURE ANALYSES:")
    print("")

    all_feature_tokens = set()
    for model in models:
        if model.include_bigrams == True:
            model.analyse_feature_importances(print_top_features=True)
            all_feature_tokens = all_feature_tokens.union(set(model.processor.token_index_dict.keys()))

    print('\n'+'#'*85+'\n')

    all_feature_tokens = list(all_feature_tokens)
    true_feature_ranks = np.zeros((len(models), len(all_feature_tokens)))
    fake_feature_ranks = np.zeros((len(models), len(all_feature_tokens)))
    for i, model in enumerate(models):
        for j, token in enumerate(all_feature_tokens):
            true_rank, fake_rank = model.get_feature_importance_ranks(token)
            true_feature_ranks[i,j] = true_rank
            fake_feature_ranks[i,j] = fake_rank

    print("")
    average_true_ranks = np.mean(true_feature_ranks, axis=0)
    indices = np.argsort(average_true_ranks)
    print('-'*75)
    print("FEATURES OVERALL MOST INDICATIVE OF TRUTHFUL REVIEWS")
    print('-'*75)
    for rank in range(10):
        print(f"{rank+1}. {all_feature_tokens[indices[rank]]:<29}"+(" "*(rank!=9))+f"({average_true_ranks[indices[rank]]:.1f} AVG RANK)")
    
    print("")
    average_fake_ranks = np.mean(fake_feature_ranks, axis=0)
    indices = np.argsort(average_fake_ranks)
    print('-'*75)
    print("FEATURES OVERALL MOST INDICATIVE OF DECEPTIVE REVIEWS")
    print('-'*75)
    for rank in range(10):
        print(f"{rank+1}. {all_feature_tokens[indices[rank]]:<29}"+(" "*(rank!=9))+f"({average_fake_ranks[indices[rank]]:.1f} AVG RANK)")

if __name__ == "__main__":
    main()