from models import NaiveBayesClassifier,LRClassifier,ClassificationTree,RandomForestClassifier,GradientBoostingClassifier


import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    # Specify which model configurations to evaluate
    models = [
        # Found using backward search
        #NaiveBayesClassifier(
        #    smoothing_alpha=1.0,
        #    min_df=.005,
        #    features=565,
        #    feature_mode="drop",
        #    include_bigrams=False,
        #    name="UNIGRAM W/O 565 FEATS"
        #),

        # Found using backward search
        #NaiveBayesClassifier(
        #    smoothing_alpha=0.5,
        #    min_df=.010,
        #    features=102,
        #    feature_mode="drop",
        #    include_bigrams=True,
        #    name="BIGRAM W/O 102 FEATS"
        #),

        ClassificationTree(min_df=.005, ccp_alpha=.01, include_bigrams=False, name="CT UNIGRAMS (mine)"),
        ClassificationTree(min_df=0, ccp_alpha=.01, include_bigrams=False, min_samples_leaf=1, min_samples_split=2, max_depth=10, name="CT UNIGRAMS (Stijn)"),
        ClassificationTree(min_df=0, ccp_alpha=.01, include_bigrams=True, name="CT BIGRAMS (mine)"),
        ClassificationTree(min_df=0, ccp_alpha=.0, include_bigrams=True, min_samples_leaf=4, min_samples_split=2, max_depth=10, name="CT BIGRAMS (Stijn)"),

        LRClassifier(min_df=0, c=1000, include_bigrams=False, name="LR UNIGRAMS"),
        LRClassifier(min_df=0, c=1000, include_bigrams=True, name="LR BIGRAMS"),

        LRClassifier(min_df=0, c=100, include_bigrams=False, name="LR UNIGRAMS (stijn)"),
        LRClassifier(min_df=0, c=100, include_bigrams=True, name="LR BIGRAMS (stijn)"),

        #RandomForestClassifier(
        #    min_df=0, include_bigrams=False, n_estimators=300, # 200 trees originally
        #    max_features='sqrt', min_samples_leaf=2, name="RF UNIGRAMS (mine)"
        #),

        #RandomForestClassifier(
        #    min_df=0, include_bigrams=False, n_estimators=300, criterion='entropy', 
        #    max_features='sqrt', min_samples_leaf=2, name="RF UNIGRAMS (mine + entropy)"
        #),

        #RandomForestClassifier(
        #    min_df=.005, include_bigrams=False, criterion='entropy', 
        #    n_estimators=300, min_samples_split=2, min_samples_leaf=2, 
        #    max_features='log2', name="RF UNIGRAMS (michiel)"
        #),

        #RandomForestClassifier(
        #    min_df=.02, include_bigrams=True, n_estimators=1000, # 1000 trees originally
        #    max_features='log2', min_samples_leaf=5, name="RF BIGRAMS (mine)"
        #),

        #RandomForestClassifier(
        #    min_df=.02, include_bigrams=True, n_estimators=1000, criterion='entropy', 
        #    max_features='log2', min_samples_leaf=5, name="RF BIGRAMS (mine + entropy)"
        #),

        #RandomForestClassifier(
        #    min_df=.01, include_bigrams=True, criterion='entropy', 
        #    n_estimators=300, min_samples_split=2, min_samples_leaf=2, 
        #    max_features='log2', name="RF BIGRAMS (michiel)"
        #),

        #GradientBoostingClassifier()
    ]

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*85)
    print(f"{'MODEL':<30}|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*85)
    for model in models:
        val_performance = model.get_validation_performance(n_folds=10, n_repeats=3)

        print(
            f"{model.name:<30}|\t" + \
            f"{100*np.mean(val_performance['accuracy']):.2f}%\t|\t" + \
            f"{np.mean(val_performance['precision']):.2f}\t|\t" + \
            f"{np.mean(val_performance['recall']):.2f}\t|\t" + \
            f"{np.mean(val_performance['f1']):.2f}"
        )

    print("")

    #exit()

    # NOTE: we should only start looking at test set performance in a couple of weeks or so
    #   -> modelling/hyperparameter choices should NOT be based on test set performance 

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
    
    #models[-1].analyse_feature_importances()

if __name__ == "__main__":
    main()