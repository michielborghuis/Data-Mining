from models import NaiveBayesClassifier,RandomForestClassifier

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    # Specify which model configurations to evaluate
    models = [
        #NaiveBayesClassifier(name="NaiveBayes(1952)", smoothing_alpha=1.0, n_features=train_X.shape[1]) # Uses all features
        #NaiveBayesClassifier(name="NaiveBayes(1933)", smoothing_alpha=1.0, n_features=1933), # Best unigram configuration (df_min = .005)
        #NaiveBayesClassifier(name="NaiveBayes(bigram, 3436)", smoothing_alpha=0.5, n_features=3436), # Best unigram+bigram configuration (df_min = .010)

        RandomForestClassifier(
            min_df=.01,
            include_bigrams=False,
            n_estimators=500,
            name="RF Test"
        ),]

    """    # Found using backward search
        NaiveBayesClassifier(
            smoothing_alpha=1.0,
            min_df=.005,
            features=565,
            feature_mode="drop",
            include_bigrams=False,
            name="UNIGRAM W/O 565 FEATS"
        ),
        
        # Found using forward search
        NaiveBayesClassifier(
            smoothing_alpha=1.0,
            min_df=.005,
            features=101,
            feature_mode="include",
            include_bigrams=False,
            name="UNIGRAM W/ 101 FEATS"
        ),
        
        # Found using grid search
        NaiveBayesClassifier(
            smoothing_alpha=1.0,
            min_df=.01,
            features=.4,
            feature_mode="fraction",
            include_bigrams=False,
            name="UNIGRAM W/ .4 FEATS"
        ),
        
        NaiveBayesClassifier(
            smoothing_alpha=1.0,
            min_df=.005,
            features=None,
            include_bigrams=False,
            name="UNIGRAM W/ ALL FEATS"
        ),
        
        # Found using backward search
        NaiveBayesClassifier(
            smoothing_alpha=0.5,
            min_df=.010,
            features=102,
            feature_mode="drop",
            include_bigrams=True,
            name="BIGRAM W/O 102 FEATS"
        ),
        
        # Found using forward search
        NaiveBayesClassifier(
            smoothing_alpha=0.5,
            min_df=.010,
            features=147,
            feature_mode="include",
            include_bigrams=True,
            name="BIGRAM W/ 147 FEATS"
        ),
        
        # Found using grid search
        NaiveBayesClassifier(
            smoothing_alpha=0.5,
            min_df=.01,
            features=.8,
            feature_mode="fraction",
            include_bigrams=True,
            name="BIGRAM W/ .8 FEATS"
        ),
        
        NaiveBayesClassifier(
            smoothing_alpha=0.5,
            min_df=.010,
            features=None,
            include_bigrams=True,
            name="BIGRAM W/ ALL FEATS"
        ),
    ]"""

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*85)
    print(f"{'MODEL':<25}|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*85)
    for model in models:
        val_performance = model.get_validation_performance(n_folds=10, n_repeats=1)

        print(
            f"{model.name:<25}|\t" + \
            f"{100*np.mean(val_performance['accuracy']):.2f}%\t|\t" + \
            f"{np.mean(val_performance['precision']):.2f}\t|\t" + \
            f"{np.mean(val_performance['recall']):.2f}\t|\t" + \
            f"{np.mean(val_performance['f1']):.2f}"
        )

    print("")

    exit()

    # NOTE: we should only start looking at test set performance in a couple of weeks or so
    #   -> modelling/hyperparameter choices should NOT be based on test set performance 

    print("")
    print("TEST SET PERFORMANCES:")
    print("")
    print("-"*85)
    print(f"{'MODEL':<25}|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*85)
    for model in models:
        test_set_performance = model.get_test_performance()
        print(
            f"{model.name:<25}|\t" + \
            f"{100*test_set_performance['accuracy']:.2f}%\t|\t" + \
            f"{test_set_performance['precision']:.3f}\t|\t" + \
            f"{test_set_performance['recall']:.3f}\t|\t" + \
            f"{test_set_performance['f1']:.3f}"
        )
    
    #models[-1].analyse_feature_importances()

if __name__ == "__main__":
    main()