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
        NaiveBayesClassifier(
            smoothing_alpha=1.0,
            min_df=.005,
            drop_features=19,
            include_bigrams=False,
            name="NB ALL FEATURES"
        ),
        NaiveBayesClassifier(
            smoothing_alpha=1.0,
            min_df=.005,
            drop_features=19,
            include_bigrams=False,
            name="NB 1933 FEATURES"
        ),
        NaiveBayesClassifier(
            smoothing_alpha=1.0,
            min_df=.005,
            drop_features=122,
            include_bigrams=False,
            name="NB DROP 122 FEATS"
        )
    ]

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*100)
    print(f"MODEL\t\t\t|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*100)
    for model in models:
        val_performance = model.get_validation_performance(n_folds=10, n_repeats=1)

        print(
            f"{model.name}\t|\t" + \
            f"{100*np.mean(val_performance['accuracy']):.2f}%\t|\t" + \
            f"{np.mean(val_performance['precision']):.2f}\t|\t" + \
            f"{np.mean(val_performance['recall']):.2f}\t|\t" + \
            f"{np.mean(val_performance['f1']):.2f}"
        )
    print("")


    # NOTE: we should only start looking at test set performance in a couple of weeks or so
    #   -> modelling/hyperparameter choices should NOT be based on test set performance 

    print("")
    print("TEST SET PERFORMANCES:")
    print("")
    print("-"*100)
    print(f"MODEL\t\t\t|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*100)
    for model in models:
        test_set_performance = model.get_test_performance()
        print(
            f"{model.name}\t|\t" + \
            f"{100*test_set_performance['accuracy']:.2f}%\t|\t" + \
            f"{test_set_performance['precision']:.3f}\t|\t" + \
            f"{test_set_performance['recall']:.3f}\t|\t" + \
            f"{test_set_performance['f1']:.3f}"
        )
    
    models[0].analyse_feature_importances()

if __name__ == "__main__":
    main()