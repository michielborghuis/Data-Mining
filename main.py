from data_loading import ReviewLoader
from preprocessing import ReviewProcessor
from models import NaiveBayesClassifier

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    loader = ReviewLoader()
    processor = ReviewProcessor()

    # Load reviews from files
    train_reviews, train_labels = loader.load_train_reviews()

    # Preprocess reviews to convert to count matrix
    train_X = processor.process_train_reviews(train_reviews, include_bigrams=False)

    # Remove features (uni-/bigrams) that occur in very few documents
    train_X = processor.filter_rare_terms(train_X, min_review_freq=.005)

    print(train_X.shape)
    #exit()

    # Specify which model configurations to evaluate
    models = [
        #NaiveBayesClassifier(name="NaiveBayes(alpha=2)", smoothing_alpha=2.0),
        NaiveBayesClassifier(name="NaiveBayes(alpha=1)", smoothing_alpha=1.0),
        #NaiveBayesClassifier(name="NaiveBayes(alpha=.1)", smoothing_alpha=0.1),
        #NaiveBayesClassifier(name="NaiveBayes(alpha=1e-5)", smoothing_alpha=1e-5)
    ]

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*100)
    print(f"MODEL\t\t\t|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*100)
    for model in models:
        cv_performance = model.get_validation_performance(train_X, train_labels, n_features=1250)
        print(
            f"{model.name}\t|\t" + \
            f"{100*cv_performance['accuracy']:.2f}%\t|\t" + \
            f"{cv_performance['precision']:.2f}\t|\t" + \
            f"{cv_performance['recall']:.2f}\t|\t" + \
            f"{cv_performance['f1']:.2f}"
        )
    print("")

    #models[0].analyse_feature_importances(index_to_word_mapping=processor.index_token_dict)

    # NOTE: we should only start looking at test set performance in a couple of weeks or so
    #   -> modelling/hyperparameter choices should NOT be based on test set performance 

    #test_reviews, test_labels = loader.load_test_reviews()
    #test_X = processor.process_test_reviews(test_reviews, include_bigrams=False)

if __name__ == "__main__":
    main()