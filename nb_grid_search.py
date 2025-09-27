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

    grid = {
        "doc_freqs": [
            .010,
            .005, # >= 4 occurences
            .004  # >= 3 occurences
        ],
        "alpha": [
            5.0,
            2.5,
            1.0,
            0.5,
            0.1
        ],
        "n_features": [
            250,
            500,
            750,
            1000,
            1250,
            1500,
            1750,
            2000
        ]
    }

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*100)
    print(f"MODEL\t\t\t|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*100)

    for df in grid['doc_freqs']:
        for alpha in grid['alpha']:
            for n_feats in grid['n_features']:
                # Preprocess reviews to convert to count matrix
                train_X = processor.process_train_reviews(train_reviews, include_bigrams=False)

                # Remove features (uni-/bigrams) that occur in very few documents
                train_X = processor.filter_rare_terms(train_X, min_review_freq=df)

                # Specify which model configuration to evaluate
                model = NaiveBayesClassifier(name=f"DF{df}_A{alpha}_#F{n_feats}", smoothing_alpha=alpha)

                if n_feats > train_X.shape[1]:
                    continue

                cv_performance = model.get_validation_performance(train_X, train_labels, n_features=n_feats)
                print(
                    f"{model.name}\t|\t" + \
                    f"{100*cv_performance['accuracy']:.2f}%\t|\t" + \
                    f"{cv_performance['precision']:.2f}\t|\t" + \
                    f"{cv_performance['recall']:.2f}\t|\t" + \
                    f"{cv_performance['f1']:.2f}"
                )

    print("")

if __name__ == "__main__":
    main()