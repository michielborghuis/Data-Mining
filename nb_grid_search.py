from models import NaiveBayesClassifier

import numpy as np
import random
import json

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    grid = {
        "doc_freqs": [
            .020,
            .010,
            .005, # >= 4 occurences
            .004  # >= 3 occurences
        ],
        "alpha": [
            5.0,
            2.0,
            1.0,
            0.5,
            0.1
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
            # Specify which model configuration to evaluate
            model = NaiveBayesClassifier(
                smoothing_alpha=alpha,
                min_df=df,
                drop_features=0,
                include_bigrams=False,
                name=f"df_{df}_alpha_{alpha}"
            )

            val_performance = model.get_validation_performance(n_folds=10, n_repeats=2)

            print(
                f"{model.name}\t|\t" + \
                f"{100*np.mean(val_performance['accuracy']):.2f}%\t|\t" + \
                f"{np.mean(val_performance['precision']):.2f}\t|\t" + \
                f"{np.mean(val_performance['recall']):.2f}\t|\t" + \
                f"{np.mean(val_performance['f1']):.2f}"
            )

if __name__ == "__main__":
    main()