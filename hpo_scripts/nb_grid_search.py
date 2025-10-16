import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from models.models import NaiveBayesClassifier

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    include_bigrams = True

    grid = {
        "doc_freqs": [
            .025,
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
        ]
    }

    best_acc = 0
    best_model = " "*18

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*105)
    print(f"{'MODEL':<35}|\tACC\t|\t\t{'BEST':<27}|\tACC")
    print("-"*105)
    for df in grid['doc_freqs']:
        for alpha in grid['alpha']:
            # Specify which model configuration to evaluate
            model = NaiveBayesClassifier(
                smoothing_alpha=alpha,
                min_df=df,
                include_bigrams=include_bigrams,
                name=f"NB_bi-{include_bigrams}_df-{df}_a-{alpha}"
            )

            val_performance = model.get_validation_performance(n_folds=10, n_repeats=1)

            if np.mean(val_performance['accuracy']) > best_acc:
                best_acc = np.mean(val_performance['accuracy'])
                best_model = model.name

            print(
                f"{model.name:<35}|\t" + \
                f"{100*np.mean(val_performance['accuracy']):.2f}%\t|\t" + \
                f"{best_model:<35}|\t" + \
                f"{100*best_acc:.2f}%"
            )

if __name__ == "__main__":
    main()