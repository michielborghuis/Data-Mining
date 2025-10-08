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
        ],
        "feature_fractions": [
            .1, 
            .2,
            .3,
            .4,
            .5,
            .6,
            .7,
            .8,
            .9,
            1
            
        ]
    }

    print("BIGRAMS")

    best_acc = 0
    best_model = " "*18

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*90)
    print(f"MODEL\t\t\t|\tACC\t|\t\tBEST\t\t|\tACC")
    print("-"*90)
    for df in grid['doc_freqs']:
        for alpha in grid['alpha']:
            for frac in grid["feature_fractions"]:
                # Specify which model configuration to evaluate
                model = NaiveBayesClassifier(
                    smoothing_alpha=alpha,
                    min_df=df,
                    features=frac,
                    feature_mode="fraction",
                    include_bigrams=True,
                    name=f"df_{df}_a_{alpha}_frac_{frac}"
                )

                val_performance = model.get_validation_performance(n_folds=10, n_repeats=1)

                if np.mean(val_performance['accuracy']) > best_acc:
                    best_acc = np.mean(val_performance['accuracy'])
                    best_model = model.name

                print(
                    f"{model.name}\t|\t" + \
                    f"{100*np.mean(val_performance['accuracy']):.2f}%\t|\t" + \
                    f"{best_model}\t|\t" + \
                    f"{100*best_acc:.2f}%"
                )

if __name__ == "__main__":
    main()