from models import LRClassifier

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    grid = {
        "doc_freqs": [
            #.050,
            #.020,
            #.010,
            #.005, # >= 4 occurences
            0
        ],
        "C": [
            #1,
            #10,
            #100,
            500,
            750,
            1_000,
            2_000,
            5_000,
            #10_000,
            #100_000,
        ]
    }

    print("BIGRAMS")

    best_acc = 0
    best_model = " "*18

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*90)
    print(f"{'MODEL':<30}|\tACC\t|\t\t{'BEST':<30}|\tACC")
    print("-"*90)
    for df in grid['doc_freqs']:
        for c in grid['C']:
            # Specify which model configuration to evaluate
            model = LRClassifier(
                min_df=df,
                c=c,
                include_bigrams=True,
                name=f"LR_bi_df_{df}_c_{c}"
            )

            val_performance = model.get_validation_performance(n_folds=10, n_repeats=10)

            if np.mean(val_performance['accuracy']) > best_acc:
                best_acc = np.mean(val_performance['accuracy'])
                best_model = model.name

            print(
                f"{model.name:<30}|\t" + \
                f"{100*np.mean(val_performance['accuracy']):.2f}%\t|\t" + \
                f"{best_model:<30}|\t" + \
                f"{100*best_acc:.2f}%"
            )

if __name__ == "__main__":
    main()