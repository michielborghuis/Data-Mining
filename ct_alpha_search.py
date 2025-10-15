from models import ClassificationTree

import numpy as np
import random
import json

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    doc_freqs = [
        .100,
        .050,
        .025,
        .010,
        .005, # >= 4 occurences
        .0
    ]

    print("CT BIGRAMS (entropy)")

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*110)
    print(f"DF MIN\t|\tMEAN ALPHA (±STD)\t|\tMIN ALPHA\t|\tMAX ALPHA\t|\tACC(±STD)")
    print("-"*110)
    for df in doc_freqs:
        # Specify which model configuration to evaluate
        model = ClassificationTree(
            ccp_alpha=0.0,
            min_df=df,
            criterion="entropy",
            include_bigrams=True
        )

        alphas, accs = model.alpha_cross_validation(n_folds=10, n_repeats=1)

        print(
            f"{df}\t|\t" + \
            f"{np.mean(alphas):.5f} (±{np.std(alphas):.5f})\t|\t" + \
            f"{np.min(alphas):.5f}\t\t|\t" + \
            f"{np.max(alphas):.5f}\t\t|\t" + \
            f"{100*np.mean(accs):.2f}% (±{100*np.std(accs):.2f}%)"
        )

if __name__ == "__main__":
    main()