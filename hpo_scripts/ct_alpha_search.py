import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from models.models import ClassificationTree

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    include_bigrams = True

    criteria = [
        'entropy',
        'gini'
    ]

    doc_freqs = [
        .100,
        .050,
        .025,
        .010,
        .005,
        .0
    ]

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*135)
    print(f"{'MODEL':<35}|\tMEAN ALPHA (±STD)\t|\tMIN ALPHA\t|\tMAX ALPHA\t|\tACC(±STD)")
    print("-"*135)
    for criterion in criteria:
        for df in doc_freqs:
            # Specify which model configuration to evaluate
            model = ClassificationTree(
                ccp_alpha=0.0,
                min_df=df,
                criterion=criterion,
                include_bigrams=include_bigrams,
                name = f"CT_bi-{include_bigrams}_df-{df}_crit-{criterion}"
            )

            alphas, accs = model.alpha_cross_validation(n_folds=10, n_repeats=1)

            print(
                f"{model.name:<35}|\t" + \
                f"{np.mean(alphas):.5f} (±{np.std(alphas):.5f})\t|\t" + \
                f"{np.min(alphas):.5f}\t\t|\t" + \
                f"{np.max(alphas):.5f}\t\t|\t" + \
                f"{100*np.mean(accs):.2f}% (±{100*np.std(accs):.2f}%)"
            )

if __name__ == "__main__":
    main()