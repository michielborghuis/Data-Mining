import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from models.models import ClassificationTree

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    # UNIGRAM GRID
    grid = {
         # {df_min}: {range_of_alphas} # ALPHAS FOUND USING ct_alpha_search.py SCRIPT
        "doc_freqs": {
            .100: {
                "gini": {"a_min": .00273, "a_max": .01315}, 
                "entropy": {"a_min": .00347, "a_max": .02137}, 
            },
            .050: {
                "gini": {"a_min": .00000, "a_max": .00961}, 
                "entropy": {"a_min": .00478, "a_max": .01942}, 
            },
            .025: {
                "gini": {"a_min": .00172, "a_max": .01555}, 
                "entropy": {"a_min": .00521, "a_max": .03274}, 
            },
            .010: {
                "gini": {"a_min": .00231, "a_max": .01888}, 
                "entropy": {"a_min": .00597, "a_max": .03183}, 
            },
            .005: {
                "gini": {"a_min": .00173, "a_max": .01310}, 
                "entropy": {"a_min": .00602, "a_max": .03588}, 
            },
            .0: {
                "gini": {"a_min": .00260, "a_max": .01241}, 
                "entropy": {"a_min": .00478, "a_max": .02172}, 
            }
        }
    }

    # BIGRAM GRID
    grid = {
         # {df_min}: {range_of_alphas} # ALPHAS FOUND USING ct_alpha_search.py SCRIPT
        "doc_freqs": {
            .100: {
                "gini": {"a_min": .00166, "a_max": .01315}, 
                "entropy": {"a_min": .00620, "a_max": .02137}, 
            },
            .050: {
                "gini": {"a_min": .00231, "a_max": .01076}, 
                "entropy": {"a_min": .00000, "a_max": .02959}, 
            },
            .025: {
                "gini": {"a_min": .00000, "a_max": .01174}, 
                "entropy": {"a_min": .00478, "a_max": .02967}, 
            },
            .010: {
                "gini": {"a_min": .00278, "a_max": .02034}, 
                "entropy": {"a_min": .00478, "a_max": .03191}, 
            },
            .005: {
                "gini": {"a_min": .00000, "a_max": .01917}, 
                "entropy": {"a_min": .00000, "a_max": .02178}, 
            },
            .0: {
                "gini": {"a_min": .00000, "a_max": .01931}, 
                "entropy": {"a_min": .00563, "a_max": .03410}, 
            }
        }
    }

    print("CT BIGRAM")

    best_acc = 0
    best_model = " "*18

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*110)
    print(f"{'MODEL':<35}|\tACC\t|\t\tBEST\t\t\t|\tACC")
    print("-"*110)
    for df in grid['doc_freqs'].keys():
        for criterion in grid['doc_freqs'][df].keys():
            bounds = grid['doc_freqs'][df][criterion]

            if bounds['a_min'] == 0:
                alpha_space = np.geomspace(1e-4, bounds['a_max'], num=10)
                alpha_space[0] = 0
            else:
                alpha_space = np.geomspace(bounds['a_min'], bounds['a_max'], num=10)

            for alpha in alpha_space:
                # Specify which model configuration to evaluate
                model = ClassificationTree(
                    min_df=df,
                    include_bigrams=True,
                    ccp_alpha=alpha,
                    criterion=criterion,
                    name=f"CT_bi_df_{df}_a_{alpha:.5f}_{criterion}"
                )

                val_performance = model.get_validation_performance(n_folds=10, n_repeats=1)

                if np.mean(val_performance['accuracy']) > best_acc:
                    best_acc = np.mean(val_performance['accuracy'])
                    best_model = model.name

                print(
                    f"{model.name:<35}|\t" + \
                    f"{100*np.mean(val_performance['accuracy']):.2f}%\t|\t" + \
                    f"{best_model}\t|\t" + \
                    f"{100*best_acc:.2f}%"
                )

if __name__ == "__main__":
    main()