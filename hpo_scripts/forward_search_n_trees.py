import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from models.models import GradientBoostingClassifier, RandomForestClassifier

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    prev_acc = -1
    n_trees = 300
    step_size = 50

    print("UNIGRAMS")

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*60)
    print(f"{'MODEL':<40}|\tACC")
    print("-"*60)
    while True:
        # Specify which model configuration to evaluate
        model = GradientBoostingClassifier(
            include_bigrams=True,
            min_df=.01,
            n_estimators=300,
            learning_rate=.1,
            max_depth=10,
            subsample=.5,
            max_features='log2',
            name=f"{n_trees} TREES"
        )

        val_performance = model.get_validation_performance(n_folds=10, n_repeats=1)
        accuracy = np.mean(val_performance['accuracy'])

        print(
            f"{model.name:<40}|\t" + \
            f"{100*np.mean(val_performance['accuracy']):.2f}%"
        )

        if accuracy < prev_acc:
            break

        prev_acc = accuracy
        n_trees += step_size

if __name__ == "__main__":
    main()