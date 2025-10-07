from models import NaiveBayesClassifier

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*30+" "*8+"-"*30)
    print(f"MODEL\t\t|\tACC\t\tMAX ACC\t\t|\tSE")
    print("-"*30+" "*8+"-"*30)

    drop_features = 0
    best_acc = 0
    standard_error = 0
    while True:
        model = NaiveBayesClassifier(
            smoothing_alpha=1.0,
            min_df=.005,
            drop_features=drop_features,
            include_bigrams=False,
            name=f"DROP {drop_features} FEATS"
        )
        
        val_accs = model.get_validation_performance(n_folds=10, n_repeats=1)['accuracy']
        cur_acc = np.mean(val_accs)

        # STANDARD ERROR VARIANT
        if cur_acc > best_acc:
            best_acc = cur_acc
            standard_error = np.std(val_accs)/np.sqrt(len(val_accs))

        print(
            f"{model.name}\t|\t" + \
            f"{100*cur_acc:.2f}%\t\t" + \
            f"{100*best_acc:.2f}%\t\t|\t" + \
            f"{100*standard_error:.2f}%"
        )

        if best_acc - cur_acc > standard_error:
            print("Exiting!")
            break

        drop_features += 1

if __name__ == "__main__":
    main()