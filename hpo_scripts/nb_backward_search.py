import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from models.models import NaiveBayesClassifier

import matplotlib.pyplot as plt
import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    print("UNIGRAM")
    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*30+" "*8+"-"*30)
    print(f"MODEL\t\t|\tACC\t\tMAX ACC\t\t|\tSE")
    print("-"*30+" "*8+"-"*30)

    drop_features = 0
    step_size = 1
    all_accs = []
    best_acc = 0
    standard_error = 0
    while True:
        try:
            model = NaiveBayesClassifier(
                smoothing_alpha=1.0,
                min_df=.005,
                feature_mode='drop',
                features=drop_features,
                include_bigrams=False,
                name=f"DROP {drop_features} FEATS"
            )

            #model = NaiveBayesClassifier(
            #    smoothing_alpha=0.5,
            #    min_df=.01,
            #    feature_mode='drop',
            #    features=drop_features,
            #    include_bigrams=True,
            #    name=f"DROP {drop_features} FEATS"
            #)
            
            val_accs = model.get_validation_performance(n_folds=10, n_repeats=1)['accuracy']
            cur_acc = np.mean(val_accs)
            all_accs.append(cur_acc)

            # STANDARD ERROR VARIANT
            if cur_acc >= best_acc:
                best_acc = cur_acc
                standard_error = np.std(val_accs)/np.sqrt(len(val_accs))

            print(
                f"{model.name}\t|\t" + \
                f"{100*cur_acc:.2f}%\t\t" + \
                f"{100*best_acc:.2f}%\t\t|\t" + \
                f"{100*standard_error:.2f}%"
            )

            if best_acc - cur_acc > 2*standard_error:
                print("Exiting!")
                break

            drop_features += step_size
        except:
            print(f"Exiting! (presumably, drop_features={drop_features} exceeds total number of features)")
            break
    
    plt.plot(np.arange(0, drop_features+step_size, step_size)[:len(all_accs)], all_accs, marker='s')
    plt.xlabel("# Of Dropped Features")
    plt.ylabel("Cross-Validation Accuracy")
    plt.grid(True)
    plt.title("k-Backward Search Results")
    plt.show()

if __name__ == "__main__":
    main()