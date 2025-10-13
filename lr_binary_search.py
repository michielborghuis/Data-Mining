from models import LRClassifier

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    c_min = .1
    c_max = 100_000

    print("BIGRAMS")

    best_acc = 0
    performance_dict = {}

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*90)
    print(f"{'MODEL':<30}|\tACC\t|\t\t{'BEST':<22}|\tACC")
    print("-"*90)

    while True:
        c_mid = np.sqrt(c_min * c_max)  # geometric mean for log-scale search

        candidates = [c_min, c_mid, c_max]
        scores = []

        for c in candidates:
            model = LRClassifier(
                min_df=0,
                c=c,
                include_bigrams=True,
                name=f"LR BIGRAM (c={c:.3f})"
            )

            if model.name in performance_dict.keys():
                val_performance = performance_dict[model.name]
            else:
                val_performance = model.get_validation_performance(n_folds=10, n_repeats=10)
                performance_dict[model.name] = val_performance

            if np.mean(val_performance['accuracy']) > best_acc:
                best_acc = np.mean(val_performance['accuracy'])
                best_model = model.name

            print(
                f"{model.name:<30}|\t" + \
                f"{100*np.mean(val_performance['accuracy']):.2f}%\t|\t" + \
                f"{best_model:<30}|\t" + \
                f"{100*best_acc:.2f}%"
            )

            scores.append(np.mean(val_performance['accuracy']))

        # Find the best among the three
        best_idx = int(np.argmax(scores))
        best_c = candidates[best_idx]
        best_score = scores[best_idx]

        # Narrow search interval depending on where the best C lies
        if best_idx == 0:  # best at lower end
            c_max = c_mid
        elif best_idx == 2:  # best at upper end
            c_min = c_mid
        else:  # best in the middle
            c_min = np.sqrt(c_min * c_mid)
            c_max = np.sqrt(c_mid * c_max)

        if (c_max / c_min) < 1.01:
            break

if __name__ == "__main__":
    main()