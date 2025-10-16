from models.models import RandomForestClassifier

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    print("UNIGRAMS")

    grid = {
        "include_bigrams": [
            False,
            #True
        ],
        "doc_freqs": [
            .025,
            .010,
            .005,
            .0
        ],
        "criterion": [
            "gini",
            "entropy"
        ],
        "n_estimators": [
            300
        ],
        "max_depth": [
            10,
            20,
            None
        ],
        "min_samples_splits": [
            2,
            4,
            8
        ],
        "min_samples_leaf": [
            1,
            2,
            4,
        ],
        "max_features": [
            "sqrt",
            "log2",
        ],
    }

    best_acc = 0
    best_model = ""

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*150)
    print(f"{'MODEL':<50}|\tACC\t|\tPREC\t|\tREC\t|\tF1\t|\t\tBEST MODEL")
    print("-"*150)
    for include_bigrams in grid['include_bigrams']:
        for df in grid['doc_freqs']:
            for crit in grid['criterion']:
                for n_est in grid['n_estimators']:
                    for max_depth in grid['max_depth']:
                        for min_split in grid['min_samples_splits']:
                            for min_leaf in grid['min_samples_leaf']:
                                if min_split < 2*min_leaf: # Redundant so leads to bias in search
                                    continue
                                for max_feat in grid['max_features']:
                                
                                    # Specify which model configuration to evaluate
                                    model = RandomForestClassifier(
                                        name=f"DF{df}_BG{include_bigrams}_CR{crit}_MD{max_depth}_MSS{min_split}_MSL{min_leaf}_MF{max_feat}",
                                        min_df=df,
                                        include_bigrams=include_bigrams,
                                        criterion=crit,
                                        n_estimators=n_est,
                                        max_depth=max_depth,
                                        min_samples_split=min_split,
                                        min_samples_leaf=min_leaf,
                                        max_features=max_feat
                                    )
    
                                    cv_performance = model.get_validation_performance(n_folds=10, n_repeats=1)

                                    if np.mean(cv_performance['accuracy']) > best_acc:
                                        best_model = model.name
                                        best_acc = np.mean(cv_performance['accuracy'])
                                    
                                    print(
                                        f"{model.name:<50}|\t" + \
                                        f"{100*np.mean(cv_performance['accuracy']):.2f}%\t|\t" + \
                                        f"{np.mean(cv_performance['precision']):.2f}\t|\t" + \
                                        f"{np.mean(cv_performance['recall']):.2f}\t|\t" + \
                                        f"{np.mean(cv_performance['f1']):.2f}\t|\t" + \
                                        f"{best_model:<50}"
                                    )

    print("")

if __name__ == "__main__":
    main()
