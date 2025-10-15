from models import GradientBoostingClassifier

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    results_file = "gb_grid_search_results.txt"
    def write_result(line: str):
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    grid = {
        "include_bigrams": [
            #False,
            True
        ],
        "doc_freqs": [
            .025,
            .010,
            .005,
            .0
        ],
        "n_estimators": [
            300
        ],
        "learning_rates": [
            0.1,
            0.05,
            0.01,
            0.005,
            0.001
        ],
        "max_depth": [
            5,
            10,
            20,
            None
        ],
        "subsample": [
            0.5,
            0.75,
            1.0
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
    write_result("")
    write_result("CROSS-VALIDATION PERFORMANCES:")
    write_result("")
    write_result("-"*150)
    write_result(f"{'MODEL':<50}|\tACC\t|\tPREC\t|\tREC\t|\tF1\tBEST MODEL")
    write_result("-"*150)

    for include_bigrams in grid['include_bigrams']:
        for df in grid['doc_freqs']:
            for n_est in grid['n_estimators']:
                for learning_rate in grid['learning_rates']:
                    for max_depth in grid['max_depth']:
                        for subsample in grid['subsample']:
                            for max_feat in grid['max_features']:

                                # Specify which model configuration to evaluate
                                model = GradientBoostingClassifier(
                                    name=f"DF{df}_BG{include_bigrams}_NE{n_est}_LR{learning_rate}_MD{max_depth}_SS{subsample}_MF{max_feat}",
                                    min_df=df,
                                    include_bigrams=include_bigrams,
                                    n_estimators=n_est,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    subsample=subsample,
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
                                write_result(
                                    f"{model.name:<50}|\t" + \
                                    f"{100*np.mean(cv_performance['accuracy']):.2f}%\t|\t" + \
                                    f"{np.mean(cv_performance['precision']):.2f}\t|\t" + \
                                    f"{np.mean(cv_performance['recall']):.2f}\t|\t" + \
                                    f"{np.mean(cv_performance['f1']):.2f}\t|\t" + \
                                    f"{best_model:<50}"
                                )

    print("")
    write_result("")

if __name__ == "__main__":
    main()
