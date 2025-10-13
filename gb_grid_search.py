from data_loading import ReviewLoader
from preprocessing import ReviewProcessor
from models import GradientBoostingClassifier

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    loader = ReviewLoader()
    processor = ReviewProcessor()

    # Load reviews from files
    train_reviews, train_labels = loader.load_train_reviews()

    results_file = "gb_grid_search_results.txt"
    def write_result(line: str):
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# GradientBoostingClassifier(name="GradientBoosting(basic)", n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, max_features=None),
# GradientBoostingClassifier(name="GradientBoosting(n=300)", n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, max_features="sqrt"),

    grid = {
        "include_bigrams": [
            False,
            True
        ],
        "doc_freqs": [
            .010,
            .005,
            .001
        ],
        "n_estimators": [
            100,
            200,
            300
        ],
        "learning_rates": [
            0.1,
            0.05,
            0.01
        ],
        "max_depth": [
            None,
            3,
            10
        ],
        "subsample": [
            0.5,
            0.8,
            1.0
        ],
        "max_features": [
            "sqrt",
            "log2",
        ],
    }

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*100)
    print(f"MODEL\t\t\t|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*100)
    write_result("")
    write_result("CROSS-VALIDATION PERFORMANCES:")
    write_result("")
    write_result("-"*100)
    write_result(f"MODEL\t\t\t|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    write_result("-"*100)

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

                                print(
                                    f"{model.name}\t|\t" + \
                                    f"{100*np.mean(cv_performance['accuracy']):.2f}%\t|\t" + \
                                    f"{np.mean(cv_performance['precision']):.2f}\t|\t" + \
                                    f"{np.mean(cv_performance['recall']):.2f}\t|\t" + \
                                    f"{np.mean(cv_performance['f1']):.2f}"
                                )
                                write_result(
                                    f"{model.name}\t|\t" + \
                                    f"{100*np.mean(cv_performance['accuracy']):.2f}%\t|\t" + \
                                    f"{np.mean(cv_performance['precision']):.2f}\t|\t" + \
                                    f"{np.mean(cv_performance['recall']):.2f}\t|\t" + \
                                    f"{np.mean(cv_performance['f1']):.2f}"
                                )

    print("")
    write_result("")

if __name__ == "__main__":
    main()
