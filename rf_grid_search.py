from data_loading import ReviewLoader
from preprocessing import ReviewProcessor
from models import RandomForestClassifier

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

    results_file = "rf_grid_search_results.txt"
    def write_result(line: str):
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# RandomForestClassifier(name="RandomForest(basic)", criterion="gini", n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt"),
# RandomForestClassifier(name="RandomForest(n=200)", criterion="entropy", n_estimators=200, max_depth=None, min_samples_split=80, min_samples_leaf=2, max_features="log2"),

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
        "criterion": [
            "gini",
            "entropy"
        ],
        "n_estimators": [
            100,
            200,
            300
        ],
        "max_depth": [
            None,
            3,
            10
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
            for crit in grid['criterion']:
                for n_est in grid['n_estimators']:
                    for max_depth in grid['max_depth']:
                        for min_split in grid['min_samples_splits']:
                            for min_leaf in grid['min_samples_leaf']:
                                for max_feat in grid['max_features']:
                                
                                    # Specify which model configuration to evaluate
                                    model = RandomForestClassifier(
                                        name=f"DF{df}_BG{include_bigrams}_CR{crit}_NE{n_est}_MD{max_depth}_MSS{min_split}_MSL{min_leaf}_MF{max_feat}",
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
