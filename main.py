from data_loading import ReviewLoader
from preprocessing import ReviewProcessor
from models import NaiveBayesClassifier

def main() -> None:
    loader = ReviewLoader()
    processor = ReviewProcessor()

    train_reviews, train_labels = loader.load_train_reviews()
    train_X = processor.process_train_reviews(train_reviews, include_bigrams=False)
    train_X = processor.filter_rare_terms(train_X, min_review_freq=.01)

    models = [
        NaiveBayesClassifier(name="NaiveBayes(Laplace)", smoothing_alpha=1.0),
        NaiveBayesClassifier(name="NaiveBayes(alpha=.1)", smoothing_alpha=0.1)
    ]

    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*100)
    print(f"MODEL\t\t\t|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*100)
    for model in models:
        cv_performance = model.get_validation_performance(train_X, train_labels)
        print(
            f"{model.name}\t|\t" + \
            f"{100*cv_performance['accuracy']:.2f}%\t|\t" + \
            f"{cv_performance['precision']:.2f}\t|\t" + \
            f"{cv_performance['recall']:.2f}\t|\t" + \
            f"{cv_performance['f1']:.2f}"
        )
    print("")

    # NOTE: we should only start looking at test set performance in a couple of weeks or so
    #   -> modelling/hyperparameter choices should NOT be based on test set performance 

    #test_reviews, test_labels = loader.load_test_reviews()
    #test_X = processor.process_test_reviews(test_reviews, include_bigrams=False)

if __name__ == "__main__":
    main()