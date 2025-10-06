from data_loading import ReviewLoader
from preprocessing import ReviewProcessor
from models import NaiveBayesClassifier,Logisticregression,SingleClassificationTree,RandomForestClassifier,GradientBoostingClassifier


import numpy as np
import random

from sklearn import tree; 
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    loader = ReviewLoader()
    processor = ReviewProcessor()

    # Load reviews from files
    train_reviews, train_labels = loader.load_train_reviews()

    # Preprocess reviews to convert to count matrix
    train_X = processor.process_train_reviews(train_reviews, include_bigrams=False)

    # Remove features (uni-/bigrams) that occur in very few documents
    train_X = processor.filter_rare_terms(train_X, min_review_freq=.005)

    print(train_X.shape)
    #exit()

    # Specify which model configurations to evaluate

    models = [ 
        
        NaiveBayesClassifier(name="NaiveBayes(Laplace)", smoothing_alpha=1.0),
        NaiveBayesClassifier(name="NaiveBayes(alpha=.1)", smoothing_alpha=0.1),
        SingleClassificationTree(name="SingleTree(alpha=.0001)", ccp_alpha=0.0001),
        SingleClassificationTree(name="SingleTree(alpha=.001)", ccp_alpha=0.001),
        SingleClassificationTree(name="SingleTree(alpha=.05)", ccp_alpha=0.05),
        SingleClassificationTree(name="SingleTree(alpha=.01)", ccp_alpha=0.01),
        SingleClassificationTree(name="SingleTree(alpha=.1)", ccp_alpha=0.1),
        Logisticregression(name="logisticRegression(alpha=.001)", c =0.001),
        Logisticregression(name="logisticRegression(alpha=.01)", c =0.01),
        Logisticregression(name="logisticRegression(alpha=.1)", c =0.1),
        Logisticregression(name="logisticRegression(alpha== 1)", c =1),
        Logisticregression(name="logisticRegression(alpha== 10)", c =10),
        Logisticregression(name="logisticRegression(alpha== 100)", c =100),
        # NaiveBayesClassifier(name="NaiveBayes(alpha=2)", smoothing_alpha=2.0),
        # NaiveBayesClassifier(name="NaiveBayes(alpha=1)", smoothing_alpha=1.0),
        # NaiveBayesClassifier(name="NaiveBayes(alpha=.1)", smoothing_alpha=0.1),
        # NaiveBayesClassifier(name="NaiveBayes(alpha=1e-5)", smoothing_alpha=1e-5),
        # RandomForestClassifier(name="RandomForest(n=50)", n_estimators=50),
        # RandomForestClassifier(name="RandomForest(n=100)", n_estimators=100),
        # RandomForestClassifier(name="RandomForest(n=200)", n_estimators=200),
        GradientBoostingClassifier(name="GradientBoosting(n=300)", n_estimators=300),
        #GradientBoostingClassifier(name="GradientBoosting(n=1000)", n_estimators=1000)
    ]
    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*100)
    print(f"MODEL\t\t\t|\tACC\t|\tPREC\t|\tREC\t|\tF1")
    print("-"*100)
    for model in models:
        cv_performance = model.get_validation_performance(train_X, train_labels, n_features=1250)
        print(
            f"{model.name}\t|\t" + \
            f"{100*cv_performance['accuracy']:.2f}%\t|\t" + \
            f"{cv_performance['precision']:.2f}\t|\t" + \
            f"{cv_performance['recall']:.2f}\t|\t" + \
            f"{cv_performance['f1']:.2f}"
        )
    print("")

    #models[0].analyse_feature_importances(index_to_word_mapping=processor.index_token_dict)
    models[12].analyse_feature_importances(index_to_word_mapping=processor.index_token_dict)
    
    # NOTE: we should only start looking at test set performance in a couple of weeks or so
    #   -> modelling/hyperparameter choices should NOT be based on test set performance 

    #test_reviews, test_labels = loader.load_test_reviews()
    #test_X = processor.process_test_reviews(test_reviews, include_bigrams=False)

if __name__ == "__main__":
    main()