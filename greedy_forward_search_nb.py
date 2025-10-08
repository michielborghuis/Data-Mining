from models import NaiveBayesClassifier

import matplotlib.pyplot as plt
import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def main() -> None:
    #print("UNIGRAM (continued, best=565)")
    print("BIGRAM")
    #print("UNIGRAM")
    print("")
    print("CROSS-VALIDATION PERFORMANCES:")
    print("")
    print("-"*30+" "*8+"-"*30)
    print(f"MODEL\t\t|\tACC\t\tMAX ACC\t\t|\tSE")
    print("-"*30+" "*8+"-"*30)

    features = 143
    step_size = 1
    all_accs = []
    best_acc = .8312
    standard_error = .03
    stall_counter = 0
    patience = 50
    while True:
        try:
            #model = NaiveBayesClassifier(
            #    smoothing_alpha=1.0,
            #    min_df=.005,
            #    features=features,
            #    feature_mode="include",
            #    include_bigrams=False,
            #    name=f"{features} FEATURES"
            #)

            model = NaiveBayesClassifier(
                smoothing_alpha=0.5,
                min_df=.01,
                features=features,
                feature_mode="include",
                include_bigrams=True,
                name=f"{features} FEATURES"
            )
            
            val_accs = model.get_validation_performance(n_folds=10, n_repeats=1)['accuracy']
            cur_acc = np.mean(val_accs)
            all_accs.append(cur_acc)

            # STANDARD ERROR VARIANT
            if cur_acc >= best_acc:
                best_acc = cur_acc
                standard_error = np.std(val_accs)/np.sqrt(len(val_accs))
                stall_counter =0
            else:
                stall_counter += 1

            print(
                f"{model.name}\t|\t" + \
                f"{100*cur_acc:.2f}%\t\t" + \
                f"{100*best_acc:.2f}%\t\t|\t" + \
                #f"{stall_counter}"
                f"{100*standard_error:.2f}%"
            )

            #if stall_counter == patience:
            if best_acc - cur_acc > standard_error:
                print("Exiting!")
                break

            features += step_size
        except:
            print(f"Exiting! (presumably, features={features} exceeds total number of features)")
            break
    
    plt.plot(np.arange(0, features+step_size, step_size)[:len(all_accs)], all_accs, marker='s')
    plt.xlabel("# Of Features")
    plt.ylabel("Cross-Validation Accuracy")
    plt.grid(True)
    plt.title("k-Forward Search Unigrams Results")
    plt.show()

if __name__ == "__main__":
    main()