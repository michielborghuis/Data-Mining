from models import SingleClassificationTree
import numpy as np
from itertools import product

np.random.seed(42)

print("")
print("CROSS-VALIDATION PERFORMANCES:")
print("")
print("-"*85)
print(f"{'MODEL':<25}|\tACC\t|\tPREC\t|\tREC\t|\tF1")
print("-"*85)

best_score = -1.0
best_parameters = None
best_accuracy = None
best_prec = None
best_rec = None

grid = {
    'max_depth': [None, 5, 10],
    'ccp_alpha': [0, 0.001, 0.01],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4], 
}


for max_depth, min_samples_split, min_samples_leaf, ccp_alpha in product(grid['max_depth'],grid['min_samples_split'],grid['min_samples_leaf'],grid['ccp_alpha'] ):
    model = SingleClassificationTree(
        name="SingleClassificationTree",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha
    )
    model.include_bigrams = False
    val_performance = model.get_validation_performance(n_folds=10, n_repeats=1)
    
    mean_f1 = np.mean(val_performance['f1'])
    mean_acc = np.mean(val_performance['accuracy'])
    mean_prec = np.mean(val_performance['precision'])
    mean_rec = np.mean(val_performance['recall'])
    
    if mean_f1 > best_score:
        best_score = mean_f1
        best_accuracy = mean_acc
        best_prec = mean_prec
        best_rec = mean_rec
        
        best_parameters = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'ccp_alpha': ccp_alpha
        }

    print(
            f"{model.name:<25}|\t" + \
            f"{100*np.mean(val_performance['accuracy']):.3f}%\t|\t" + \
            f"{np.mean(val_performance['precision']):.3f}\t|\t" + \
            f"{np.mean(val_performance['recall']):.3f}\t|\t" + \
            f"{np.mean(val_performance['f1']):.3f}"
        )

# eindresultaat printen
print("\nBeste parametercombinatie:")
print(best_parameters)
print(f"Beste gemiddelde F1-score: {best_score:.4f}")
print(f"accuracy: {best_accuracy:.4f}")
print(f"precision: {best_prec:.4f}")
print(f"recal: {best_rec:.4f}")
