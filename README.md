Code base for the practical assignment part of the Data Mining course, as completed by Gijs van Nieuwkoop, Michiel Borghuis and Stijn van Huët.

All external libraries/packages necessary to run the code present in this code base can be found in the accompanying requirements.txt file.

To reproduce results included in the corresponding written report, simply run ```python main.py ``` from this directory.

Some explanation about the organisation of the code:

- **main.py** → script used to generate all final results (performance, statistical testing, feature analysis)
- **/models/** → contains implementations of classifiers
- **/hpo_scripts/** → contains scripts used to run hyperparameter optimization procedures (grid searches etc.), can be ran using ```python -m hpo_scripts.{FILENAME}```
- **/utils/** → contains general purpose auxiliary scripts (data loading, preprocessing etc.)
