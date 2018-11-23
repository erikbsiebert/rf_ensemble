# RF Ensemble
An experimentation framework to evaluate the use of Random Forest for classification of gene expression of brain cells.

 [EnsembleBuilder]: <https://github.com/erikbsiebert/rf_ensemble/blob/master/ModelSelector.py>
 [configuration]: <https://github.com/erikbsiebert/rf_ensemble/blob/master/configuration.txt>
 [python]: <https://www.python.org/downloads/>
 [sklearn]: <https://scikit-learn.org/stable/install.html>
 [article]: <https://github.com/erikbsiebert/rf_ensemble/blob/master/Article.pdf>

## Usage
The framework has been developed in a way to automate the entire experimentation process, therefore only initial configuration is required. The configuration file named [*configurationt.txt*][configuration] contains the following properties:

| Property | Description |
| -------- | ----------- |
| *decomp_method* | specifiy the decomposition method to be used to build the ensemble. Only 'one-vs-one' and 'one-vs-all' values are supported. |
| *num_classes* | Number of classes in the classification experiment. |
| *initial* | Initial amount of features to be tested on feature selection phase. |
| *final* | Final amount of features to be tested on feature selection phase. |
| *step* | Step size for each iteration on feature selection phase. |
| *dataset_path* | Dataset file path. This path is relational to the script's directory. File must be in tsv format. |
| *testset_path* | Testset file path. This path is relational to the script's directory. File must be in tsv format. |
| *n_estimators_fa* | Number of estimators to be used in Random Forest on feature analysis phase. |
| *n_estimators_ms* | Number of estimators to be used in Random Forest on feature/model selection phase. |

To start the experiment, run [EnsembleBuilder.py][EnsembleBuilder].

## Dataset

The dataset and testset must meet the following rules:
- First line is the header of the dataset (testset excluded)
- First column holds sample's code
- Second column holds sample's real class code

## Output

At the end of the experiment, 2 files are created:
- results.html: an interative report of the results of each classifier, quantity of columns used by each classifier and its F1 scores; and confusion matrixes given the sample's condition and sample's tissue alone.
- logs.txt: log results of the last experiment run. Results.html is created from this log file.

## Dependencies
This framework depends on [Python 3.5+][python] and the [*sklearn*][sklearn] machine learning library.

## Final results
An unpublished article has been written on the final results of the main experiment which can be found [here][article].
