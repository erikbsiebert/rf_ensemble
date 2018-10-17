# ============================================================================#
#                          File: EnsembleBuilder.py                           #
#                         Author: Erik Berg Siebert                           #
# ============================================================================#
# Class holding methods used for model selection.                             #
# ============================================================================#
from FeatureAnalyser import FeatureAnalyser
from itertools import combinations
from ModelSelector import ModelSelector
from Utils import limit_columns, log_results, plot_results

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class EnsembleBuilder():
    """Class holding methods used for experimenting"""

    def __init__(self):
        self.read_configuration()
        self.column_rule = {
            'initial': int(self.config['initial']),
            'final': int(self.config['final']),
            'step': int(self.config['step'])
        }


    def read_configuration(self):
        print('Reading configuration file...')
        self.config = {}

        with open("configuration.txt") as f:
            for line in f:
               (key, val) = line.split()
               self.config[key] = val


    def build_dataset(self, class1, class2):
        print('\tBuilding data set...', end=' ')

        codes, X, y = [[], [], []]
        with open(self.config['dataset_path'], 'r') as f:
            f.readline()
            for line in f:
                data = line.split('\t')
                sample_class = int(data[1])
                if sample_class in class1:
                    codes.append(data[0])
                    X.append([float(value) for value in data[2:]])
                    y.append(0)
                elif sample_class in class2:
                    codes.append(data[0])
                    X.append([float(value) for value in data[2:]])
                    y.append(1)

        print('%d samples' % len(X))
        return {'codes': codes, 'X': X, 'y': y}


    def handle_one_vs_one(self):
        print('\nBuilding classifiers with OVO decomposition method...')
        results = []

        for combination in combinations(range(int(self.config['num_classes'])), 2):
            print('\nExecuting procedure for combination %d and %d...' % (combination[0], combination[1]))
            dataset = self.build_dataset([combination[0]], [combination[1]])
            result = self.execute_combination(dataset)
            result['class1'] = combination[0]
            result['class2'] = combination[1]
            results.append(result)

        return results


    def handle_one_vs_all(self):
        print('\nBuilding classifiers with OVA decomposition method...')
        results = []

        for sample_class in range(int(self.config['num_classes'])):
            print('\nExecuting procedure for combination %d and rest...' % sample_class)
            all_classes = [c for c in range(int(self.config['num_classes']))]
            del all_classes[sample_class]
            dataset = self.build_dataset([sample_class], all_classes)
            result = self.execute_combination(dataset)
            result['class1'] = sample_class
            result['class2'] = 'rest'
            results.append(result)

        return results


    def execute_combination(self, dataset):
        feature_analyser = FeatureAnalyser(dataset, self.column_rule, self.config['n_estimators_fa'])
        updated_column_rule = feature_analyser.analyze()
        model_selector = ModelSelector(dataset, updated_column_rule, self.config['n_estimators_ms'])
        result = model_selector.run()
        return result


    def predict_ovo(self, X):
        scores = [0] * int(self.config['num_classes'])

        for predictor in self.predictors:
            limited_X, _ = limit_columns([X], predictor['columns'])
            prediction = predictor['classifier'].predict(limited_X)[0]
            prediction = int(predictor['class1']) if prediction == 0 else int(predictor['class2'])
            scores[prediction] += 1

        print(scores)

        y = scores.index(max(scores))

        return y


    def predict_ova(self, X):
        scores = []

        for predictor in self.predictors:
            limited_X, _ = limit_columns([X], predictor['columns'])
            scores.append(predictor['classifier'].predict_proba(limited_X)[0][0])

        print(scores)

        y = scores.index(max(scores))

        return y


    def run_prediction(self):
        y = []
        y_pred = []
        with open(self.config['testset_path'], 'r') as f:
            f.readline()
            for line in f:
                data = line.split('\t')
                expected_y = int(data[1])
                y.append(expected_y)
                X = [float(value) for value in data[2:]]

                prediction = -1
                if self.config['decomp_method'] == 'one-vs-one':
                    prediction = self.predict_ovo(X)

                elif self.config['decomp_method'] == 'one-vs-all':
                    prediction = self.predict_ova(X)

                print('Real %d -> Predicted %d' % (expected_y, prediction))

                y_pred.append(prediction)

        return {'y': y, 'y_pred': y_pred}


    def build(self):
        if self.config['decomp_method'] == 'one-vs-one':
            self.predictors = self.handle_one_vs_one()

        elif self.config['decomp_method'] == 'one-vs-all':
            self.predictors = self.handle_one_vs_all()

        else:
            print('Unknown decomposition method!')
            return -1

        predictions = self.run_prediction()
        log_results(self.predictors, self.config, predictions)
        plot_results()

ensemble_builder = EnsembleBuilder()
ensemble_builder.build()
