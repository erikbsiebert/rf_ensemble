# ============================================================================#
#                          File: EnsembleBuilder.py                           #
#                          Author: Erik Berg Siebert                          #
#                           Created: 05th Sep, 2017                           #
# ============================================================================#
# Class holding methods used for model selection.                             #
# ============================================================================#
from FeatureAnalyser import FeatureAnalyser
from itertools import combinations
from ModelSelector import ModelSelector


class EnsembleBuilder():
    """Class holding methods used for experimenting"""

    def __init__(self):
        # Classifier configuration section
        self.read_configuration()
        self.column_rule = {
            'initial': int(self.config['initial']),
            'final': int(self.config['final']),
            'step': int(self.config['step'])
        }
        print(self.config)


    def read_configuration(self):
        self.config = {}
        with open("configuration.txt") as f:
            for line in f:
               (key, val) = line.split()
               self.config[key] = val


    def build_dataset(self, class1, class2):
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

        return {'codes': codes, 'X': X, 'y': y}


    def handle_one_vs_one(self):
        predictors = []

        for combination in combinations(range(int(self.config['num_classes'])), 2):
            print(combination)
            dataset = self.build_dataset([combination[0]], [combination[1]])
            predictor = self.execute_combination(dataset)
            predictors.append(predictor)

        return predictors


    def handle_one_vs_all(self):
        predictors = []

        for sample_class in range(int(self.config['num_classes'])):
            all_classes = range(int(self.config['num_classes']))
            del all_classes[sample_class]
            dataset = self.build_dataset([sample_class], all_classes)
            predictor = self.execute_combination(dataset)
            predictors.append(predictor)

        return predictors


    def execute_combination(self, dataset):
        updated_column_rule = FeatureAnalyser(dataset, self.column_rule)       
        return ModelSelector(dataset, updated_column_rule)


    def build(self):
        if self.config['decomp_method'] == 'one-vs-one':
            predictors = self.handle_one_vs_one()

        elif self.config['decomp_method'] == 'one-vs-all':
            predictors = self.handle_one_vs_all()

        else:
            print('Unknown decomposition method!')
            return -1

ensemble_builder = EnsembleBuilder()