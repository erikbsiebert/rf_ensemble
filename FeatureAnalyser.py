# ============================================================================#
#                           File: FeatureAnalyser.py                          #
#                          Author: Erik Berg Siebert                          #
# ============================================================================#
# Class holding methods used for model selection.                             #
# ============================================================================#
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from operator import itemgetter


class FeatureAnalyser():
    """Class holding methods used for experimenting"""

    def __init__(self, dataset, column_rule, n_estimators):
        # Classifier configuration section
        self.column_rule = column_rule
        self.X = dataset['X']
        self.y = dataset['y']
        self.classifier = RandomForestClassifier(criterion='entropy', n_jobs=-1, n_estimators=n_estimators)


    def analyze(self):
        print('\tAnalyzing features...')

        self.classifier.fit(self.X, self.y)

        index_result = [{'id': x, 'importance': 0.0} for x in range(len(self.X[0]))]
        for index, num in enumerate(self.classifier.feature_importances_):
            index_result[index]['importance'] += num

        sorted_index_result = sorted(index_result, key=itemgetter('importance'), reverse=True)

        self.column_rule['columns'] = sorted_index_result

        return self.column_rule
