# ============================================================================#
#                           File: FeatureAnalyser.py                          #
#                          Author: Erik Berg Siebert                          #
#                           Created: 05th Sep, 2017                           #
# ============================================================================#
# Class holding methods used for model selection.                             #
# ============================================================================#
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class FeatureAnalyser():
    """Class holding methods used for experimenting"""

    def __init__(self, dataset, column_rule):
        # Classifier configuration section
        self.column_rule
        self.X = dataset['X']
        self.y = dataset['y']
        self.classifier = RandomForestClassifier(n_jobs=-1)

        # TO-DO: return updated column_rule with columns in order of importance
        self.column_rule['columns'] = self.analyze()
        return self.column_rule


    def analyze(self):

        # TO-DO: Train classifier prediction
        self.classifier.fit(current_X, current_y)

        # TO-DO: Return columns in decreasing order of importance
        return self.classifier.importance