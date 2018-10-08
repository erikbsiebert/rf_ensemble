# ============================================================================#
#                           File: ModelSelector.py                            #
#                          Author: Erik Berg Siebert                          #
#                           Created: 05th Sep, 2017                           #
# ============================================================================#
# Class holding methods used for model selection.                             #
# ============================================================================#
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class ModelSelector():
    """Class holding methods used for experimenting"""

    def __init__(self, dataset, column_rule):
        # Classifier configuration section
        # Params:
        #     dataset - Dictionary holding dataset
        #         codes: Holds sample code
        #         X:     Holds gene expressions
        #         y:     Holds expected class
        #     column_rule - Dictionary holding the rule for column limitation
        #         columns: Holds the columns ordered by importance
        #         initial: Holds initial quantity of columns
        #         final:   Holds final quantity of columns that must be reached
        #         step:    Holds the step value which increments the quantity of columns each iteration

        self.column_rule = column_rule
        self.X = dataset['X']
        self.y = dataset['y']
        self.classifier = RandomForestClassifier(n_jobs=-1)

    def run(self):
        # Initialize best scores
        best_f1_score = 0.0
        best_columns = 0

        for columns in range(column_rule['initial'], column_rule['final'], column_rule['step']):
            # Limit columns
            current_X, current_y = limit_columns(self.X, self.y, columns)

            # Run prediction
            y_pred = cross_val_predict(self.classifier, current_X, current_y, cv=10)

            # Get F1 Score
            current_f1_score = float(f1_score(y_true, y_pred)) * 100

            # Update best score
            if current_f1_score > best_f1_score:
                best_f1_score, best_columns = [current_f1_score, columns]

        # TO-DO: return trained classifier instead of dictionary
        return {'columns': best_columns, 'f1_score': best_f1_score}


    def limit_columns(self, X, y, n_columns):
        current_columns = column_rule['columns'][:n_columns]
        return [[row[index] for index in current_columns] for row in X], y