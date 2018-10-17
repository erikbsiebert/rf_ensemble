# ============================================================================#
#                           File: ModelSelector.py                            #
#                          Author: Erik Berg Siebert                          #
# ============================================================================#
# Class holding methods used for model selection.                             #
# ============================================================================#
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from Utils import limit_columns


class ModelSelector():
    """Class holding methods used for experimenting"""

    def __init__(self, dataset, column_rule, n_estimators):
        """ Classifier configuration section
            Params:
                dataset - Dictionary holding dataset
                    codes: Holds sample code
                    X:     Holds gene expressions
                    y:     Holds expected class
                column_rule - Dictionary holding the rule for column limitation
                    columns: Holds the columns ordered by importance
                    initial: Holds initial quantity of columns
                    final:   Holds final quantity of columns that must be reached
                    step:    Holds the step value which increments the quantity of columns each iteration
        """

        self.column_rule = column_rule
        self.X = dataset['X']
        self.y = dataset['y']
        self.classifier = RandomForestClassifier(criterion='entropy', n_jobs=-1, random_state=1337, n_estimators=n_estimators)


    def run(self):
        print('\tSelecting best model for combination...')
        print('\tFrom %d to %d each %d' % (self.column_rule['initial'], self.column_rule['final'], self.column_rule['step']))
        best_f1_score = 0.0
        best_columns = 0
        best_n_columns = 0

        for n_columns in range(self.column_rule['initial'], self.column_rule['final'] + 1, self.column_rule['step']):
            print('\t\tTesting with %d columns...' % n_columns, end='')

            # Limit columns
            current_columns = self.column_rule['columns'][:n_columns]
            current_X, columns = limit_columns(self.X, current_columns)

            # Run prediction
            try:
                y_pred = cross_val_predict(self.classifier, current_X, self.y, cv=10, n_jobs=-1)
            except ValueError as e:
                y_pred = cross_val_predict(self.classifier, current_X, self.y, cv=5, n_jobs=-1)

            # Get F1 Score
            current_f1_score = float(f1_score(self.y, y_pred)) * 100
            print(' -> F1: %.2f%%' % current_f1_score)

            # Update best score
            if current_f1_score > best_f1_score:
                best_f1_score, best_columns, best_n_columns = [current_f1_score, columns, n_columns]

            # Stop iteration when perfect precision is achieved
            if current_f1_score == 100.00:
                break

        print('\tBest F1 at %.2f%% with %d columns' % (best_f1_score, best_n_columns))

        # Since cross-validation won't train the classifier, it needs to be trained
        final_X, _ = limit_columns(self.X, best_columns)
        self.classifier.fit(final_X, self.y)

        results = {
            'classifier': self.classifier, 
            'columns': best_columns, 
            'n_columns': best_n_columns, 
            'f1_score': best_f1_score
        }

        return results
