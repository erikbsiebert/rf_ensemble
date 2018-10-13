# ============================================================================#
#                           File: Utils.py                                    #
#                          Author: Erik Berg Siebert                          #
# ============================================================================#
# File holding utility functions used throughout the framework.               #
# ============================================================================#

def limit_columns(X, current_columns):
    return [[row[column['id']] for column in current_columns] for row in X], current_columns

def log_results(results, config):
    with open('logs.txt', 'w') as f:
        configs = '%s,%s,%s,%s,%s,%s' % (
            config['decomp_method'],
            config['num_classes'],
            config['initial'],
            config['final'],
            config['step'],
            config['dataset_path']
        )
        f.write(configs)

        for result in results:
            entry = '\n%s,%s,%d,%.2f' % (
                result['class1'],
                result['class2'],
                result['n_columns'],
                result['f1_score']
            )
            f.write(entry)
