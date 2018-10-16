# ============================================================================#
#                           File: Utils.py                                    #
#                          Author: Erik Berg Siebert                          #
# ============================================================================#
# File holding utility functions used throughout the framework.               #
# ============================================================================#
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import csv
import numpy
from os.path import isfile
from sklearn.metrics import f1_score, confusion_matrix


def limit_columns(X, current_columns):
    return [[row[column['id']] for column in current_columns] for row in X], current_columns

def log_results(results, config, predictions):
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

        f.write('\n' + ','.join([str(pred) for pred in predictions['y']]))
        f.write('\n' + ','.join([str(pred) for pred in predictions['y_pred']]))

        for result in results:
            entry = '\n%s,%s,%d,%.2f' % (
                result['class1'],
                result['class2'],
                result['n_columns'],
                result['f1_score']
            )
            f.write(entry)

    #for result in results:
    #    with open('columns_%s_vs_%s.txt' % (result['class1'], result['class2']), 'w') as f:
    #        f.write(','.join([str(number) for number in result['columns']]))

def get_available_filename():
    filename = 'results.html'
    n_try = 0
    while isfile(filename):
        n_try += 1
        filename = 'results_%d.html' % n_try
    return filename

def get_tissue_classes(y_list):
    y_new = []

    for y in y_list:
        if y < 2:
            y_new.append(0)
        elif y < 5:
            y_new.append(1)
        else:
            y_new.append(2)

    return y_new

def get_prediction_results(y, y_pred):
    tissue_condition_f1 = float(f1_score(y, y_pred, average='micro')) * 100
    #print('Tissue-Condition F1 Score: %.2f' % tissue_condition_f1)
    tissue_condition_cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4, 5, 6])
    #print(tissue_condition_cm)

    y_tissue = get_tissue_classes(y)
    y_pred_tissue = get_tissue_classes(y_pred)

    tissue_f1 = float(f1_score(y_tissue, y_pred_tissue, average='micro')) * 100
    #print('Tissue-Condition F1 Score: %.2f' % tissue_f1)
    tissue_cm = confusion_matrix(y_tissue, y_pred_tissue, labels=[0, 1, 2])
    #print(tissue_cm)

    return tissue_condition_f1, tissue_condition_cm, tissue_f1, tissue_cm

def format_cm(cm, footer):
    cm = [['<b>%d</b>' % index] + list(row) for index, row in enumerate(cm)]
    cm = [[value if value != 0 else '-' for value in row] for row in cm]
    cm.append(footer)
    cm = list(map(list, zip(*cm)))

    return cm

def plot_results():
    print('Creating plot from logged results...')
    # Data Setup
    x, y, rows, header = [],[],[],[]
    with open('logs.txt', 'r') as log_file_stream:
        log_file_rows = csv.reader(log_file_stream, delimiter=',')
        rows = [row for row in log_file_rows]

    header, y_exp, y_pred = rows[:3]
    del rows[:3]
    y_exp = [int(value) for value in y_exp]
    y_pred = [int(value) for value in y_pred]

    for row in rows:
        x.append('%s vs. %s' % (row[0], row[1]))
        y.append(float(row[3])/100.0)

    table_data = [['<b>%s vs. %s</b>' % (row[0], row[1]), row[2], row[3]] for row in rows]
    table_data = list(map(list, zip(*table_data)))

    f1_data, cm_data, f1_t_data, cm_t_data = get_prediction_results(y_exp, y_pred)

    cm_data = format_cm(cm_data, ['','','','<b>F1 Score:</b>','<b>%.2f%%</b>' % f1_data,'','',''])
    cm_t_data = format_cm(cm_t_data, ['','<b>F1 Score:</b>','<b>%.2f%%</b>' % f1_t_data,''])

    table = go.Table(
        domain=dict(x=[0, 1.],
                    y=[.25, .55]),
        header=dict(
            #values=list(df.columns[1:]),
            values=['<b>Classifier</b>', '<b>Columns</b>', '<b>F1 Score</b>'],
            font=dict(color = 'white', size=14),
            line = dict(color = '#506784'),
            align = ['center'],
            height=30,
            fill = dict(color='grey'),
        ),
        cells=dict(
            values = table_data,
            line = dict(color = '#506784'),
            align = ['center','left'],
            fill = dict(color='#f5f5fa'),
            font = dict(color = '#506784', size = 12),
            height=25,
        )
    )

    table2 = go.Table(
        domain=dict(x=[0, .45],
                    y=[0, .23]),
        header=dict(
            #values=list(df.columns[1:]),
            values=[
                '<b>Confusion</b><br><b>Matrix</b>',
                '<b>0</b>',
                '<b>1</b>',
                '<b>2</b>',
                '<b>3</b>',
                '<b>4</b>',
                '<b>5</b>',
                '<b>6</b>'
            ],
            font=dict(color = 'white', size=14),
            line = dict(color = '#506784'),
            align = ['center'],
            height=30,
            fill = dict(color='grey'),
        ),
        cells=dict(
            values = cm_data,
            line = dict(color = '#506784'),
            align = ['center'],
            fill = dict(color='#f5f5fa'),
            font = dict(color = '#506784', size = 12),
            height=25,
        )
    )

    table3 = go.Table(
        domain=dict(x=[.65, .9],
                    y=[0, .20]),
        header=dict(
            values=[
                '<b>Confusion</b><br><b>Matrix</b>',
                '<b>0</b>',
                '<b>1</b>',
                '<b>2</b>'
            ],
            font=dict(color = 'white', size=14),
            line = dict(color = '#506784'),
            align = ['center'],
            height=30,
            fill = dict(color='grey'),
        ),
        cells=dict(
            values = cm_t_data,
            line = dict(color = '#506784'),
            align = ['center'],
            fill = dict(color='#f5f5fa'),
            font = dict(color = '#506784', size = 12),
            height=25,
        )
    )

    # Classifier Score Chart configuration
    precision_trace = go.Bar(
        x=x,
        y=y,
        name='F1 Score',
        marker=dict(
            color='rgba(0,150,0,.7)'
        ),
        xaxis='x1', yaxis='y1'
    )

    mean_trace = go.Scatter(
        x=x,
        y=[numpy.mean(y)] * len(x),
        name='Mean F1 Score',
        mode = 'lines',
        line = dict(
            color = 'rgb(205, 12, 24)',
            width = 4,
            dash = 'dash'
        ),
        xaxis='x1', yaxis='y1'
    )

    max_trace = go.Scatter(
        x=x,
        y=[max(y)] * len(x),
        name='Max F1 Score',
        mode = 'lines',
        line = dict(
            color = 'rgb(0, 0, 255)',
            width = 4,
            dash = 'dash'
        ),
        xaxis='x1', yaxis='y1'
    )

    min_trace = go.Scatter(
        x=x,
        y=[min(y)] * len(x),
        name='Min F1 Score',
        mode = 'lines',
        line = dict(
            color = 'rgb(0, 0, 0)',
            width = 4,
            dash = 'dash'
        ),
        xaxis='x1', yaxis='y1'
    )

    axis=dict(
        showline=True,
        zeroline=False,
        showgrid=True,
        mirror=True,
        ticklen=4, 
        gridcolor='#ffffff',
        tickfont=dict(size=10)
    )

    layout2 = dict(
        width=1825,
        height=1600,
        autosize=False,
        title='Test Results with %s, %s classes, from %s columns to %s each %s' %
            (header[0], header[1], header[2], header[3], header[4]),
        margin = dict(t=75, l=50),
        showlegend=False,          
        xaxis1=dict(axis, **dict(domain=[0, 1], anchor='y1', showticklabels=False)),
        xaxis2=dict(axis, **dict(domain=[0, 1], anchor='y2', showticklabels=False)),          
        yaxis1=dict(axis, **dict(domain=[.6, 1], anchor='x1', range=[min(y)-.02, 1.0]), tickformat='%'),  
        yaxis2=dict(axis, **dict(domain=[.6, 1], anchor='x2', range=[min(y)-.02, 1.0]), tickformat='%'),
        plot_bgcolor='rgba(228, 222, 249, 0.65)'
    )

    fig2 = dict(data=[table, table2, table3, precision_trace, mean_trace, max_trace, min_trace], layout=layout2)
    filename = get_available_filename()
    py.plot(fig2, filename=filename)
    print('Results plotted in %s' % filename)
