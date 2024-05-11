# import numpy as np
# from hyperopt import base
# base.have_bson = False
# from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
# from sklearn.metrics import accuracy_score,confusion_matrix
# from hyperopt import tpe
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from graphviz import Digraph
# from datetime import datetime
# import sys
# from sklearn.datasets import load_digits
# from sklearn.datasets import load_iris
# import json



def pipeline_generator(data_file_path):
    
    # dataframe = pd.read_csv(data_file_path)
    # dataframe = dataframe.dropna()
    # data = dataframe.values
    # X, y = data[:, :-1], data[:, -1]
    # y = y.astype('int32')
    iris = load_iris()

    X = iris.data
    y = iris.target
    test_size = int(0.2 * len(y))
    np.random.seed(13)
    indices = np.random.permutation(len(X))
    X_train = X[indices[:-test_size]]
    y_train = y[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    y_test = y[indices[-test_size:]]
    
    # Instantiate a HyperoptEstimator with the search space and number of evaluations
    estim = HyperoptEstimator(classifier=any_classifier("my_clf"),
                             preprocessing=any_preprocessing("my_pre"),
                             loss_fn=accuracy_score,
                              algo=tpe.suggest,
                              max_evals=10,
                              trial_timeout=300,  verbose= False)

    estim.fit(X_train, y_train)


    dataset_name = os.path.basename(data_file_path).split('.')[0]
    file_name = f"accuracy_{dataset_name}.json"
    # Calculate accuracy
    accuracy = estim.score(X_test, y_test)

    # Store accuracy and dataset name in a dictionary
    data = {
        "dataset_name": dataset_name,
        "accuracy": accuracy
    }

    if not os.path.exists('hyperopt-results/metric'):
            os.makedirs('hyperopt-results/metric')

    # Write data to JSON file
    with open(f'hyperopt-results/metric/{file_name}', "w") as json_file:
        json.dump(data, json_file)

    # print(estim.best_model())

    y_pred = estim.predict(X_test)
    pipeline= estim.best_model()

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if not os.path.exists('hyperopt-results/images'):
        os.makedirs('hyperopt-results/images')


    img_filename = f"hyperopt-results/images/{dataset_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}-conf_matrix.png"

    plt.savefig(img_filename)


    # Generate workflow visualization and save it to a file:

    if not os.path.exists('hyperopt-results/dataflows'):
        os.makedirs('hyperopt-results/dataflows')
    graph_filename = f"hyperopt-results/dataflows/{dataset_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}-dataflow"
    graph = Digraph('DataFlow', filename=graph_filename)
    graph.attr(rankdir='LR')

    graph.node('Dataset', 'Dataset: \n'+dataset_name, fillcolor='orange', style='filled',shape='rectangle')
    graph.node('Visualization', 'Visualization: \n Confusion Matrix', fillcolor='lightgreen', style='filled',shape='rectangle')
    algo = pipeline['learner']
    algo_name = str(algo).split('(')[0]
    graph.node('Algorithm', 'Algorithm: \n'+algo_name, fillcolor='lightblue', style='filled',shape='rectangle')
   
    if len(pipeline['preprocs']) != 0:
        prepro = pipeline['preprocs'][0]
        prepro_name = str(prepro).split('(')[0]
        graph.node('Preprocessing', 'Preprocessing: \n'+prepro_name, fillcolor='lightblue', style='filled',shape='rectangle')

        graph.edge('Dataset', 'Preprocessing')
        graph.edge('Preprocessing', 'Algorithm')
        graph.edge('Algorithm','Visualization')
    else:
        graph.edge('Dataset', 'Algorithm')
        graph.edge('Algorithm','Visualization')

    graph.format = 'png'
    graph.render(view=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path of csv file>")
        sys.exit(1)

    file_path = sys.argv[1]
    pipeline_generator(file_path)