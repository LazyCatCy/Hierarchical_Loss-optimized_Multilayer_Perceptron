# One vs All traditional model
# Including MLP, RandomForest and SVM

import os
import numpy as np

from Dataloader import load_data, make_label, data_reshape

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def dataloader():
    """Load Data"""

    # Path
    dataset_path = {
        'PD': r'E:\张春奎项目\终极代码整理\Data\Rats\PD_power',
        'LID': r'E:\张春奎项目\终极代码整理\Data\Rats\LID_power',
        'Nor': r'E:\张春奎项目\终极代码整理\Data\Rats\SD'
    }

    # Train Dataset
    train_data = []
    train_label = []

    for key, path in dataset_path.items():
        files = os.listdir(path)

        for file in files:
            file_name = os.path.splitext(file)[0]

            if file_name not in \
                    ['4_R1', '4_SD1_Power', '4_SD2_Power', '4_SD3_Power']:
                data = load_data(os.path.join(path, file))
                length = data.shape[0]

                train_data.append(data[:int(length * 0.7), :, :])
                train_label.extend(make_label(data[:int(length * 0.7), :, :], key))

    # Test Dataset
    test_data = []
    test_label = []

    for key, path in dataset_path.items():
        if key in ['PD', 'LID']:
            files = os.listdir(path)

            for file in files:
                file_name = os.path.splitext(file)[0]

                if file_name not in \
                        ['4_R1', '4_SD1_Power', '4_SD2_Power', '4_SD3_Power']:
                    data = load_data(os.path.join(path, file))
                    length = data.shape[0]

                    test_data.append(data[int(length * 0.7):, :, :])
                    test_label.extend(make_label(data[int(length * 0.7):, :, :], key))

    train_data, _ = data_reshape(np.concatenate(train_data, axis=0), [0, 1, 2, 3, 4, 5, 6, 7])
    test_data, _ = data_reshape(np.concatenate(test_data, axis=0), [0, 1, 2, 3, 4, 5, 6, 7])

    return train_data, train_label, test_data, test_label


def train(model_pattern, x_train, y_train):
    """One vs All Training"""

    # Choose model
    estimator = None
    if model_pattern == 'mlp':
        estimator = MLPClassifier(
            hidden_layer_sizes=(112,),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True
        )
    elif model_pattern == 'rf':
        estimator = RandomForestClassifier()
    elif model_pattern == 'svm':
        estimator = SVC(probability=True)

    # One vs All Model
    clf = OneVsRestClassifier(estimator)

    # Train
    clf.fit(x_train, y_train)

    return clf


def test(model, x_test):
    """Test"""

    # Test
    y_pred = model.predict(x_test)

    # Categories
    categories = model.classes_

    # Proba
    proba = dict()

    for i in range(len(categories)):
        pred_prob = model.estimators_[i].predict_proba(x_test)
        if categories[i] == 1:
            proba['LID_0'] = pred_prob[:, 0]
            proba['LID_1'] = pred_prob[:, 1]
        elif categories[i] == 0:
            proba['PD_0'] = pred_prob[:, 0]
            proba['PD_1'] = pred_prob[:, 1]
        elif categories[i] == 2:
            proba['Nor_0'] = pred_prob[:, 0]
            proba['Nor_1'] = pred_prob[:, 1]

    non_counts = 0
    for i in range(len(y_pred)):

        if (proba['LID_0'][i] > proba['LID_1'][i]) and \
                (proba['PD_0'][i] > proba['PD_1'][i]):
            y_pred[i] = 2

            non_counts = non_counts + 1

        if y_pred[i] == 'Nor':
            if proba['LID_1'][i] > proba['PD_1'][i]:
                y_pred[i] = 'LID'
            elif proba['LID_1'][i] < proba['PD_1'][i]:
                y_pred[i] = 'PD'
    return y_pred, non_counts


def output_result(model_pattern, train_data, train_label, test_data, test_label):

    # Train
    model = train(model_pattern=model_pattern, x_train=train_data, y_train=train_label)

    # Closed-domain Result
    y_pred_closed = model.predict(test_data)

    closed_accuracy = accuracy_score(test_label, y_pred_closed)
    closed_f1 = f1_score(test_label, y_pred_closed, average='weighted')

    # Open-domain Result
    y_pred_open, non_count = test(model, test_data)

    open_accuracy = accuracy_score(test_label, y_pred_open)
    open_f1 = f1_score(test_label, y_pred_open, average='weighted')

    return closed_accuracy, closed_f1, open_accuracy, open_f1


def main():

    # Load Data
    train_data, train_label, test_data, test_label = dataloader()

    for model in ['svm']:

        print(f'\n{model} Model:')
        print('-' * 30)
        print('Loop\taccuracy(c)\tf1 score(c)\taccuracy(o)\tf1 score(o)')

        for loop in range(10):
            closed_accuracy, closed_f1, open_accuracy, open_f1 = output_result(
                model, train_data, train_label, test_data, test_label
            )
            print(f'{loop}\t{closed_accuracy}\t{closed_f1}\t{open_accuracy}\t{open_f1}')


if __name__ == '__main__':
    main()
