import os
import sys
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from sklearn import svm
from hmmlearn import hmm

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn import metrics

from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

DATA_DIR = "../AIBO/AIBO/wav/"
TRAIN_DIR = "../AIBO/AIBO/wav/AIBO-O_ARFF"
TEST_DIR = "../AIBO/AIBO/wav/AIBO-M_ARFF"
EMODB_DIR = '../EMO-DB_ARFF'
CLASSIFIERS = "./classifiers/"


class ArffReader:

    def __init__(self, filepath):
        self.attributes = list()
        self.data = list()
        with open(filepath, 'r') as file:
            file_content = file.readlines()
        for line in file_content:
            if line.startswith("@attribute"):
                self.attributes.append([line.split()[1], line.split()[2]])
        self.data = [eval(val) for val in line.strip('\n').split(
            ',') if val != '?' and isinstance(eval(val), float)]


def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print('Confusion matrix')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():

    NB_CLASSES = '5'
    show_plot = False
    if "-show-plot" in sys.argv:
        show_plot = True
    if len(sys.argv) > 1 and isinstance(eval(sys.argv[1]), int):
        NB_CLASSES = sys.argv[1]

    TEST_EMODB = False

    if TEST_EMODB:
        train_npy = "full_aibo.npy"
        test_npy = "emo-db.npy"
        TEST_DIR = EMODB_DIR
    else:
        train_npy = "train_aibo-o.npy"
        test_npy = "test_aibo-m.npy"

    # Load train data
    if not os.path.exists(train_npy):
        print("Reading train data...", end='', flush=True)
        train = list()
        for file in sorted(os.listdir(TRAIN_DIR)):
            rd = ArffReader(os.path.join(TRAIN_DIR, file))
            train.append(rd.data)
        train = np.array(train)
        print("Done!")
        np.save(train_npy, train)
        print("Train dataset saved.")
    else:
        print("Loading train dataset...", end='', flush=True)
        train = np.load(train_npy)
        print("Done!")

    min_max_scaler = preprocessing.MinMaxScaler()
    train = min_max_scaler.fit_transform(train)

    x_train = np.array(train)

    # Load test data
    if not os.path.exists(test_npy):
        print("Reading test data...", end='', flush=True)
        test = list()
        for file in sorted(os.listdir(TEST_DIR)):
            rd = ArffReader(os.path.join(TEST_DIR, file))
            test.append(rd.data)
        test = np.array(test)
        print("Done!")
        np.save(test_npy, test)
        print("Test dataset saved.")
    else:
        print("Loading test dataset...", end='', flush=True)
        test = np.load(test_npy)
        print("Done!")

    test = min_max_scaler.fit_transform(test)

    x_test = np.array(test)

    # Get labels
    if TEST_EMODB:
        train_labels = pd.read_csv(
            DATA_DIR + "chunk_labels_" + NB_CLASSES + "cl_corpus.txt",
            sep=' ', header=None, names=['id', 'label', 'score'])
        train_labels = train_labels['label']

        if NB_CLASSES == '2':
            idl_labels = ['F', 'N']  #, 'L']
            test_labels = ['IDL' if file[5] in idl_labels else 'NEG' for file in sorted(os.listdir(TEST_DIR))]
        else:
            dict_labels = {
                'W': 'A',
                'N': 'N',
                'F': 'P',
                'L': 'N',
                'A': 'R',
                'T': 'N',  # Tristesse : Reste ou Neutre ?
                'E': 'R',
            }
            test_labels = list()
            for file in sorted(os.listdir(TEST_DIR)):
                test_labels.append(dict_labels[file[5]])
        list_labels = test_labels
    else:
        list_labels = pd.read_csv(
            DATA_DIR + "chunk_labels_" + NB_CLASSES + "cl_corpus.txt",
            sep=' ', header=None, names=['id', 'label', 'score'])

        train_labels = [l for i, l in enumerate(
            list_labels['label']) if list_labels['id'][i].startswith('O')]
        test_labels = [l for i, l in enumerate(
            list_labels['label']) if list_labels['id'][i].startswith('M')]
        list_labels = list(list_labels['label'])

    class_names = list(set(list_labels))

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # Feature selection
    """
    print(x_train.shape)
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(x_train, y_train)
    x_train = selector.transform(x_train)
    print(x_train.shape)

    x_test = selector.transform(x_test)
    """
    """
    ch2 = SelectKBest(chi2)
    x_train = ch2.fit_transform(x_train, y_train)
    x_test = ch2.transform(x_test)
    """
    # Train model
    clf_name = 'svc_' + NB_CLASSES + 'cl_0.pkl'
    if not os.path.exists(CLASSIFIERS + clf_name):
        print("Learning model...", end='', flush=True)
        # classifier = svm.LinearSVC(C=0.01, penalty="l1", dual=False)
        classifier = svm.SVC(kernel='linear')
        classifier.fit(x_train, y_train)
        print("Done!")
        joblib.dump(classifier, CLASSIFIERS + clf_name)
        print("Model saved.")
    else:
        print("Loading model...", end='', flush=True)
        classifier = joblib.load(CLASSIFIERS + clf_name)
        print("Done!")

    """
    hmm_model = hmm.GaussianHMM(n_components=1, covariance_type="diag",
                                init_params="cm", params="cmt",
                                algorithm='viterbi')
    hmm_model.fit(x_train)
    hidden_states = hmm_model.predict(x_test)
    """

    # Predict and print metrics
    print("Making predictions...", end='', flush=True)
    y_pred = classifier.predict(x_test)
    print("Done!\n")

    print("Macro Recall (UA):", metrics.recall_score(
        y_test, y_pred, labels=class_names, average='macro'))
    print("Macro Precision (UA):", metrics.precision_score(
        y_test, y_pred, labels=class_names, average='macro'))
    print("Recall (WA):", metrics.recall_score(
        y_test, y_pred, labels=class_names, average='weighted'))
    print("Precision (WA):", metrics.precision_score(
        y_test, y_pred, labels=class_names, average='weighted'))

    print("Micro Precision/Recall:", metrics.recall_score(
        y_test, y_pred, labels=class_names, average='micro'))

    print("==============")
    print(metrics.classification_report(y_test, y_pred))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=class_names)
    np.set_printoptions(precision=2)

    if show_plot:
        # Plot confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix')
        plt.show()


if __name__ == '__main__':
    main()
