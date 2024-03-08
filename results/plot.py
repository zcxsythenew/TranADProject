import pickle
import numpy
import os
import matplotlib.pyplot
import sklearn
import sklearn.metrics
import torch


def main():
    folder = 'results/tran-ad-64-gru'
    roc = matplotlib.pyplot.figure(1)
    ax = roc.add_subplot()
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    for i in range(5):
        path = os.path.join(folder, f'words_{i}.pkl')
        with open(path, 'rb') as f:
            words = pickle.load(f)
            fpr: numpy.ndarray = words['fpr']
            tpr: numpy.ndarray = words['tpr']
            labels: numpy.ndarray = words['label']
            scores: numpy.ndarray = words['score']
            auc = sklearn.metrics.roc_auc_score(labels, scores)
            ax.plot(fpr, tpr, label=f'AUC={auc}')
    
    ax.legend(loc='lower right')
    matplotlib.pyplot.savefig(os.path.join(folder, 'ROC.jpg'))

    ax.clear()
    matplotlib.pyplot.clf()

    pr = matplotlib.pyplot.figure(2)
    ax = pr.add_subplot()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    
    for i in range(5):
        path = os.path.join(folder, f'words_{i}.pkl')
        with open(path, 'rb') as f:
            words = pickle.load(f)
            recall: numpy.ndarray = words['recall']
            precision: numpy.ndarray = words['precision']
            labels: numpy.ndarray = words['label']
            scores: numpy.ndarray = words['score']
            ap = sklearn.metrics.average_precision_score(labels, scores)
            ax.plot(recall, precision, label=f'AP={ap}')
    
    ax.legend(loc='upper right')
    matplotlib.pyplot.savefig(os.path.join(folder, 'PR.jpg'))


if __name__ == '__main__':
    main()
