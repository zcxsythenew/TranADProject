import pickle
import numpy
import os
import sklearn
import sklearn.metrics
import torch
import pandas


def get_tpr_from_fpr(fprg: numpy.ndarray, tprg: numpy.ndarray, fpr: float) -> float:
    assert(fprg.shape == tprg.shape and len(fprg.shape) == 1)
    fpr_min = 1.0
    tpr_min = -1.0
    fpr_max = 0.0
    tpr_max = -1.0
    for i in range(len(fprg)):
        if fpr_min > fprg[i] and fprg[i] >= fpr:
            fpr_min = fprg[i]
            tpr_min = tprg[i]
        if fpr_max < fprg[i] and fprg[i] <= fpr:
            fpr_max = fprg[i]
            tpr_max = tprg[i]
    if tpr_min == -1.0:
        return tpr_max
    if tpr_max == -1.0:
        return tpr_min
    return (tpr_min + tpr_max) / 2


def main() -> None:
    folder = 'results/no-node2vec'

    for i in range(5):
        path = os.path.join(folder, f'words_{i+1}.pkl')
        with open(path, 'rb') as f:
            words = pickle.load(f)
            fprg: numpy.ndarray = words['fpr']
            tprg: numpy.ndarray = words['tpr']
            arr = []
            for fpr in range(0, 1000, 1):
                arr.append([fpr / 1000., get_tpr_from_fpr(fprg, tprg, fpr / 1000.)])
            arr = numpy.array(arr)
            data_frame = pandas.DataFrame(arr)
            with open(os.path.join(folder, f'words_{i+1}.csv'), 'w') as out:
                data_frame.to_csv(out)


if __name__ == '__main__':
    main()
