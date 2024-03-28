import pickle
import numpy
import os
import sklearn
import sklearn.metrics
import torch
import pandas


def main() -> None:
    folder = 'results/tran-ad-220-gru-no-adjust'

    for i in range(10):
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
