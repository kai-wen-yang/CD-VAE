"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import numpy as np
import os
import lib_regression
import argparse
import pdb
from sklearn.linear_model import LogisticRegressionCV

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
args = parser.parse_args()
print(args)

def main():
    # initial setup
    dataset_list = ['cifar10']
    adv_test_list = ['BIM', 'FGSM',  'PGD', 'PGD-L2', 'CWL2']
    print('evaluate the Mahalanobis estimator')
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', \
                  'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']

    list_best_results_ours, list_best_results_index_ours = [], []
    for dataset in dataset_list:
        print('load train data: ', dataset)
        outf = args.outf + args.net_type + '_' + dataset + '/'
        list_best_results_out, list_best_results_index_out = [], []
        for out in adv_test_list:
            best_auroc, best_result, best_index = 0, 0, 0
            for score in score_list:
                print('load train data: ', out, ' of ', score)
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                X_val, Y_val, X_test, Y_test = lib_regression.block_split_adv(total_X, total_Y)
                pivot = int(X_val.shape[0] / 6)
                X_train = np.concatenate((X_val[:pivot], X_val[2*pivot:3*pivot], X_val[4*pivot:5*pivot]))
                Y_train = np.concatenate((Y_val[:pivot], Y_val[2*pivot:3*pivot], Y_val[4*pivot:5*pivot]))
                X_val_for_test = np.concatenate((X_val[pivot:2*pivot], X_val[3*pivot:4*pivot], X_val[5*pivot:]))
                Y_val_for_test = np.concatenate((Y_val[pivot:2*pivot], Y_val[3*pivot:4*pivot], Y_val[5*pivot:]))
                lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
                y_pred = lr.predict_proba(X_train)[:, 1]
                #print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
                y_pred = lr.predict_proba(X_val_for_test)[:, 1]
                #print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))
                results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
                if best_auroc < results['TMP']['AUROC']:
                    best_auroc = results['TMP']['AUROC']
                    best_index = score
                    best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        list_best_results_ours.append(list_best_results_out)
        list_best_results_index_ours.append(list_best_results_index_out)

    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print("results of Mahalanobis")
    for in_list in list_best_results_ours:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ adv_test_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index_ours[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1


if __name__ == '__main__':
    main()
