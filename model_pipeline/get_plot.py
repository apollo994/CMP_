#!/usr/bin/env python3
#Fabio Zanarello, Sanger Institute, 2020

import sys
import argparse

import matplotlib.pyplot as plt
import pickle

import numpy as np

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, f1_score, auc


# def get_mean(l):
#     arrays = [np.array(x) for x in l]
#     res = [np.mean(k) for k in zip(*arrays)]
#
#     return res


def main():
    parser = argparse.ArgumentParser(description='My nice tool.')
    parser.add_argument('--exp', metavar='EXP', help='name of the experiment')
    parser.add_argument('--pos', metavar='POS', type=int, default=1 , help='positive label (0-1)')


    args = parser.parse_args()




    ############################################################################
    #open real and preds


    with open(f'{args.exp}/results/{args.exp}_real.pk', 'rb') as real_pi:
                reals = pickle.load(real_pi)

    with open(f'{args.exp}/results/{args.exp}_pred_p.pk', 'rb') as pred_p_pi:
                preds_p = pickle.load(pred_p_pi)

    with open(f'{args.exp}/results/{args.exp}_pred_c.pk', 'rb') as pred_c_pi:
                preds_c = pickle.load(pred_c_pi)




    # print (reals[0])
    # print (preds_p[0])

    #NO SKILLS PREDICTOR
    ns_probs = [0 for _ in range(len(reals[0]))]
    ns_auc = roc_auc_score(reals[0], ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(reals[0], ns_probs)
    no_skill_ratio = round(len(reals[0][reals[0]==args.pos]) / len(reals[0]),2)



    ############################################################################
    #prediction plots

    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,5), constrained_layout=True,)

    _ = fig.suptitle(f'{args.exp} predicions result')

    _ = ax1.set_title('ROC curve')
    _ = ax1.set_xlabel('False Positive Rate')
    _ = ax1.set_ylabel('True Positive Rate')
    _ = ax1.plot(ns_fpr, ns_tpr, label = f'No Skill = 0.5', linestyle='--')



    _ = ax2.set_title('Precision-recall')
    _ = ax2.set_xlabel('Recall')
    _ = ax2.set_ylabel('Precision')
    _ = ax2.plot([0, 1], [no_skill_ratio, no_skill_ratio], linestyle='--', label= f'No Skill = {no_skill_ratio}')

    ############################################################################

    mean_fpr = []
    mean_tpr = []
    mean_precision_th = []
    mean_recall_th = []

    #initialise summary table

    with open(f'{args.exp}/results/{args.exp}_metrics.csv', 'w') as metrics:
        metrics.write('sample,prec,rec,auROC,auPR\n')

        f = 0

        for r, p_c, p_p in zip (reals, preds_c, preds_p):

            f += 1

            #prediction metrics
            precision = precision_score(r, p_c)
            recall = recall_score(r, p_c)

            fpr_th, tpr_th, _ = roc_curve(r, p_p, pos_label=args.pos)
            AUC_ROC = round(auc(fpr_th, tpr_th),3)
            # mean_fpr.append(list(fpr_th))
            # mean_tpr.append(list(tpr_th))

            precision_th, recall_th, _ = precision_recall_curve(r, p_p, pos_label=args.pos)
            PR_AUC = round(auc(recall_th, precision_th),3)
            # mean_precision_th.append(list(precision_th))
            # mean_recall_th.append(list(recall_th))

            metrics.write(f'fold_{f},{precision},{recall},{AUC_ROC},{PR_AUC}\n')

            _ = ax1.plot(fpr_th, tpr_th, label=f'fold_{f} = {AUC_ROC}', alpha=0.3)
            _ = ax2.plot(recall_th, precision_th, label=f'fold_{f} = {PR_AUC}', alpha=0.3)

    # mean_fpr = get_mean(mean_fpr)
    # mean_tpr = get_mean(mean_tpr)
    # mean_precision_th = get_mean(mean_precision_th)
    # mean_recall_th = get_mean(mean_recall_th)
    #
    # mean_AUC_ROC = round(auc(mean_fpr, mean_tpr),3)
    # mean_PR_AUC = round(auc(mean_recall_th, mean_precision_th),3)
    #
    # _ = ax1.plot(mean_fpr, mean_tpr, label=f'mean = {mean_AUC_ROC}', linewidth=3)
    # _ = ax2.plot(mean_recall_th, mean_precision_th, label=f'mean = {mean_PR_AUC}', linewidth=3)

    all_real = []
    all_pred = []
    all_labe = []

    for re,pre in zip(reals, preds_p):
        all_real = all_real+list(re)
        all_pred = all_pred+list(pre)

    for lab in preds_c:
        all_labe = all_labe+list(lab)

    precision = precision_score(all_real, all_labe)
    recall = recall_score(all_real, all_labe)

    all_fpr_th, all_tpr_th, _ = roc_curve(all_real, all_pred, pos_label=args.pos)
    all_AUC_ROC = round(auc(all_fpr_th, all_tpr_th),3)

    all_precision_th, all_recall_th, _ = precision_recall_curve(all_real, all_pred, pos_label=args.pos)
    all_PR_AUC = round(auc(all_recall_th, all_precision_th),3)

    _ = ax1.plot(all_fpr_th, all_tpr_th, label=f'all = {all_AUC_ROC}', linewidth=3)
    _ = ax2.plot(all_recall_th, all_precision_th, label=f'all = {all_PR_AUC}', linewidth=3)

    with open(f'{args.exp}/results/{args.exp}_metrics.csv', 'a') as metrics:
        metrics.write(f'all,{precision},{recall},{all_AUC_ROC},{all_PR_AUC}\n')


    _ = ax1.legend()
    _ = ax2.legend()
    plt.savefig(f'{args.exp}/results/{args.exp}_predictions_result.png')



if __name__ == "__main__":
  main()
