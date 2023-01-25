import pandas as pd
import os
import numpy as np
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from cycler import cycler

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

import torch.nn.functional as F

try:
    import torch
except ImportError:
    torch = None

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    fontsize = 90
    axis_fontsize = 40

    # cm = confusion_matrix(y_true, y_pred)

    cm_plot = plt.figure(figsize=(20,20))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'fontsize': fontsize})
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=axis_fontsize)
    plt.yticks(tick_marks, classes, fontsize=axis_fontsize)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    # plt.tight_layout()
    plt.ylabel('True label', fontsize=axis_fontsize)
    plt.xlabel('Predicted label', fontsize=axis_fontsize)

    return cm_plot

def plot_grouped_bars(args, y_true, y_pred, pred_classes, true_classes, title='Relational Bar Plots'):

    bar_plot = plt.figure()

    graph_dicts = {cls: [0]*len(true_classes) for cls in pred_classes}
    for i in range(len(y_pred)):
        pml_grade = y_true[i]
        tumor_grade = y_pred[i]

        graph_dicts[pred_classes[tumor_grade]][pml_grade] += 1 # basically increments the count of the pml grade by 1 if the specific pml grade is predicted as the particular tumor grade

    df = pd.DataFrame.from_dict(graph_dicts, orient='index', columns=true_classes) # PML GRADES as columns
    df['tumor_grade'] = df.index
    print(df)

    df.plot(x='tumor_grade',
        kind='bar',
        stacked=False,
        title='PCGA grade characterization')

    plt.xticks(rotation=0)

    plt.savefig('PCGA_Tumor_Relations_{}'.format(args.fold_idx))

    return bar_plot
    # return 0

### Evaluator for graph classification
class Evaluator:
    def __init__(self, dataset_class):
        self.dataset_class = dataset_class
        self.classdict = dataset_class.label_encoding

        self.eval_metrics = dataset_class.eval_metrics # list of metrics to be computed

    def _parse_and_check_input(self, input_dict):
        if 'acc' in self.eval_metrics or 'precision' in self.eval_metrics or 'recall' in self.eval_metrics or 'cm' in self.eval_metrics:
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()


            ## check type
            if not isinstance(y_true, np.ndarray):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            return y_true, y_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))


    def eval(self, input_dict):

        metric_scores = {}

        y_true, y_pred = self._parse_and_check_input(input_dict)
        for metric in self.eval_metrics:
            if metric == 'acc':
                metric_scores['acc'] = self._eval_acc(y_true, y_pred)['acc']
            elif metric == 'precision':
                metric_scores['precision'] = self._eval_precision(y_true, y_pred)['precision']
            elif metric == 'recall':
                metric_scores['recall'] = self._eval_recall(y_true, y_pred)['recall']
            elif metric == 'cm':
                cm_scores = self._eval_cm(y_true, y_pred)
                metric_scores['cm'] = cm_scores['cm']
                metric_scores['specificity'] = cm_scores['specificity']
                metric_scores['sensitivity'] = cm_scores['sensitivity']
            else:
                raise ValueError('Undefined eval metric %s ' % (metric))

        return metric_scores

    def plot_curves(self, plot_dict):

        curves = {}

        curves['roc'] = self._plot_roc(plot_dict)
        curves['pr'] = self._plot_pr(plot_dict)

        return curves

    def average_scores(self, score_dicts):
        scores = {}

        scores = self._avg_metrics(score_dicts)
        return scores

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc' or self.eval_metric == 'ap':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_graph, num_task)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_graph, num_task)\n'
            desc += 'where y_pred stores score values (for computing AUC score),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
            desc += 'nan values in y_true are ignored during evaluation.\n'
        elif self.eval_metric == 'rmse':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_graph, num_task)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_graph, num_task)\n'
            desc += 'where num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
            desc += 'nan values in y_true are ignored during evaluation.\n'
        elif self.eval_metric == 'acc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += 'where y_pred stores predicted class label (integer),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
        elif self.eval_metric == 'F1':
            desc += '{\'seq_ref\': seq_ref, \'seq_pred\': seq_pred}\n'
            desc += '- seq_ref: a list of lists of strings\n'
            desc += '- seq_pred: a list of lists of strings\n'
            desc += 'where seq_ref stores the reference sequences of sub-tokens, and\n'
            desc += 'seq_pred stores the predicted sequences of sub-tokens.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'ap':
            desc += '{\'ap\': ap}\n'
            desc += '- ap (float): Average Precision (AP) score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'rmse':
            desc += '{\'rmse\': rmse}\n'
            desc += '- rmse (float): root mean squared error averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'acc':
            desc += '{\'acc\': acc}\n'
            desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'F1':
            desc += '{\'F1\': F1}\n'
            desc += '- F1 (float): F1 score averaged over samples.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

        return desc

    def _plot_roc(self, plot_dicts):

        fontsize = 20
        axis_fontsize = 15

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        roc_plot, ax = plt.subplots(dpi=400.0)
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Chance", alpha=0.8)
        num_classes = len(self.classdict)

        colors = ['r', 'g', 'b']
        for class_idx in range(num_classes):
            for fold_idx, p_dict in enumerate(plot_dicts):
                y_true = p_dict['y_true']
                y_prob = p_dict['y_prob'][:, class_idx]
                fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=class_idx)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(auc(fpr, tpr))

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)

            ax.plot(mean_fpr,
                    mean_tpr,
                    label="%s ROC(AUC = %0.2f $\pm$ %0.2f)" % (list(self.classdict.keys())[class_idx], mean_auc, std_auc),
                    color=colors[class_idx],
                    lw=2,
                    alpha=0.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color=colors[class_idx],
                alpha=0.2,
                label="$\pm$ 1 std. dev.")

        plot_name = self.dataset_class.__class__.__name__

        # ax.set_title(plot_name, fontdict={'fontsize': fontsize})
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_xlabel('False Positive Rate', fontdict={'fontsize': axis_fontsize, 'fontweight': 'bold'})
        ax.set_ylabel('True Positive Rate', fontdict={'fontsize': axis_fontsize, 'fontweight': 'bold'})
        ax.legend(loc="lower right")

        return roc_plot

    def _plot_pr(self, plot_dict):

        fontsize = 20
        axis_fontsize = 15

        prs = []
        aucs = []
        mean_recall = np.linspace(0, 1, 100)

        pr_plot, ax = plt.subplots(dpi=400.0)
        num_classes = len(self.classdict)

        colors = ['r', 'g', 'b']
        for class_idx in range(num_classes):
            for fold_idx, p_dict in enumerate(plot_dict):
                y_true = p_dict['y_true']
                y_prob = p_dict['y_prob'][:, class_idx]
                precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=class_idx)
                interp_pr = np.interp(mean_recall, precision, recall) # could be reverse
                interp_pr[0] = 1.0
                prs.append(interp_pr)
                aucs.append(auc(recall, precision))

            mean_precision = np.mean(prs, axis=0)
            mean_precision[-1] = 0.0
            mean_auc = auc(mean_recall, mean_precision)
            std_auc = np.std(aucs)

            ax.plot(mean_precision,
                    mean_recall,
                    color=colors[class_idx],
                    label="%s PR(AUC = %0.2f $\pm$ %0.2f)" % (list(self.classdict.keys())[class_idx], mean_auc, std_auc),
                    lw=2,
                    alpha=0.8)

            std_precision = np.std(prs, axis=0)
            precision_upper = np.minimum(mean_precision + std_precision, 1)
            precision_lower = np.maximum(mean_precision - std_precision, 0)
            ax.fill_between(
                mean_recall,
                precision_lower,
                precision_upper,
                color=colors[class_idx],
                alpha=0.2,
                label="$\pm$ 1 std. dev.")

        plot_name = self.dataset_class.__class__.__name__

        # ax.set_title(plot_name, fontdict={'fontsize': fontsize})
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_xlabel('False Positive Rate', fontdict={'fontsize': axis_fontsize, 'fontweight': 'bold'})
        ax.set_ylabel('True Positive Rate', fontdict={'fontsize': axis_fontsize, 'fontweight': 'bold'})
        ax.legend(loc="lower left")

        return pr_plot

    def _avg_metrics(self, score_dicts):

        num_classes = len(self.classdict)
        num_folds = len(score_dicts)

        avg_metrics = {class_name:{} for class_name in self.classdict.keys()}

        for (class_name, class_idx) in self.classdict.items():

            precisions = []
            recalls = []
            specificities = []
            for fold_idx, score_dict in enumerate(score_dicts):

                precisions.append(score_dict['precision'][class_idx])
                recalls.append(score_dict['recall'][class_idx])
                specificities.append(score_dict['specificity'][class_idx])

            avg_metrics[class_name]['precision'] = (np.mean(precisions), np.std(precisions))
            avg_metrics[class_name]['recall'] = (np.mean(recalls), np.std(recalls))
            avg_metrics[class_name]['specificity'] = (np.mean(specificities), np.std(specificities))

        return avg_metrics





    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list)/len(rocauc_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}

    def _eval_precision(self, y_true, y_pred):
        labels = np.array(list(self.classdict.values()))
        p_scores = precision_score(y_true, y_pred, labels=labels, average=None)

        return {'precision': p_scores}

    def _eval_recall(self, y_true, y_pred):
        labels = np.array(list(self.classdict.values()))
        r_scores = recall_score(y_true, y_pred, labels=labels, average=None)

        return {'recall': r_scores}

    def _eval_cm(self, y_true, y_pred):
        labels = np.array(list(self.classdict.values()))
        cm_scores = confusion_matrix(y_true, y_pred, labels=labels)

        FP = cm_scores.sum(axis=0) - np.diag(cm_scores)
        FN = cm_scores.sum(axis=1) - np.diag(cm_scores)
        TP = np.diag(cm_scores)
        TN = cm_scores.sum() - (FP + FN + TP)

        specificity = TN/(TN+FP)
        sensitivity = TP/(TP+FN)


        return {'cm': cm_scores, 'specificity': specificity, 'sensitivity': sensitivity}