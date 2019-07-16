from sklearn import metrics

def get_eval_metrics(y_pred, y_gold):
    labels = list(set(y_gold))
    labels.sort()
    scores = metrics.precision_recall_fscore_support(y_gold, y_pred, average=None, labels=labels)
    string = '>>> label specific details \n'
    total = len(y_gold)
    for index, label in enumerate(labels):
        prec, rec, f1 = scores[0][index], scores[1][index], scores[2][index]
        gold_count = y_gold.count(label)
        pred_count = y_pred.count(label)
        string += '>>> label: {} | gold num {}/{:.2f} | pred num {}/{:.2f} | prec {:.4f} | rec {:.4f} | f1 {:.4f} \n'.format(label,
                            gold_count, gold_count/total, pred_count, pred_count/total, prec, rec, f1)
    string += '-'*80
    return string