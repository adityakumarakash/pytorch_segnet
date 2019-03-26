import numpy as np

class RunningScore(object):
    """Keeps track of the IoU scores for predictions"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def fast_hist(self, label_true, label_pred, num_class):
        mask = (label_true >= 0) & (label_true < num_class)
        hist = np.bincount(num_class * label_true[mask].astype(int) + label_pred[mask],
                           minlength=num_class ** 2).reshape(num_class, num_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self.fast_hist(lt.flatten(), lp.flatten(), self.num_classes)

    def get_scores(self):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))

        disp_scores = "Overall Acc: \t{}\n".format(acc)
        disp_scores += "Mean Acc: \t{}\n".format(acc_cls)
        disp_scores += "FreqW Acc: \t{}\n".format(fwavacc)
        disp_scores += "Mean IoU: \t{}\n".format(mean_iu)
        
        for k, v in cls_iu.items():
            disp_scores += "class {}: {}\n".format(k, v)
        
        return mean_iu, disp_scores

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
