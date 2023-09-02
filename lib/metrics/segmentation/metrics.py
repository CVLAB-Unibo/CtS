from collections import defaultdict, deque

import numpy as np
import torch


def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meters(new_metric_list)
    return metric_logger


class MetricLogger(object):
    """Metric logger.
    All the meters should implement following methods:
        __str__, summary_str, reset
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                count = v.numel()
                value = v.item() if count == 1 else v.sum().item()
            elif isinstance(v, np.ndarray):
                count = v.size
                value = v.item() if count == 1 else v.sum().item()
            else:
                assert isinstance(v, (float, int))
                value = v
                count = 1
            self.meters[k].update(value, count)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def add_meters(self, meters):
        if not isinstance(meters, (list, tuple)):
            meters = [meters]
        for meter in meters:
            self.add_meter(meter.name, meter)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return getattr(self, attr)

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append("{}: {}".format(name, meter.summary_str))
        return self.delimiter.join(metric_str)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()


class AverageMeter(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    default_fmt = "{avg:.4f} ({global_avg:.4f})"
    default_summary_fmt = "{global_avg:.4f}"

    def __init__(self, window_size=20, fmt=None, summary_fmt=None):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0
        self.fmt = fmt or self.default_fmt
        self.summary_fmt = summary_fmt or self.default_summary_fmt

    def update(self, value, count=1):
        self.values.append(value)
        self.counts.append(count)
        self.sum += value
        self.count += count

    @property
    def avg(self):
        return np.sum(self.values) / np.sum(self.counts)

    @property
    def global_avg(self):
        return self.sum / self.count if self.count != 0 else float("nan")

    def reset(self):
        self.values.clear()
        self.counts.clear()
        self.sum = 0.0
        self.count = 0

    def __str__(self):
        return self.fmt.format(avg=self.avg, global_avg=self.global_avg)

    @property
    def summary_str(self):
        return self.summary_fmt.format(global_avg=self.global_avg)


class SegAccuracy(AverageMeter):
    """Segmentation accuracy"""

    name = "seg_acc"

    def __init__(self, ignore_index=-100):
        super(SegAccuracy, self).__init__()
        self.ignore_index = ignore_index

    def update_dict(self, preds, labels):
        seg_logit = preds["seg_logit"]  # (b, c, n)
        seg_label = labels["seg_label"]  # (b, n)
        pred_label = seg_logit.argmax(1)

        mask = seg_label != self.ignore_index
        seg_label = seg_label[mask]
        pred_label = pred_label[mask]

        tp_mask = pred_label.eq(seg_label)  # (b, n)
        self.update(tp_mask.sum().item(), tp_mask.numel())


class SegIoU(object):
    """Segmentation IoU
    References: https://github.com/pytorch/vision/blob/master/references/segmentation/utils.py
    """

    def __init__(self, num_classes, ignore_index=-100, name="seg_iou"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = None
        self.name = name

    def update_dict(self, preds, labels):
        seg_logit = preds["seg_logit"]  # (batch_size, num_classes, num_points)
        seg_label = labels["seg_label"]  # (batch_size, num_points)
        pred_label = seg_logit.argmax(1)

        mask = seg_label != self.ignore_index
        seg_label = seg_label[mask]
        pred_label = pred_label[mask]

        # Update confusion matrix
        # TODO: Compare the speed between torch.histogram and torch.bincount after pytorch v1.1.0
        n = self.num_classes
        with torch.no_grad():
            if self.mat is None:
                self.mat = seg_label.new_zeros((n, n))
            inds = n * seg_label + pred_label
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat = None

    @property
    def iou(self):
        h = self.mat.float()
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return iou

    @property
    def global_avg(self):
        return self.iou.mean().item()

    @property
    def avg(self):
        return self.global_avg

    def __str__(self):
        return "{iou:.4f}".format(iou=self.iou.mean().item())

    @property
    def summary_str(self):
        return str(self)


import numpy as np
from sklearn.metrics import confusion_matrix as CM


class Evaluator(object):
    def __init__(self, class_names, labels=None):
        self.class_names = tuple(class_names)
        self.num_classes = len(class_names)
        self.labels = np.arange(self.num_classes) if labels is None else np.array(labels)
        assert self.labels.shape[0] == self.num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred_label, gt_label):
        """Update per instance

        Args:
            pred_label (np.ndarray): (num_points)
            gt_label (np.ndarray): (num_points,)

        """
        # convert ignore_label to num_classes
        # refer to sklearn.metrics.confusion_matrix
        gt_label[gt_label == -100] = self.num_classes
        confusion_matrix = CM(gt_label.flatten(), pred_label.flatten(), labels=self.labels)
        self.confusion_matrix += confusion_matrix

    def batch_update(self, pred_labels, gt_labels):
        assert len(pred_labels) == len(gt_labels)
        for pred_label, gt_label in zip(pred_labels, gt_labels):
            self.update(pred_label, gt_label)

    @property
    def overall_acc(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    @property
    def overall_iou(self):
        class_iou = np.array(self.class_iou.copy())
        class_iou[np.isnan(class_iou)] = 0
        return np.mean(class_iou)

    @property
    def class_seg_acc(self):
        return [
            self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i])
            for i in range(self.num_classes)
        ]

    @property
    def class_iou(self):
        iou_list = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            p = self.confusion_matrix[:, i].sum()
            g = self.confusion_matrix[i, :].sum()
            union = p + g - tp
            if union == 0:
                iou = float("nan")
            else:
                iou = tp / union
            iou_list.append(iou)
        return iou_list

    def print_table(self):
        from tabulate import tabulate

        header = ["Class", "Accuracy", "IOU", "Total"]
        seg_acc_per_class = self.class_seg_acc
        iou_per_class = self.class_iou
        table = []
        for ind, class_name in enumerate(self.class_names):
            table.append(
                [
                    class_name,
                    seg_acc_per_class[ind] * 100,
                    iou_per_class[ind] * 100,
                    int(self.confusion_matrix[ind].sum()),
                ]
            )
        return tabulate(table, headers=header, tablefmt="psql", floatfmt=".2f")

    def save_table(self, filename):
        from tabulate import tabulate

        header = ("overall acc", "overall iou") + self.class_names
        table = [[self.overall_acc, self.overall_iou] + self.class_iou]
        with open(filename, "w") as f:
            # In order to unify format, remove all the alignments.
            f.write(
                tabulate(
                    table,
                    headers=header,
                    tablefmt="tsv",
                    floatfmt=".5f",
                    numalign=None,
                    stralign=None,
                )
            )
