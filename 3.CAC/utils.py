import torch
import torch.nn as nn
from torchmetrics.functional import pairwise_euclidean_distance
import torch.nn.functional as F
from tqdm import tqdm
import sys

sys.path.append("../")
from metrics import compute_metrics


class CACLoss(nn.Module):
    """
    Adapted from:
        - Miller, Dimity, et al. "Class anchor clustering: A loss for
          distance-based open set recognition." Proceedings of the IEEE/CVF
          Winter Conference on Applications of Computer Vision. 2021.
        - https://github.com/dimitymiller/cac-openset

    Extended with open-set classification.
    """

    def __init__(self, num_classes, magnitude=10.0, anchor_scale=0.1, device="cpu"):
        super(CACLoss, self).__init__()
        self.num_classes = num_classes
        self.anchor_scale = anchor_scale
        self.centres = magnitude * torch.eye(num_classes, device=device)

    def forward(self, input, label):
        dists = pairwise_euclidean_distance(input, self.centres)
        anchor = torch.gather(dists, 1, label.view(-1, 1))
        tuplet = (anchor - dists).exp().sum(dim=1).log()
        loss = tuplet.mean() + self.anchor_scale * anchor.mean()
        return loss, anchor.mean()


class CACClassif:
    """
    Follows the implementation of the ArcFace 'CosineClassif' class.
    """

    def __init__(self, model, train_loader, valid_loader, test_loader, num_classes, device):
        super(CACClassif, self).__init__()
        self.model = model
        self.model.eval()

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.centres = torch.eye(self.num_classes)

        self.features_train, self.features_valid, self.features_test = None, None, None
        self.labels_train, self.labels_valid, self.labels_test = None, None, None
        self._find_features()
        self._update_centers()

    def _find_features(self):
        def encode_set(loader):
            features, labels = [], []
            for batch, label in tqdm(loader):
                features.append(self.model(batch.to(self.device)).detach().cpu())
                labels.append(label)
            return torch.concat(features, dim=0), torch.concat(labels, dim=0)

        with torch.no_grad():
            print("Encoding train set...")
            self.features_train, self.labels_train = encode_set(self.train_loader)

            print("Encoding validation set...")
            self.features_valid, self.labels_valid = encode_set(self.valid_loader)

            print("Encoding test set...")
            self.features_test, self.labels_test = encode_set(self.test_loader)

    def _update_centers(self):
        new_centres = torch.zeros_like(self.centres)
        n_per_label = self.labels_train.unique(return_counts=True)

        for features, lab in zip(self.features_train, self.labels_train):
            new_centres[lab] += features

        for lab, n in zip(*n_per_label):
            new_centres[lab] /= n
        self.centres = new_centres

    def validate_thresholds(self, quantile=0.95):
        dists = pairwise_euclidean_distance(self.features_valid, self.centres)
        uniq, _ = self.labels_valid.unique().sort()
        gammas = dists * (1 - F.softmin(dists, dim=1))
        gamma_true = torch.gather(gammas, 1, self.labels_valid.view(-1, 1)).flatten()
        self.thresholds = torch.tensor(
            [torch.quantile(gamma_true[self.labels_valid == lab], quantile) for lab in uniq]
        )

    def classify(self, use_th=True):
        if self.thresholds == None:
            raise Exception("Thresholds are not evaluated. Run validate_thresholds")
        dists = pairwise_euclidean_distance(self.features_test, self.centres)
        gammas = dists * (1 - F.softmin(dists, dim=1))
        pred = gammas.argmin(axis=1)
        # Find the thresholds and add the values at the end of the scores.
        if use_th:
            th_expanded = self.thresholds[pred]
            scores = torch.concat((gammas, th_expanded.view(-1, 1)), dim=1)
            pred = scores.argmin(dim=1)
        return pred.cpu().numpy(), self.labels_test.cpu().numpy()

    def multiple_quantile_classify(self, min_q, max_q=1.0):
        f1_os = []
        accuracy_os = []
        accuracy_known = []
        accuracy_known_th = []
        accuracy_unknown_th = []
        q_range = torch.arange(min_q, max_q, 0.01)
        for q in q_range:
            self.validate_thresholds(quantile=q)
            preds_no_th, labels_test = self.classify(use_th=False)
            preds, labels_test = self.classify(use_th=True)

            result = compute_metrics(labels_test, preds_no_th, preds, self.num_classes)

            accuracy_known += [round(result["acc_known"], 4)]
            accuracy_known_th += [round(result["acc_known_th"], 4)]
            accuracy_unknown_th += [round(result["acc_unk_th"], 4)]
            accuracy_os += [round(result["acc_os"], 4)]
            f1_os += [round(result["f1_os"], 4)]

        # print(f'"Overall best quantile": {q_range[torch.tensor(f1_os).argmax(dim=0)].round(decimals=2)}')
        # print(f'"quantile": \t{([round(el.item(), 2) for el in q_range])},')
        # print(f'"known_class_acc": \t\t{accuracy_known},')
        # print(f'"known_class_acc_th": \t{accuracy_known_th},')
        # print(f'"unknown_class_acc_th": {accuracy_unknown_th},')
        # print(f'"open_set_acc": \t\t{accuracy_os},')
        # print(f'"open_set_f_score": \t{f1_os}')

        result = {}
        result["quantile"] = [round(el.item(), 2) for el in q_range]
        result["known_class_acc"] = accuracy_known
        result["known_class_acc_th"] = accuracy_known_th
        result["unknown_class_acc_th"] = accuracy_unknown_th
        result["open_set_acc"] = accuracy_os
        result["open_set_f_score"] = f1_os
        return result
