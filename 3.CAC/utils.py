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
        
        dists = pairwise_euclidean_distance(self.features_test, self.centres)
        self.gammas_test = dists * (1 - F.softmin(dists, dim=1))

        dists = pairwise_euclidean_distance(self.features_valid, self.centres)
        self.gammas_valid = dists * (1 - F.softmin(dists, dim=1))

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

    def find_thresholds(self, quantile=0.95):
        dists = pairwise_euclidean_distance(self.features_valid, self.centres)
        uniq, _ = self.labels_train.unique().sort()
        gammas = dists * (1 - F.softmin(dists, dim=1))

        indexes = self.labels_valid < self.num_classes
        known_valid = self.labels_valid[indexes]
        
        gamma_true = torch.gather(gammas, 1, known_valid.view(-1, 1)).flatten()
    
        self.thresholds = torch.tensor(
            [torch.quantile(gamma_true[known_valid == lab], quantile) for lab in uniq]
        )
        
    def set_thresholds(self, threshold):
        uniq, _ = self.labels_train.unique().sort()
        self.thresholds = torch.empty(uniq.shape)
        threshold = threshold.to(torch.float64)
        
        for lab in uniq:
            self.thresholds[lab] = threshold

    def classify(self, test = False, use_th=True):
        if self.thresholds == None:
            raise Exception("Thresholds are not evaluated. Run validate_thresholds")

        if test:
            gammas = self.gammas_test
            labels = self.labels_test.cpu().numpy()
        else:
            gammas = self.gammas_valid
            labels = self.labels_valid.cpu().numpy()
        
        pred = gammas.argmin(axis=1)
        
        # Find the thresholds and add the values at the end of the scores.
        if use_th:
            th_expanded = self.thresholds[pred]
            scores = torch.concat((gammas, th_expanded.view(-1, 1)), dim=1)
            pred = scores.argmin(dim=1)
        
        return pred.cpu().numpy(), labels

    def multiple_quantile_classify(self, min_th, max_th=1.0, step=0.01, use_quantile=False):
        f1_os = []
        accuracy_os = []
        accuracy_known = []
        accuracy_known_th = []
        accuracy_unknown_th = []
        # Use custom range for plain threshold
        if use_quantile:  
            th_range = torch.arange(min_th, max_th, step)
        else:
            th_range = torch.cat((torch.arange(0.000001, 0.00001, 0.000001),
                                  torch.arange(0.00001, 0.0001, 0.00001),
                                  torch.arange(0.0001, 0.001, 0.0001),
                                  torch.arange(0.001, 0.01, 0.001) ))
        for th in th_range:
            if use_quantile:
                # If set the th value if considered as the quantile
                self.find_thresholds(th)
            else:
                self.set_thresholds(th)    
                
            preds_no_th, labels_test = self.classify(use_th=False)
            preds, labels_test = self.classify(use_th=True)

            # Compute metrics
            result = compute_metrics(labels_test, preds_no_th, preds, self.num_classes)
            
            accuracy_known += [round(result["acc_known"], 4)]
            accuracy_known_th += [round(result["acc_known_th"], 4)]
            accuracy_unknown_th += [round(result["acc_unk_th"], 4)]
            accuracy_os += [round(result["acc_os"], 4)]
            f1_os += [round(result["f1_os"], 4)]
        
        result = {}
        result["threshold"] = [round(el.item(), 2) for el in th_range]
        result["known_class_acc"] = accuracy_known
        result["known_class_acc_th"] = accuracy_known_th
        result["unknown_class_acc_th"] = accuracy_unknown_th
        result["open_set_acc"] = accuracy_os
        result["open_set_f_score"] = f1_os
        return result