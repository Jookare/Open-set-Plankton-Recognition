import torch
from pytorch_ood.detector import OpenMax as oodOpenMax
from torch import nn 
from tqdm import tqdm
import sys
sys.path.append('../')
from metrics import compute_metrics

class OpenMax(nn.Module):

    """
    Adapted from
    - Ge, ZongYuan, et al. "Generative openmax for multi-class open set 
      classification." arXiv preprint arXiv:1707.07418 (2017).
    Extended utility from pytorch-ood.
    """
    
    def __init__(self, model, train_loader, valid_loader, test_loader, device, tailsize=20, alpha=5):
        super(OpenMax, self).__init__()
        self.model = model
        self.model.eval()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = train_loader.dataset.num_classes
        self._calc_feature_vectors()

        print("Fitting weibull model...")
        self.openmax = oodOpenMax(None, tailsize=tailsize, alpha=alpha)
        self.openmax.fit_features(self.features_train, self.labels_train)

        self.valid_probs = torch.tensor(self.openmax._openmax.predict(self.features_valid.numpy())).roll(-1, dims=1)
        self.test_probs = torch.tensor(self.openmax._openmax.predict(self.features_test.numpy())).roll(-1, dims=1)

    def _calc_feature_vectors(self):
        def encode_set(loader):
            features, labels = [], []
            for batch, label in tqdm(loader):
                features.append(self.model(batch.to(self.device)).cpu())
                labels.append(label)
            return torch.concat(features, dim=0), torch.concat(labels, dim=0)
        
        with torch.no_grad():
            print("Encoding train set...")
            self.features_train, self.labels_train = encode_set(self.train_loader)

            print("Encoding valid set...")
            self.features_valid, self.labels_valid = encode_set(self.valid_loader)
            
            print("Encoding test set...")
            self.features_test, self.labels_test = encode_set(self.test_loader)
            
    def find_thresholds(self, quantile):
        uniq, _ = self.labels_train.unique().sort()
        self.thresholds = torch.empty(uniq.shape)
        quantile = quantile.to(torch.float64)

        indexes = self.labels_valid < self.num_classes
        known_valid = self.labels_valid[indexes]
        
        for lab in uniq:
            probs = self.valid_probs.clone()
            probs = probs[indexes]
            probs = probs[known_valid == lab, lab]
            self.thresholds[lab] = probs.quantile(q=quantile)

    def set_thresholds(self, threshold):
        uniq, _ = self.labels_train.unique().sort()
        self.thresholds = torch.empty(uniq.shape)
        threshold = threshold.to(torch.float64)
        
        # Set same threshold for each class
        for lab in uniq:
            self.thresholds[lab] = threshold        
    
    def classify(self, test = False, use_th=True):
        # Classify labels works with class specific thresholds
        if test:
            probs = self.test_probs.clone()
            labels = self.labels_test.cpu().numpy()
        else:
            probs = self.valid_probs.clone()
            labels = self.labels_valid.cpu().numpy()
        
        if use_th:
            for i in range(self.num_classes):
                class_probs = probs[:, i]
                class_threshold = self.thresholds[i]
                class_probs[class_probs < class_threshold] = 0
                
            probs = probs.argmax(dim=1).cpu()
        else:
            probs = probs.argmax(dim=1).cpu()
            
        # Return predictions and labels
        return probs.numpy(), labels
    
    
    def test_multiple_thresholds(self, min_th, max_th = 1.0, step=0.01, use_quantile=False):
        f1_os = []
        accuracy_os = []
        accuracy_known = []
        accuracy_known_th = []
        accuracy_unknown_th = []
        th_range = torch.arange(min_th, max_th, step)
        for th in th_range:
            if use_quantile:
                # If set the th value if considered as the quantile
                self.find_thresholds(th)
            else:
                self.set_thresholds(th)    
            
            preds_no_th, labels_test = self.classify(use_th = False)
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
