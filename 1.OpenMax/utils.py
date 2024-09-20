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
    
    def __init__(self, model, train_loader, test_loader, device, tailsize=20, alpha=5):
        super(OpenMax, self).__init__()
        self.model = model
        self.model.eval()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = train_loader.dataset.num_classes
        self._calc_feature_vectors()

        print("Fitting weibull model...")
        self.openmax = oodOpenMax(None, tailsize=tailsize, alpha=alpha)
        self.openmax.fit_features(self.features_train, self.labels_train)

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
            
            print("Encoding test set...")
            self.features_test, self.labels_test = encode_set(self.test_loader)
            
    def classify(self, threshold, use_th=True):
        # Ensure device compatibility
        if not torch.is_tensor(threshold):
            threshold = torch.tensor(threshold)

        logits = self.features_test.numpy()
        # Prediction and threshold adjustment
        probs = torch.tensor(self.openmax._openmax.predict(logits))
        if use_th:
            probs[:, 0] = torch.maximum(probs[:, 0], threshold)
            probs = probs.roll(-1, dims=1).argmax(dim=1).cpu()
        else:
            probs = probs.roll(-1, dims=1).argmax(dim=1).cpu()
            
        # Return predictions and labels
        return probs.numpy(), self.labels_test.numpy()
      
    def multiple_quantile_classify(self, min_q, max_q = 1.0):
        f1_os = []
        accuracy_os = []
        accuracy_known = []
        accuracy_known_th = []
        accuracy_unknown_th = []
        q_range = torch.arange(min_q, max_q, 0.01)
        for th in q_range:
            preds_no_th, labels_test = self.classify(threshold=th, use_th = False)
            preds, labels_test = self.classify(threshold=th)
        
            # Compute metrics
            result = compute_metrics(labels_test, preds_no_th, preds, self.num_classes)
        
            accuracy_known += [round(result["acc_known"], 4)]
            accuracy_known_th += [round(result["acc_known_th"], 4)]
            accuracy_unknown_th += [round(result["acc_unk_th"], 4)]
            accuracy_os += [round(result["acc_os"], 4)]
            f1_os += [round(result["f1_os"], 4)]
        
        result = {}
        result["quantile"] = [round(el.item(), 2) for el in q_range]
        result["known_class_acc"] = accuracy_known
        result["known_class_acc_th"] = accuracy_known_th
        result["unknown_class_acc_th"] = accuracy_unknown_th
        result["open_set_acc"] = accuracy_os
        result["open_set_f_score"] = f1_os
        return result
