import math
import torch
import torch.nn as nn
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm import tqdm
import sys

sys.path.append("../")
from metrics import compute_metrics

class NetConfig(nn.Module):
    def __init__(self, backbone, size=512, num_classes=12, feature=False, scale=20.0, margin=0.50, easy_margin=False):
        super(NetConfig, self).__init__()
        self.num_classes = num_classes
        self.feature = feature
        self.backbone = backbone
        self.fc = ArcMarginProduct(size, num_classes, feature_scale=scale, additive_margin=margin, easy_margin=easy_margin)

    def forward(self, x, label = None):
        x = self.backbone(x)
        if self.feature:
            return x
        fc = self.fc(x, label)
        return fc
    

    def short_info(self):
        return f"Arc Net, num_classes={self.num_classes}, base net=\n{self.backbone.short_info()}"

class ArcMarginProduct(nn.Module):

    """
    Adapted from:
        - Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou; "ArcFace:
          Additive Angular Margin Loss for Deep Face Recognition"; Proceedings
          of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
          (CVPR), 2019, pp. 4690-4699 
        - https://github.com/ronghuaiyang/arcface-pytorch
    
    Implement of large margin arc distance.
    args:
        in_features: size of each input sample
        out_features: size of each output sample
    kwargs:
        feature_scale: (s) norm of input feature
        additive_margin: (m) margin
        easy_margin: bool

        cos(theta + m)
    """

    def __init__(self, in_features, out_features, **kwargs):
        super(ArcMarginProduct, self).__init__()
        
        self.s = kwargs.get('feature_scale', 2.39)
        self.m = kwargs.get('additive_margin', 0.95)
        self.easy_margin = kwargs.get('easy_margin', False)
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        # cos(theta) & phi(theta)
        cosine = nn.functional.linear(
            nn.functional.normalize(input),
            nn.functional.normalize(self.weight)
        )
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # torch.where(out_i = {x_i if condition_i else y_i)
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', s=' + str(self.s) \
            + ', m=' + str(self.m) + ')'



class CosineClassif():

    """
    Decision pipeline proposed in
     - Badreldeen Bdawy Mohamed, Ola, et al. "Open-Set Plankton Recognition Using Similarity Learning."
       International Symposium on Visual Computing. Cham: Springer International Publishing, 2022.
    """

    def __init__(self, model, gallery_loader, valid_loader, test_loader, device):
        super(CosineClassif, self).__init__()
        self.model = model
        self.model.eval()
        self.gallery_loader = gallery_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
                
        self.features_gallery, self.features_valid, self.features_test = None, None, None
        self.labels_gallery,   self.labels_valid,   self.labels_test   = None, None, None
        self.num_classes = self.gallery_loader.dataset.num_classes
        self._find_features()
        
        self.dists_valid = pairwise_cosine_similarity(self.features_gallery, self.features_valid)
        self.dists_test = pairwise_cosine_similarity(self.features_gallery, self.features_test)

    def _find_features(self):
        def encode_set(loader):
            features, labels = [], []
            for batch, label in tqdm(loader):
                features.append(self.model(batch.to(self.device)).cpu())
                labels.append(label)
            return torch.concat(features, dim=0), torch.concat(labels, dim=0)
        
        with torch.no_grad():
            print("Encoding gallery set...")
            self.features_gallery, self.labels_gallery = encode_set(self.gallery_loader)
            
            print("Encoding validation set...")
            self.features_valid, self.labels_valid = encode_set(self.valid_loader)
            
            print("Encoding test set...")
            self.features_test, self.labels_test = encode_set(self.test_loader)
        
    def find_thresholds(self, quantile):
        uniq, _ = self.labels_gallery.unique().sort()
        self.thresholds = torch.empty(uniq.shape)

        indexes = self.labels_valid < self.num_classes
        known_valid = self.labels_valid[indexes]
        for lab in uniq:
            features_gallery = self.features_gallery[self.labels_gallery == lab]
            features_valid   = self.features_valid[indexes]
            features_valid   = features_valid[known_valid == lab]
            
            dists = pairwise_cosine_similarity(features_gallery, features_valid)
            self.thresholds[lab] = dists.quantile(q=quantile)
    
    def set_thresholds(self, threshold):
        uniq, _ = self.labels_train.unique().sort()
        self.thresholds = torch.empty(uniq.shape)
        threshold = threshold.to(torch.float64)
        
        # Set same threshold for each class
        for lab in uniq:
            self.thresholds[lab] = threshold   
            
    def classify(self, test = False, use_th = True):
        if test:
            dists = self.dists_test
            labels = self.labels_test.cpu().numpy()
        else:
            dists = self.dists_valid
            labels = self.labels_valid.cpu().numpy()
        
        vals, idx = dists.max(dim=0)
        pred = self.labels_gallery[idx]
        if use_th:
            pred[vals < self.thresholds[pred]] = self.num_classes
            
        return pred.numpy(), labels

    def test_multiple_thresholds(self, min_th, max_th=1.0, step=0.01, use_quantile=False):
        f1_os = []
        accuracy_os = []
        accuracy_known = []
        accuracy_known_th = []
        accuracy_unknown_th = []
        
        # Use custom range for plain threshold
        if use_quantile:  
            th_range = torch.arange(min_th, max_th, step)
        else:
            th_range = torch.cat((torch.arange(0.991, 0.999, 0.001),
                                  torch.arange(0.999, 0.9999, 0.0001),
                                  torch.arange(0.9999, 0.99999, 0.00001),
                                  torch.arange(0.99999, 0.999999, 0.000001))) 
        
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
        result["threshold"] = [round(el.item(), 8) for el in th_range]
        result["known_class_acc"] = accuracy_known
        result["known_class_acc_th"] = accuracy_known_th
        result["unknown_class_acc_th"] = accuracy_unknown_th
        result["open_set_acc"] = accuracy_os
        result["open_set_f_score"] = f1_os
        return result

