import torch
from torch.utils.data import DataLoader
import sys
import numpy as np
import argparse
from utils import OpenMax

sys.path.append('./')
sys.path.append('../')
from dataset_process import dataset_creator
from train_utils import train_test_transform, get_device, get_dataset_path, get_trial_path
from metrics import compute_metrics
from backbones import Backbone

# Parse arguments 
def parse_arguments():
    parser = argparse.ArgumentParser(description='Open Set Classifier Training')
    parser.add_argument('--num_trials', default = 5, type = int, help='Number of trials to average results over?')
    parser.add_argument('--start_trial', default = 0, type = int, help='Trial number to start evaluation for?')
    parser.add_argument('--backbone', default = None, type = str, help='Define backbone model', choices = ['resnet18', 'densenet121'])
    parser.add_argument('--num_unk', required = True, type = int, help='Number of unknown classes. Set 0 to train in closed set mode')
    parser.add_argument('--use_quantile', action='store_true', help="If flag is set, quantile will be used")
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--name', default="", type = str, help='Name of the model file') 
    return parser.parse_args()


def get_classifier(args, trial):
    device = get_device()
    path = get_dataset_path(args.dataset)
    trial_path = get_trial_path(args.dataset)
    
    # Data transforms
    transform_train = train_test_transform(num_channels=3, normalize=args.no_normalize).train()
    transform_test = train_test_transform(num_channels=3, normalize=args.no_normalize).eval()
    transform = (transform_train, transform_test)

    # Define variables
    batch_size = args.batch_size

    # Define the unknown indices
    train_dataset, valid_dataset, test_dataset = dataset_creator.get_datasets(path, trial_path, trial, transform, unk_in_valid=True, gallery=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)
    test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)

    # Set the num classes
    num_classes = train_dataset.num_classes

    # Model
    file_name = f"./models_phyto/OpenMax_{args.backbone}_{trial_num}_{args.name}_best_acc.pth"
    print(f"Currently testing model: {file_name}\n")
    model = Backbone(args.backbone, num_classes)
    model.load_state_dict(torch.load(file_name))
    model = model.to(device)
    
    # OpenMax model
    openmax = OpenMax(model, train_loader, valid_loader, test_loader, device, tailsize=20, alpha=5)
    return openmax

# Get arguments
args = parse_arguments()

# Define arrays
start_trial = args.start_trial
num_trials = args.num_trials

# Finds the best threshold using validation data
valid_result = []
for trial_num in range(start_trial, start_trial + num_trials):
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    openmax = get_classifier(args, trial_num)
    
    result = openmax.test_multiple_thresholds(min_th=0.75, max_th = 1.0, step = 0.01, use_quantile=args.use_quantile)
    valid_result.append(result)

# Find the best threshold on average for the validation data and use it for test data
metric_values = np.array([trial["open_set_f_score"] for trial in valid_result])
mean_primary_metric = np.mean(metric_values, axis=0)
best_threshold_idx = np.argmax(mean_primary_metric)
best_threshold = torch.tensor(valid_result[0]['threshold'][best_threshold_idx])

# Test set
test_result = []
for trial_num in range(start_trial, start_trial + num_trials):
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    # Get the classifier
    openmax = get_classifier(args, trial_num)
    
    # set the thresholds
    if args.use_quantile:
        openmax.find_thresholds(best_threshold)
    else:
        openmax.set_thresholds(best_threshold)  
        
    preds_no_th, labels_test = openmax.classify(use_th=False, test=True)
    preds, labels_test = openmax.classify(use_th=True, test=True)

    # Compute metrics
    results = compute_metrics(labels_test, preds_no_th, preds, openmax.num_classes)

    result = {}
    result["quantile"] = [best_threshold]
    result["known_class_acc"] = [round(results["acc_known"], 4)]
    result["known_class_acc_th"] = [round(results["acc_known_th"], 4)]
    result["unknown_class_acc_th"] = [round(results["acc_unk_th"], 4)]
    result["open_set_acc"] = [round(results["acc_os"], 4)]
    result["open_set_f_score"] = [round(results["f1_os"], 4)]
    test_result.append(result)

print(valid_result)
print(test_result)