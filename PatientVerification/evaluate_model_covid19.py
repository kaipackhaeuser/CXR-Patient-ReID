import torch
from torch.utils import data
from networks.SiameseNetwork import SiameseNetwork
from utils import Utils
from sklearn import metrics
import torchvision.transforms as transforms
from dataset.SiameseDatasetCovid19 import SiameseDatasetCovid19
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser('Evaluate Trained Model  On Covid Data')
parser.add_argument('--image_path', help='the path where the images are stored')
args = parser.parse_args()

image_path = args.image_path

# Define some important parameters
image_size = 256
n_samples = 1548
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model and its state dict
model = SiameseNetwork(network='ResNet-50', in_channels=3, n_features=128).cuda()
model.load_state_dict(torch.load('./trained_models/verification_approach_final_model.pth'))

# Define the data set and the corresponding data loader
test_set = SiameseDatasetCovid19(phase='testing', n_channels=3, n_samples=792294, transform=transform,
                                 image_path=image_path)
test_loader = data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

# Testing phase
y_true, y_pred = Utils.test(model, test_loader)
y_true, y_pred = [y_true.numpy(), y_pred.numpy()]

# Compute the evaluation metrics!
fp_rates, tp_rates, thresholds = metrics.roc_curve(y_true, y_pred)
auc = metrics.roc_auc_score(y_true, y_pred)
y_pred_thresh = Utils.apply_threshold(y_pred, 0.5)
accuracy, f1_score, precision, recall, report, confusion_matrix = Utils.get_evaluation_metrics(y_true, y_pred_thresh)
Utils.save_results_to_file(auc, accuracy, f1_score, precision, recall, report, confusion_matrix, './archive/', 'covid')

# Print the evaluation metrics!
print('EVALUATION METRICS:')
print('AUC: ' + str(auc))
print('Accuracy: ' + str(accuracy))
print('F1-Score: ' + str(f1_score))
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
print('Report: ' + str(report))
print('Confusion matrix: ' + str(confusion_matrix))

# Plot ROC curve
plt.figure()
plt.plot(fp_rates, tp_rates, label='ROC Curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('./archive/trained_model_ROC_curve_covid.png')
