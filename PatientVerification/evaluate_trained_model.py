import torch
from networks.SiameseNetwork import SiameseNetwork
from utils import Utils
from sklearn import metrics
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import argparse


parser = argparse.ArgumentParser('Evaluate Trained Model On ChestX-ray14 Data')
parser.add_argument('--image_path', help='the path where the images are stored')
args = parser.parse_args()

image_path = args.image_path

# Define some important parameters. Do not change to reproduce our best results.
image_size = 256
n_samples = 100000
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the pre-trained model
if torch.cuda.is_available():
    model = SiameseNetwork(network='ResNet-50', in_channels=3, n_features=128).cuda()
    model.load_state_dict(torch.load('./trained_models/verification_approach_final_model.pth'))
else:
    model = SiameseNetwork(network='ResNet-50', in_channels=3, n_features=128)
    model.load_state_dict(torch.load('./trained_models/verification_approach_final_model.pth', map_location='cpu'))

# Load the data
test_loader = Utils.get_data_loaders(phase='testing', data_handling='balanced', n_channels=3, n_samples=n_samples,
                                     transform=transform, image_path=image_path, batch_size=32, shuffle=False,
                                     num_workers=0, pin_memory=True, save_path=None)

# Testing phase
model.eval()
y_true = None
y_pred = None

print('Testing----->')
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        inputs1, inputs2, labels = batch

        if y_true is None:
            y_true = labels
        else:
            y_true = torch.cat((y_true, labels), 0)

        if torch.cuda.is_available():
            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

        outputs = model(inputs1, inputs2)
        outputs = torch.sigmoid(outputs)

        if y_pred is None:
            y_pred = outputs.cpu()
        else:
            y_pred = torch.cat((y_pred, outputs.cpu()), 0)

        print('Progress: ' + str(np.round((i + 1) * 100 / len(test_loader), 2)) + '%')

y_pred = y_pred.squeeze()

# Compute evaluation metrics
fp_rates, tp_rates, thresholds = metrics.roc_curve(y_true, y_pred)
auc = metrics.roc_auc_score(y_true, y_pred)
y_pred_thresh = Utils.apply_threshold(y_pred, 0.5)
accuracy, f1_score, precision, recall, report, confusion_matrix = Utils.get_evaluation_metrics(y_true, y_pred_thresh)
Utils.save_results_to_file(auc, accuracy, f1_score, precision, recall, report, confusion_matrix, './archive/',
                           'chestXray14')

# Print the metrics
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
plt.savefig('./archive/trained_model_ROC_curve_chestXray14.png')
