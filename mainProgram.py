import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir)])

        # Define the RGB colors for each class
        self.class_colors = {
            (2, 0, 0): 0,       # LTE class
            (127, 0, 0): 1,     # NR class
            (200, 161, 159): 2  # Noise class
        }
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label
        label = cv2.imread(self.label_paths[idx])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # Map RGB colors to class indices
        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, idx in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = idx

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            label_mask = torch.from_numpy(label_mask).long()

        return image, label_mask


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  
    transforms.ToTensor()
])

train_dataset = SemanticSegmentationDataset(
    image_dir='/kaggle/input/spectrogramdata-new/trainSet/input',  
    label_dir='/kaggle/input/spectrogramdata-new/trainSet/label',
    transform=train_transform
)

val_dataset = SemanticSegmentationDataset(
    image_dir='/kaggle/input/spectrogramdata-new/testSet/input',
    label_dir='/kaggle/input/spectrogramdata-new/testSet/label',
    transform=train_transform
)

# Prepare Data for training
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Check Data again
print(len(train_dataset))
print(len(val_dataset))
print(len(train_dataloader))
print(len(val_dataloader))
print(len(test_dataloader))

# import PyTorch framwork and some libs for training
import torch
from torch.optim import Adam
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F

# Building the our proposed model from architecture
class PRMNet(nn.Module):
    def __init__(self, n_classes):
        super(PRMNet, self).__init__()
        
        # Multi-scale
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3)
        self.batchnorm = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(192)
        self.conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)


        # Conv Layer
        # Layer 1
        self.conv_layer1_1 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer1_2 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        
        # Layer 2
        self.conv_layer2_1 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer2_2 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer2_3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer2_4 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer2_5 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        
        # Layer 3
        self.conv_layer3_1 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer3_2 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer3_3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer3_4 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer3_5 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer3_6 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer3_7 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer3_8 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        
        # Layer 4
        self.conv_layer4_1 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer4_2 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer4_3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer4_4 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer4_5 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer4_6 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer4_7 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer4_8 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer4_9 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer4_10 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer4_11 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        
        # Layer 5
        self.conv_layer5_1 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer5_2 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer5_3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer5_4 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer5_5 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer5_6 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer5_7 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_layer5_8 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        
        # Layer 6
        self.conv_layer6 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        
        
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.upsample8 = nn.Upsample(scale_factor=8, mode="bilinear")


        self.conv_last1 =  nn.Conv2d(192, 64, kernel_size=5, stride=1, padding=2)
        self.batchnorm_last1 = nn.BatchNorm2d(64)
        self.conv_last2 =  nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2)
        self.batchnorm_last2 = nn.BatchNorm2d(16)
        self.conv_last3 =  nn.Conv2d(16, n_classes, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        
        # Multi-Scale
        scale1 = self.conv1(x)
        scale2 = self.conv2(x)
        scale3 = self.conv3(x)
        scale4 = self.conv4(x)
        x01 = torch.cat([scale1, scale2], dim=1)
        x02 = torch.cat([scale3, scale4], dim=1)
        x = torch.cat([x01, x02], dim=1)
        x = self.batchnorm(x)
        x = F.relu(x)
        
        # Layer1
        x1 = self.conv_layer1_1(x)
        x1 = F.relu(x1)
        x1 = self.conv_layer1_2(x1)
        x1 = F.relu(x1)
        
        # Layer2
        x2 = self.maxpool_layer(x1)
        x1 = self.conv_layer2_1(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer2_2(x2)
        x2 = F.relu(x2)
        x1 = self.conv_layer2_3(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer2_4(x2)
        x2 = F.relu(x2)
        x1 = self.conv_layer2_5(x1)
        x1 = F.relu(x1)
        
        # Layer3
        x3_1 = self.maxpool_layer(x2)
        x3_2 = self.maxpool_layer(x1)
        x3_2 = self.maxpool_layer(x3_2)
        x3 = x3_1 + x3_2
        
        x2_ = x2
        x2_1 = self.conv_layer3_1(x2)
        x2_1 = F.relu(x2_1)
        x2_2 = self.maxpool_layer(x1)
        x2 = x2_1 + x2_2
        
        x1_1 = self.conv_layer3_2(x1)
        x1_1 = F.relu(x1_1)
        x1_2 = self.upsample2(x2_)
        x1 = x1_1 + x1_2
        ###################
        x1 = self.conv_layer3_3(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer3_4(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv_layer3_5(x3)
        x3 = F.relu(x3)
        
        x1 = self.conv_layer3_6(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer3_7(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv_layer3_8(x3)
        x3 = F.relu(x3)
        
        # Layer4
        x4_1 = self.maxpool_layer(x3)
        x4_2 = self.maxpool_layer(x2)
        x4_2 = self.maxpool_layer(x4_2)
        x4_3 = self.maxpool_layer(x1)
        x4_3 = self.maxpool_layer(x4_3)
        x4_3 = self.maxpool_layer(x4_3)
        x4 = x4_1 + x4_2
        x4 = x4 + x4_3
        
        x3_ = x3
        x3_1 = self.conv_layer4_1(x3)
        x3_1 = F.relu(x3_1)
        x3_2 = self.maxpool_layer(x2)
        x3_3 = self.maxpool_layer(x1)
        x3_3 = self.maxpool_layer(x3_3)
        x3 = x3_1 + x3_2
        x3 = x3 + x3_3
        
        x2_ = x2
        x2_1 = self.conv_layer4_2(x2)
        x2_1 = F.relu(x2_1)
        x2_2 = self.upsample2(x3_)
        x2_3 = self.maxpool_layer(x1)
        x2 = x2_1 + x2_2
        x2 = x2 + x2_3
        
        x1_1 = self.conv_layer4_3(x1)
        x1_1 = F.relu(x1_1)
        x1_2 = self.upsample2(x2_)
        x1_3 = self.upsample4(x3_)
        x1 = x1_1 + x1_2
        x1 = x1 + x1_3
        #####################
        x1 = self.conv_layer4_4(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer4_5(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv_layer4_6(x3)
        x3 = F.relu(x3)
        
        x4 = self.conv_layer4_7(x4)
        x4 = F.relu(x4)
        
        x1 = self.conv_layer4_8(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer4_9(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv_layer4_10(x3)
        x3 = F.relu(x3)
        
        x4 = self.conv_layer4_11(x4)
        x4 = F.relu(x4)
        
        # Layer5
        x4_ = x4
        x4_1 = self.conv_layer5_1(x4)
        x4_1 = F.relu(x4_1)
        x4_2 = self.maxpool_layer(x3)
        x4_3 = self.maxpool_layer(x2)
        x4_3 = self.maxpool_layer(x4_3)
        x4_4 = self.maxpool_layer(x1)
        x4_4 = self.maxpool_layer(x4_4)
        x4_4 = self.maxpool_layer(x4_4)
        x4 = x4_1 + x4_2
        x4 = x4 + x4_3
        x4 = x4 + x4_4
        
        x3_ = x3
        x3_1 = self.conv_layer5_2(x3)
        x3_1 = F.relu(x3_1)
        x3_2 = self.maxpool_layer(x2)
        x3_3 = self.maxpool_layer(x1)
        x3_3 = self.maxpool_layer(x3_3)
        x3_4 = self.upsample2(x4_)
        x3 = x3_1 + x3_2
        x3 = x3 + x3_3
        x3 = x3 + x3_4
        
        x2_ = x2
        x2_1 = self.conv_layer5_3(x2)
        x2_1 = F.relu(x2_1)
        x2_2 = self.upsample2(x3_)
        x2_3 = self.maxpool_layer(x1)
        x2_4 = self.upsample4(x4_)
        x2 = x2_1 + x2_2
        x2 = x2 + x2_3
        x2 = x2 + x2_4
        
        x1_1 = self.conv_layer5_4(x1)
        x1_1 = F.relu(x1_1)
        x1_2 = self.upsample2(x2_)
        x1_3 = self.upsample4(x3_)
        x1_4 = self.upsample8(x4_)
        x1 = x1_1 + x1_2
        x1 = x1 + x1_3
        x1 = x1 + x1_4
        #########################
        x1 = self.conv_layer5_5(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv_layer5_6(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv_layer5_7(x3)
        x3 = F.relu(x3)
        
        x4 = self.conv_layer5_8(x4)
        x4 = F.relu(x4)
        
        # Layer6
        x1_1 = self.conv_layer6(x1)
        x1_1 = F.relu(x1_1)
        x1_2 = self.upsample2(x2)
        x1_3 = self.upsample4(x3)
        x1_4 = self.upsample8(x4)
        x1 = x1_1 + x1_2
        x1 = x1 + x1_3
        x1 = x1 + x1_4
        x1 = self.conv_out(x1)
        
        x = self.conv(x)
        x = torch.cat([x, x1], dim=1)
        x = self.batchnorm2(x)
        x = F.relu(x)
        
        x = self.conv_last1(x)
        x = F.relu(x)
        x = self.conv_last2(x)
        x = F.relu(x)
        x = self.conv_last3(x)
        
        return F.softmax(x)
    

classes = 3  # including 3 classes: LTE, 5G NR, and Noise
model = PRMNet(classes)
model = nn.DataParallel(model)  # Using many GPUs or CPUs for training to improve speed of training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # check type of device again
model.to(device)

def count_parameters(model):  # The function to count total parameters stored in memory, using for model
    return sum(p.numel() for p in model.parameters())

total_params = count_parameters(model)
print(f"Total parameters: {total_params}")


from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision

# Creating train and evaluate function
def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    # Instantiate metrics
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)
    
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        preds = torch.argmax(outputs, dim=1)
        
        # Update confusion matrix
        confmat(preds, labels)
        
        # Update metrics
        iou_metric(preds, labels)
        f1_metric(preds, labels)
        accuracy_metric(preds, labels)
        precision_metric(preds, labels)
        
        # Update tqdm description with metrics
        pbar.set_postfix({
            'Batch Loss': f'{loss.item():.4f}',
            'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
            'Mean IoU': f'{iou_metric.compute():.4f}',
            'Mean F1 Score': f'{f1_metric.compute():.4f}',
            'Mean Precision': f'{precision_metric.compute():.4f}'
        })
    
    cm = confmat.compute().cpu().numpy()  # converting to numpy
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
    confmat.reset()
   
    epoch_loss = running_loss / len(dataloader.dataset)
    # Calculate mean metrics
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()
   
    return cm_normalized, epoch_loss, mean_iou, mean_f1, mean_accuracy, mean_precision

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    
    # Instantiate metrics
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)


    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            # Update confusion matrix
            confmat(preds, labels)

            # Update metrics
            iou_metric(preds, labels)
            f1_metric(preds, labels)
            accuracy_metric(preds, labels)
            precision_metric(preds, labels)


            # Update tqdm description with metrics
            pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
                'Mean IoU': f'{iou_metric.compute():.4f}',
                'Mean F1 Score': f'{f1_metric.compute():.4f}',
                'Mean Precision': f'{precision_metric.compute():.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    cm = confmat.compute().cpu().numpy()  # Convert to numpy for easy usage
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
    confmat.reset()

    # Calculate mean metrics
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()

    return cm_normalized, epoch_loss, mean_iou, mean_f1, mean_accuracy, mean_precision

import copy
# Training Model
num_epochs = 100 
num_classes = 3
epoch_saved = 0
criterion = nn.CrossEntropyLoss()  
optimizer = Adam(model.parameters(), lr=1e-5)

best_val_accuracy = 0.0  
best_model_state = None

for epoch in range(num_epochs):
    _, epoch_loss_train, iou_score_avg_train, f1_score_avg_train, accuracy_avg_train, precision_avg_train = train_epoch(model, train_dataloader, criterion, optimizer, device, num_classes)
    _, epoch_loss_val, iou_score_avg_val, f1_score_avg_val, accuracy_avg_val, precision_avg_val = evaluate(model, val_dataloader, criterion, device, num_classes)
    
    print(f"Epoch {epoch + 1}/{100}")
    print(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {accuracy_avg_train:.4f}, Mean IoU: {iou_score_avg_train:.4f}, Mean F1 Score: {f1_score_avg_train:.4f}, Mean Precision: {precision_avg_train:.4f}")
    print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {accuracy_avg_val:.4f}, Mean IoU: {iou_score_avg_val:.4f}, Mean F1 Score: {f1_score_avg_val:.4f}, Mean Precision: {precision_avg_val:.4f}")

    if accuracy_avg_val >= best_val_accuracy:   # Choosing trainable parameters which have the highest mAcc on validation dataset
        epoch_saved = epoch + 1 
        best_val_accuracy = accuracy_avg_val
        best_model_state = copy.deepcopy(model.state_dict())
    
print("===================")
print(f"Best Model at epoch : {epoch_saved}")
torch.save(best_model_state, "PRMNet_proposed.pth")