import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import warnings
warnings.filterwarnings("ignore")
device=torch.device("cuda 0:"if torch.cuda.is_available() else "cpu")
DATA_DIR = "TB_Chest_Radiography_Database"
batch_size=32

# Define transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),             # Resize to 256x256
    transforms.CenterCrop(224),         # Crop the center 224x224
    transforms.ToTensor(),              # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

full_dataset = datasets.ImageFolder(DATA_DIR,transform=data_transforms)
train_size = int(0.7*len(full_dataset))
val_size = len(full_dataset)-train_size
train_dataset,val_dataset = torch.utils.data.random_split(full_dataset,[train_size,val_size])


# creating data loaders
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,)
# number of classes formed
num_classes=len(full_dataset.classes)

# using the model
model=models.resnet18(pretrained=True)
num_ftrs=model.fc.in_features
model.fc=nn.Linear(num_ftrs,num_classes)

model=model.to(device)


# optimizer and loss
criterion = nn.CrossEntropyLoss()# defining loss (kitna model in crct ya loss diya)
optimizer = optim.Adam(model.parameters(), lr=0.001)#optimizing the model lr---> learning rate kitna rate har baar badhna for fast4 approach

# training model or fine tuning the model
num_epochs=10#kitni baar model 10 baar same dataset se train hoga
for epoch in range(num_epochs):
    model.train()
    running_loss=0.0#setting  to cal ki model train may kitna loss hua to find the accuracyat end
    for inputs,labels in train_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()# to start optimization
        running_loss+=loss.item()*inputs.size(0)
    epoch_loss=running_loss/len(train_loader.dataset)
    print(f"Epoch [{(epoch+1)/num_epochs}], loss [{epoch_loss:.4f}]")


# model evaluvation
model.eval()
correct=0
total=0


with torch.no_grad():
    for inputs,labels in val_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=model(inputs)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()


print(f"Accuracy on validation set: {(correct / total) * 100:.2f}%")


torch.save(model.state_dict(), 'tuberculosis_prediction_model.pth')