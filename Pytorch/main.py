import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from utils.dataset import CustomDataset
from torchvision.models import vgg16
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./tensorboard')

# Define transforms
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])

# Dataset, DataLoader
train_dataset = CustomDataset(root='./datasets', transform=transform, mode='train')
val_dataset = CustomDataset(root='./datasets', transform=transform, mode='val')
print(f"Length of train, validation dataset: {len(train_dataset)}, {len(val_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = vgg16(pretrained=True)
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=6)
model.to(device)

# Loss, optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for ep in range(10):
    model.train() # Set model to training mode
    for i, (images, labels) in enumerate(tqdm(train_dataloader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass - call model.forward() method
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        writer.add_scalar("Loss", loss.item(), i)

        # Backward pass
        optimizer.zero_grad()
        loss.backward() # Calculate gradients
        optimizer.step() # Update model weights

        if i % 10 == 0:
            print(f"Epoch {ep}, Loss: {loss.item()}")

    # Validation
    model.eval() # Set model to evaluation mode
    total = 0
    correct = 0

    with torch.no_grad():
        for (images, labels) in val_dataloader:
            total += len(images)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: prediction
            outputs = model(images)
            _, pred = torch.max(outputs, dim=1)

            correct += (pred == labels).sum().item()

    print(f"After training epoch {ep}, Validation accuracy: {correct / total}")
