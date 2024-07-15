import torch
import torch.nn as nn
from model import UNET
from data_loader import prepare_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data directories
blur_dir = 'E:/Intel/new approach/dataset/train/pixel'
sharp_dir = 'E:/Intel/new approach/dataset/train/orginal'

# Prepare data loaders
train_loader, valid_loader, test_loader = prepare_data(blur_dir, sharp_dir, batch_size=10)

# Initialize model, loss function, and optimizer
model = UNET().to(device)
criterion = nn.L1Loss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for i, (b_img, s_img) in enumerate(train_loader):  
        b_img, s_img = b_img.to(device, dtype=torch.float), s_img.to(device, dtype=torch.float)
        outputs = model(b_img)
        loss = criterion(outputs, s_img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Save the model after training
torch.save(model.state_dict(), 'unet_model.pth')
print("Model saved as unet_model.pth")
