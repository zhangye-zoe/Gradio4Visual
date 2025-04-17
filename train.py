# -*- coding: utf-8 -*-

# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleMedNet

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 加载数据
train_dataset = datasets.ImageFolder('fake_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 模型 & 优化器
model = SimpleMedNet(num_classes=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练
for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'med_model.pth')
print("✅ Model saved as med_model.pth")
