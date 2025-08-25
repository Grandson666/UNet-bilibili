import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet
from data import get_dataloader

# 定义模型训练函数
def train_model(num_epochs=10, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # 实例化UNet模型
    model = UNet(in_channels=3, num_classes=1)
    model.to(device)
    train_dataloader = get_dataloader(mode='train')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            # 前向传播
            output = model(data)
            output = output["out"]
            loss = criterion(output, target)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 每个批次打印一次训练信息
            print(f'Epoch: {epoch+1}/{num_epochs} | Step: {batch_idx+1}/{len(train_dataloader)} | Loss: {loss.item():.4f}')
    print("Training finished. Saving model...")
    torch.save(model.state_dict(), "unet_drive_model.pth")
    print("Model saved as unet_drive_model.pth")

if __name__ == '__main__':
    train_model(num_epochs=20)
