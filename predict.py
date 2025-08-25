import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from model import UNet
from data import get_dataloader

def predict_image(model_path, image_path, output_path):
    """
    使用预训练的UNet模型对单个图像进行分割预测
    model_path (str): 预训练模型文件的路径
    image_path (str): 待预测的输入图像路径
    output_path (str): 保存预测掩码图像的路径
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, num_classes=1)
    # 加载预训练模型的参数到模型中，并确保其加载到正确的设备上
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    # 打开输入图像并确保其为RGB格式
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        output = output["out"]
        # 应用Sigmoid函数，将输出转换为概率值
        probabilities = torch.sigmoid(output)
        # 对概率值进行阈值处理，生成二值掩码，并移除多余的维度
        predicted_mask = (probabilities > 0.5).float().squeeze(0).squeeze(0)
    predicted_mask_np = predicted_mask.cpu().numpy() * 255
    predicted_mask_img = Image.fromarray(predicted_mask_np.astype(np.uint8))
    # 保存预测的掩码图像
    predicted_mask_img.save(output_path)
    print(f"Prediction saved to {output_path}")

if __name__ == '__main__':
    model_file = "unet_drive_model.pth"
    test_dataloader = get_dataloader(mode='test')
    # 获取测试图像的目录和第一个图像的文件名
    test_image_dir = test_dataloader.dataset.images_dir
    test_image_name = test_dataloader.dataset.images_list[0]
    input_image_file = os.path.join(test_image_dir, test_image_name)
    # 定义输出掩码文件名
    output_mask_file = "predicted_drive_mask.png"
    predict_image(model_file, input_image_file, output_mask_file)
