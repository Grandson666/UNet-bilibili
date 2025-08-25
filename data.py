import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 定义数据根目录
DATA_ROOT = "DRIVE"
TRAIN_DIR = os.path.join(DATA_ROOT, "training")
TEST_DIR = os.path.join(DATA_ROOT, "test")

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 定义图像和掩码所在的子目录
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "1st_manual")
        # 获取图像文件列表
        self.images_list = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".tif")])
    def __len__(self):
        # 返回数据集中图像的总数
        return len(self.images_list)
    def __getitem__(self, idx):
        # 根据索引获取图像和掩码的文件名
        img_name = self.images_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        # 根据图像文件名推断出对应的掩码文件名
        if "training" in self.data_dir:
            mask_name = img_name.replace("_training.tif", "_manual1.gif")
        elif "test" in self.data_dir:
            mask_name = img_name.replace("_test.tif", "_manual1.gif")
        else:
            raise ValueError("Unknown data directory format.")
        mask_path = os.path.join(self.masks_dir, mask_name)
        # 原始图像转换为RGB格式
        image = Image.open(img_path).convert("RGB")
        # 掩码图像转换为单通道灰度图
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        # 将掩码转换为二值浮点张量，大于0.5的像素设为1.0，否则为0.0
        mask = (mask > 0.5).float()
        return image, mask

def get_dataloader(batch_size=4, shuffle=True, mode='train'):
    # 根据模式类型返回相应的数据加载器
    if mode == 'train':
        data_dir = TRAIN_DIR
    elif mode == 'test':
        data_dir = TEST_DIR
    else:
        raise ValueError("Mode must be 'train' or 'test'.")
    # 检查本地文件夹是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Please ensure the 'DRIVE' folder is in your project directory.")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    # 实例化自定义的ImageDataset
    dataset = ImageDataset(data_dir, transform=transform)
    # 实例化PyTorch的DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
