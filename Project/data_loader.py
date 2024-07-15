import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class MyData(Dataset):
    def __init__(self, df, blur_path, sharpen_path, transform):
        self.df = df
        self.transform = transform
        self.bp = blur_path
        self.sp = sharpen_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        blur_image_path = os.path.join(self.bp, self.df['blur'][index])
        sharpen_image_path = os.path.join(self.sp, self.df['sharp'][index])
        bimg = cv2.imread(blur_image_path)
        simg = cv2.imread(sharpen_image_path)
        x = self.transform(bimg)
        y = self.transform(simg)
        return x, y

def prepare_data(blur_dir, sharp_dir, batch_size):
    blur = sorted(os.listdir(blur_dir))
    sharp = sorted(os.listdir(sharp_dir))
    img_names = {'blur': blur, 'sharp': sharp}
    df = pd.DataFrame(img_names)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor(),
        T.ConvertImageDtype(torch.float)
    ])

    train_df = df[:245]
    test_df = df[245:300]
    valid_df = df[300:]

    train_data = MyData(train_df, blur_dir, sharp_dir, transform)
    test_data = MyData(test_df, blur_dir, sharp_dir, transform)
    valid_data = MyData(valid_df, blur_dir, sharp_dir, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader
