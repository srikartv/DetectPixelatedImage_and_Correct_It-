import torch
import torch.nn as nn

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.conv_block1 = self.conv_block(3, 64)   
        self.conv_block2 = self.conv_block(64, 128)
        self.conv_block3 = self.conv_block(128, 256)
        self.conv_block4 = self.conv_block(256, 512)
        self.conv_block5 = self.conv_block(512, 1024)
        
        self.up_conv1 = self.up_conv(1024, 512)
        self.up_conv2 = self.up_conv(512, 256)
        self.up_conv3 = self.up_conv(256, 128)
        self.up_conv4 = self.up_conv(128, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding='same')

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        )

    def pool(self, x):
        return nn.AvgPool2d(kernel_size=2, stride=2)(x)

    def forward(self, x):
        out1 = self.conv_block1(x)
        pool1 = self.pool(out1)   # 64, 64, 64
        
        out2 = self.conv_block2(pool1)
        pool2 = self.pool(out2)  # 128, 32, 32
        
        out3 = self.conv_block3(pool2)
        pool3 = self.pool(out3)
        
        out4 = self.conv_block4(pool3)
        pool4 = self.pool(out4)  # 512, 16, 16

        out5 = self.conv_block5(pool4)  # 1024, 8, 8
        
        up1 = self.up_conv1(out5)  # 512, 16, 16
        up1 = torch.cat((out4, up1), dim=1)  # 1024, 16, 16
        up1 = self.conv_block(1024, 512)(up1)  # 512, 16, 16

        up2 = self.up_conv2(up1)
        up2 = torch.cat((out3, up2), dim=1)
        up2 = self.conv_block(512, 256)(up2)  # 256, 32, 32

        up3 = self.up_conv3(up2)
        up3 = torch.cat((out2, up3), dim=1)
        up3 = self.conv_block(256, 128)(up3)

        up4 = self.up_conv4(up3)
        up4 = torch.cat((out1, up4), dim=1)
        up4 = self.conv_block(128, 64)(up4)  # 64, 128, 128

        return self.final_conv(up4)  # 3, 128, 128

def load_model(model_path, device):
    model = UNET()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model
