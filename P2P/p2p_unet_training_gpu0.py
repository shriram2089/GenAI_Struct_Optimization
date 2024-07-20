import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import os
from matplotlib.image import imread
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
print(device)

print("modules imported")

x_train_tmp_folder_path = r"C:\Users\shrir\OneDrive\Desktop\GAN\GAN_PS\DATASET\short_range\temp_short"
x_train_dsp_folder_path = r"C:\Users\shrir\OneDrive\Desktop\GAN\GAN_PS\DATASET\short_range\disp_short"
y_train_folder_path = r"C:\Users\shrir\OneDrive\Desktop\GAN\GAN_PS\DATASET\CIRCULAR_VANE_Shape_Images_1559"

def sort_img(path):
    dict1 = {}
    lst = [0]*1560
    for i in path:
        dict1[int(i[i.index('(')+1:i.index(')')])] = i
    for j in range(1,1560):
        lst[j] = dict1[j]
    return lst[1:]

x_tmp_elements = sort_img(os.listdir(x_train_tmp_folder_path))
x_dsp_elements = sort_img(os.listdir(x_train_dsp_folder_path))
y_elements = sort_img(os.listdir(y_train_folder_path))

batch_size_list = [4, 16, 32, 64, 128, 128, 128, 64, 32, 16, 32, 64, 128]
data_list = [20, 100, 200, 300, 500, 800, 1200, 1200, 1200, 1200, 1200, 1200, 1200]
epoch_list = [50]*4 + [200]*6 + [400, 300, 200]

def get_images(tmp_elements, disp_elements, y_elements, size):
    X = np.zeros((size, 2, 256, 256))
    Y = np.zeros((size, 1, 256, 256))
    for index, (tmp_element, dsp_element) in enumerate(zip(tmp_elements, disp_elements)):
        tmp_element_path = os.path.join(x_train_tmp_folder_path, tmp_element)
        dsp_element_path = os.path.join(x_train_dsp_folder_path, dsp_element)
        img_tmp = imread(tmp_element_path)
        img_dsp = imread(dsp_element_path)

        img_tmp = img_tmp.reshape((1, 256, 256))
        img_dsp = img_dsp.reshape((1, 256, 256))

        img_combined = (np.concatenate((img_tmp, img_dsp), axis=0)-0.5)/0.5
        X[index] = img_combined

    for index, Y_train_element in enumerate(y_elements):
        element_path = os.path.join(y_train_folder_path, Y_train_element)
        img = imread(element_path)
        img = np.mean(img, axis=2)
        img = img/127.5-1
        img = img.reshape((1, 256, 256))
        Y[index] = img
    return X, Y

x, y = get_images(x_tmp_elements, x_dsp_elements, y_elements, len(x_tmp_elements))

print("Data Loaded")

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        x = self.conv(x.float())
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if dropout else None
        self.leaky_relu = nn.LeakyReLU(0.1)
        nn.init.normal_(self.conv_transpose.weight, mean=0.0, std=0.02)
        if self.conv_transpose.bias is not None:
            nn.init.constant_(self.conv_transpose.bias, 0.0)

    def forward(self, x, skip_con):
        x = self.conv_transpose(x.float())
        x = self.batch_norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = torch.cat((x, skip_con.float()), dim=1)
        x = self.leaky_relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, input_shape=(2, 256, 256)):
        super(UNet, self).__init__()
        self.e1 = EncoderBlock(input_shape[0], 64, batch_norm=False)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.e5 = EncoderBlock(512, 512)
        self.e6 = EncoderBlock(512, 512)
        self.e7 = EncoderBlock(512, 512)
    

        self.b = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        nn.init.normal_(self.b.weight, mean=0.0, std=0.02)
        if self.b.bias is not None:
            nn.init.constant_(self.b.bias, 0.0)
        self.b_relu = nn.ReLU()

        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(1024, 512)
        self.d3 = DecoderBlock(1024, 512)
        self.d4 = DecoderBlock(1024, 512, dropout=False)
        self.d5 = DecoderBlock(1024, 256, dropout=False)
        self.d6 = DecoderBlock(512, 128, dropout=False)
        self.d7 = DecoderBlock(256, 64, dropout=False)

        self.output_layer = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.float()
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        b = self.b(e7)
        b = self.b_relu(b)

        d1 = self.d1(b, e7)
        d2 = self.d2(d1, e6)
        d3 = self.d3(d2, e5)
        d4 = self.d4(d3, e4)
        d5 = self.d5(d4, e3)
        d6 = self.d6(d5, e2)
        d7 = self.d7(d6, e1)

        output = self.output_layer(d7)
        output = self.tanh(output)
        return output

model = UNet()
model.to(device)

summary(model, input_size=(2, 256, 256))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()  # Ensure inputs are float32
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

'''
for i in range(13):
    x_np, y_np = x[:data_list[i]], y[:data_list[i]]
    print(data_list[i], ",", batch_size_list[i])
    x_tensor = torch.tensor(x_np).to(device)
    y_tensor = torch.tensor(y_np).to(device)
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size_list[i])
    train(model, dataloader, criterion, optimizer, epoch_list[i], device)
    torch.save(model, f'short_p2p_unet_gpu0_{i}.pth')

print("Model saved")

'''
