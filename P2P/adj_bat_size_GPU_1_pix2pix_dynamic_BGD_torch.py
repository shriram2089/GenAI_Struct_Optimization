# -*- coding: utf-8 -*-
"""Pix2Pix_MINI_BATCH_GRAD_DESCENT-TORCH.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1H63TshczZC-o6IgoTCSgyQaRZ4qJFXO1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.image import imread
import torch.optim as optim
from torchsummary import summary

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

print("Modules imported")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(torch.__version__)

x_train_tmp_folder_path = r"/home/me22btech11023/GAN/data/temp_new/"
x_train_dsp_folder_path = r"/home/me22btech11023/GAN/data/disp_new/"
y_train_folder_path = r"/home/me22btech11023/GAN/data/CIRCULAR_VANE_Shape_Images_1559/"


x_tmp_elements = os.listdir(x_train_tmp_folder_path)
x_dsp_elements = os.listdir(x_train_dsp_folder_path)
y_elements = os.listdir(y_train_folder_path)

x_tmp_elements.sort()
x_dsp_elements.sort()
y_elements.sort()
x_dsp_elements[1115:1120],x_tmp_elements[1115:1120],y_elements[1115:1120]


x_np = x[0:500]
y_np = y[0:500]
BATCH_SIZE = 16
model_name = 'p2p_gen_d_500_bs_16.pth'


# DEVELOPING X_train MATRIX

def get_images(tmp_elements,disp_elements,y_elements,size):
  X= np.zeros((size,2,256,256))
  Y= np.zeros((size,1,256,256))
  for index, (tmp_element, dsp_element) in enumerate(zip(tmp_elements, disp_elements)):
      # Load images for each channel
      tmp_element_path = os.path.join(x_train_tmp_folder_path, tmp_element)
      dsp_element_path = os.path.join(x_train_dsp_folder_path, dsp_element)
      img_tmp = imread(tmp_element_path)
      img_dsp = imread(dsp_element_path)

      # Reshape images for each channel
      img_tmp = img_tmp.reshape((1,256,256))
      img_dsp = img_dsp.reshape((1,256,256))

      # Combine channels
      img_combined = (np.concatenate((img_tmp, img_dsp), axis=0)-0.5)/0.5

      # Assign to X_train
      X[index] = img_combined

  # DEVELOPING Y_train MATRIX
  for index,Y_train_element in enumerate(y_elements):

      element_path = os.path.join(y_train_folder_path, Y_train_element)
      img = imread(element_path)
      img = np.mean(img, axis=2)
      img = img/127.5-1
      img = img.reshape((1,256, 256))
      Y[index] = img
  return X,Y

x,y = get_images(x_tmp_elements,x_dsp_elements,y_elements,len(x_tmp_elements))

print("Data Loaded")



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Initialize weights
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

        # Initialize weights
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

class GeneratorModel(nn.Module):
    def __init__(self, input_shape=(2,256,256)):
        super(GeneratorModel, self).__init__()
        self.e1 = EncoderBlock(input_shape[0], 64, batch_norm=False)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.e5 = EncoderBlock(512, 512)
        self.e6 = EncoderBlock(512, 512)
        self.e7 = EncoderBlock(512, 512)

        self.b = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        # Initialize weights
        nn.init.normal_(self.b.weight, mean=0.0, std=0.02)
        if self.b.bias is not None:
            nn.init.constant_(self.b.bias, 0.0)

        self.b_relu = nn.ReLU()

        self.d1 = DecoderBlock(512, 512)
        self.d2 = DecoderBlock(1024, 512)  # 512 + 512 (skip connection)
        self.d3 = DecoderBlock(1024, 512)  # 512 + 512 (skip connection)
        self.d4 = DecoderBlock(1024, 512, dropout=False)  # 512 + 512 (skip connection)
        self.d5 = DecoderBlock(1024, 256, dropout=False)  # 512 + 256 (skip connection)
        self.d6 = DecoderBlock(512, 128, dropout=False)  # 256 + 128 (skip connection)
        self.d7 = DecoderBlock(256, 64, dropout=False)   # 128 + 64 (skip connection)

        self.output_layer = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        # Initialize weights
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

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
        #output = self.sigmoid(output)
        output = self.tanh(output)
        return output



# Binary Cross Entropy Loss with logits
loss = nn.BCEWithLogitsLoss()
#loss = nn.BCELoss()
# Generator Loss Function
# Generator Loss Function
def gen_loss(generated_output, g_output, target):
    lambda_ = 100
    print(generated_output.shape)
    # GAN Loss
    gan_loss = loss(generated_output, torch.ones_like(generated_output))

    # Convert target tensor to the same data type as the generated output
    target = target.type_as(generated_output)

    # L1 Loss
    l1_loss = torch.mean(torch.abs(target - g_output))

    # Total Generator Loss
    g_loss_total = gan_loss + (lambda_ * l1_loss)

    return g_loss_total



# Define the model
generator = GeneratorModel(input_shape=(2,256,256))

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


generator.to(device)

# Print the model summary
summary(generator, input_size=(2, 256, 256))

class DiscriminatorModel(nn.Module):
    def __init__(self, input_shape=(3, 256, 256)):
        super(DiscriminatorModel, self).__init__()
        self.init_weights = nn.init.normal_

        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                self.init_weights(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data = m.bias.data.float()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.float()
                m.bias.data = m.bias.data.float()

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.lrelu2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.lrelu3(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.lrelu4(x)

        x = self.conv5(x)
        return x

discriminator = DiscriminatorModel(input_shape=(3,256,256))
#discriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator.to(device)
# Print the model summary
summary(discriminator, (3,256,256))

def disc_loss(real_output, fake_output):
    real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))
    fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss



# Generator optimizer
gen_opt = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Discriminator optimizer
disc_opt = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))



def compare_images(input_test, target, epoch):

    #generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        input_test = input_test.to(device)  # Move input tensor to the same device as the model
        target = target.to(device)  # Move target tensor to the same device as the model

        generated = generator(input_test)
    print(target[0].shape, generated[0].shape)

    plt.figure(figsize=(15, 5))

    images_list = [target[0].numpy(), generated[0].numpy()]  # Move tensors back to CPU for plotting
    title = ['Real (ground truth)', 'Generated Image (fake)']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.imshow(np.reshape(images_list[i],(256,256,1)),cmap='gray')
        #plt.imshow(images_list[i].transpose(1, 2, 0) * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    plt.suptitle(f'EPOCH {epoch}')
    plt.show()
    



"""#@tf.function

#helper function
def train_step(input_img, real):
  with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
    g_output = generator(input_img, training = True)
    d_output_real = discriminator([input_img, real], training = True)
    d_generated_output = discriminator([input_img, g_output], training = True)
    g_loss_total, g_loss_gan, g_loss_l1 =gen_loss(d_generated_output, g_output, real)
    d_loss = disc_loss(d_output_real, d_generated_output)

  return g_loss_total, d_loss

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
batch_size = 4

import torch
from torch.utils.data import DataLoader, TensorDataset

# Assuming x and y are NumPy arrays
x_np = x[0:500]
y_np = y[0:500]

# Convert NumPy arrays to PyTorch tensors
x_tensor = torch.tensor(x_np)
y_tensor = torch.tensor(y_np)

# Create a PyTorch dataset
dataset = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size)

generator.to(device)
discriminator.to(device)


#def train(train_loader, generator, discriminator, gen_opt, disc_opt, gen_loss, disc_loss, epochs, device):
generator.train()
discriminator.train()

for epoch in range(1, epochs + 1):
        g_loss_total = []
        d_loss_total = []

        for i, (input_img, real) in enumerate(train_loader):
            #print(i)
            batch_size = input_img.size(0)

            input_img, real = input_img.to(device), real.to(device)

            gen_opt.zero_grad()


            g_output = generator(input_img)  #fake

            #print(d_output_fake.shape, g_output.shape)
            d_output_fake = discriminator(torch.cat([input_img, g_output], dim=1))

            g_loss = gen_loss(d_output_fake, g_output, real)  #to not update disc


            #g_loss.backward()
            g_loss.backward(retain_graph=True)
            gen_opt.step()

            print(i)
            disc_opt.zero_grad()

            d_output_real = discriminator(torch.cat([input_img, real], dim=1))
            d_output_fake = discriminator(torch.cat([input_img, g_output.detach()], dim=1))

            d_loss = disc_loss(d_output_real, d_output_fake)

            d_loss.backward()


            disc_opt.step()


            #g_loss = g_loss.detach()
            #d_loss = d_loss.detach()

            g_loss_total.append(g_loss.item())
            d_loss_total.append(d_loss.item())

            if i % 5 == 0:
                compare_images(input_img, real, epoch)

            print(f"\rEpoch [{epoch}/{epochs}], Batch [{i+1}/{len(train_loader)}], "
                  f"Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}", end='')

        print(f"\rEpoch [{epoch}/{epochs}], "
              f"Generator Loss: {sum(g_loss_total) / len(g_loss_total):.4f}, "
              f"Discriminator Loss: {sum(d_loss_total) / len(d_loss_total):.4f}")
"""

from torchvision.utils import save_image
def save_some_examples(gen, train_loader, epoch, folder):
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()






print("Training started")

def train_fn(disc, gen, loader, opt_disc, opt_gen, g_scaler, d_scaler,l1_loss,bce,L1_LAMBDA=100):
    loop = tqdm(loader, leave=True)
    t = iter(loader)
    (X,Y) = next(t)
    for idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(torch.cat([x, y], dim=1))
            D_real_loss = ss = bce(D_real, torch.ones_like(D_real))
            #D_fake = disc(x, y_fake.detach())
            D_fake = disc(torch.cat([x, y_fake.detach()], dim=1))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            #D_loss = disc_loss(D_real, D_fake)

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(torch.cat([x, y_fake], dim=1))
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1
            #G_loss = gen_loss(D_fake, y_fake, y)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            #compare_images(X, Y, 0)
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

NUM_EPOCHS=100
BCE = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()


initial_batch_size = 4
max_batch_size = 64
batch_size_increase_step = 4
loss_threshold = 0.1  # Example threshold to increase the batch size

# Create DataLoader with initial batch size
# Convert NumPy arrays to PyTorch tensors
x_tensor = torch.tensor(x_np)
y_tensor = torch.tensor(y_np)

dataset = TensorDataset(x_tensor, y_tensor)


train_loader = DataLoader(dataset, batch_size=initial_batch_size, shuffle=True)


# Function to adjust batch size dynamically
def adjust_batch_size(train_loader, current_batch_size, loss):
    if loss < loss_threshold and current_batch_size < max_batch_size:
        current_batch_size = min(current_batch_size + batch_size_increase_step, max_batch_size)
        train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
    return train_loader, current_batch_size

# Training loop with dynamic BGD
import torch
from torch.utils.data import DataLoader, TensorDataset

def train_dynamic_bgd(generator, discriminator, train_dataset, val_dataset, gen_opt, disc_opt, 
                      bce_loss, l1_loss, device, num_epochs=100, L1_LAMBDA=100, 
                      initial_batch_size=4, max_batch_size=64, batch_size_increase_step=4, loss_threshold=0.1):
   
    
    def adjust_batch_size(train_loader, current_batch_size, loss):
        if loss < loss_threshold and current_batch_size < max_batch_size:
            current_batch_size = min(current_batch_size + batch_size_increase_step, max_batch_size)
            train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
        return train_loader, current_batch_size

    # Initial DataLoader with initial batch size
    train_loader = DataLoader(train_dataset, batch_size=initial_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=initial_batch_size, shuffle=False)

    current_batch_size = initial_batch_size

    for epoch in range(num_epochs):
        generator.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Training steps
            gen_opt.zero_grad()
            disc_opt.zero_grad()
            with torch.cuda.amp.autocast():
                y_fake = generator(x)
                D_real = discriminator(torch.cat([x, y], dim=1))
                D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
                D_fake = discriminator(torch.cat([x, y_fake.detach()], dim=1))
                D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

                D_loss.backward()
                disc_opt.step()

                y_fake = generator(x)
                D_fake = discriminator(torch.cat([x, y_fake], dim=1))
                G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
                L1 = l1_loss(y_fake, y) * L1_LAMBDA
                G_loss = G_fake_loss + L1

                G_loss.backward()
                gen_opt.step()

            # Adjust batch size dynamically based on the generator loss
            train_loader, current_batch_size = adjust_batch_size(train_loader, current_batch_size, G_loss.item())

        print(f'Epoch [{epoch}/{num_epochs}], Batch Size: {current_batch_size}, Generator Loss: {G_loss.item()}')
        
train_dynamic_bgd(
    generator=generator, 
    discriminator=discriminator, 
    train_dataset=train_dataset, 
    val_dataset=val_dataset, 
    gen_opt=gen_opt, 
    disc_opt=disc_opt, 
    bce_loss=bce_loss, 
    l1_loss=l1_loss, 
    device=device, 
    num_epochs=100, 
    L1_LAMBDA=100, 
    initial_batch_size=4, 
    max_batch_size=64, 
    batch_size_increase_step=4, 
    loss_threshold=0.1
)

# Example usage:
# Assuming you have already defined and initialized your generator, discriminator, train_dataset, val_dataset,
# gen_opt, disc_opt, bce_loss, l1_loss, and device, you can call the function as follows:

# train_dynamic_bgd(generator, discriminator, train_dataset, val_dataset, gen_opt, disc_opt, 
#                   bce_loss, l1_loss, device)


"""# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
batch_size = 1

import torch
from torch.utils.data import DataLoader, TensorDataset

# Assuming x and y are NumPy arrays
x_np = x[0:20]
y_np = y[0:20]

# Convert NumPy arrays to PyTorch tensors
x_tensor = torch.tensor(x_np)
y_tensor = torch.tensor(y_np)

# Create a PyTorch dataset
dataset = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size)

generator.to(device)
discriminator.to(device)
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
real_label = 1.
fake_label = 0.
iters = 0
num_epochs = 100

print("Starting Training Loop...")

for epoch in range(num_epochs):
    # Create an iterator from the DataLoader
    it = iter(train_loader)
    input_img, real = next(it)

    # If you want to take only one image from the batch, you can index the tensors
    single_input_img = input_img[0].unsqueeze(0)
    single_real = real[0].unsqueeze(0)
    # For each batch in the dataloader
    for i, data in enumerate(train_loader):


        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        discriminator.zero_grad()

        # Format batch
        input = data[0].to(device)
        real = data[1].to(device)
        b_size = input.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        real_input = torch.cat([input, real], dim=1)
        real_output = discriminator(real_input)

        # Train with all-fake batch
        fake = generator(input)
        fake_input = torch.cat([input, fake.detach()], dim=1)
        fake_output = discriminator(fake_input)

        # Calculate D's loss
        errD = disc_loss(real_output, fake_output)
        errD.backward()
        disc_opt.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        fake_output = discriminator(torch.cat([input, fake], dim=1))
        errG = gen_loss(fake_output, fake, real)
        errG.backward()
        gen_opt.step()

        # Output training stats
        if i % 5 == 0:
            print(single_real.shape)
            compare_images(single_input_img, single_real, epoch)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Print progress
        print(f"Epoch [{epoch}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], "
              f"Generator Loss: {errG.item():.4f}, Discriminator Loss: {errD.item():.4f}", end='\r')

    # Print epoch summary
    print(f"Epoch [{epoch}/{num_epochs}], "
          f"Generator Loss: {sum(G_losses) / len(G_losses):.4f}, "
          f"Discriminator Loss: {sum(D_losses) / len(D_losses):.4f}")

# Define the optimizers
import torchvision.transforms as transforms

gen_opt = optim.Adam(generator.parameters(), lr=0.001)
disc_opt = optim.Adam(discriminator.parameters(), lr=0.001)

# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
batch_size = 1

# Assuming x and y are NumPy arrays
x_np = x[0:20]
y_np = y[0:20]

# Convert NumPy arrays to PyTorch tensors
x_tensor = torch.tensor(x_np)
y_tensor = torch.tensor(y_np)

# Create a PyTorch dataset
dataset = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size)


# Train the model
for epoch in range(100):
    for i, batch in enumerate(train_loader):
        input_img, real = batch
        input_img, real = input_img.to(device), real.to(device)

        # Train the generator
        with torch.no_grad():
            generated_output = generator(input_img)
            d_output_real = discriminator(torch.cat([input_img, real],dim=1))
            d_output_fake = discriminator(torch.cat([input_img, generated_output],dim=1))

        generated_output.requires_grad = True
        g_loss = gen_loss(d_output_fake, generated_output, real)
        g_loss.backward()
        gen_opt.step()

        # Train the discriminator
        d_output_real = discriminator(torch.cat([input_img, real],dim=1))
        d_output_fake = discriminator(torch.cat([input_img, generated_output],dim=1))

        d_loss = disc_loss(d_output_real, d_output_fake)
        d_loss.backward()
        disc_opt.step()

        if(i%5==0):
            compare_images(input_img,real,epoch)

        print(f"Epoch: {epoch+1}/{10}, Batch: {i+1}/{len(train_loader)}")

    print(f"Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}")
"""

# Save the model
torch.save(generator.state_dict(), model_name)

'''
x_tr=x[1200:1220]
y_tr=y[1200:1220]
from IPython import display
for i in range(20):
    display.display(display.HTML(f'<h3>Test Input {i+1}</h3>'))
    plt.figure(figsize=(15,5))
    plt.subplot(1,4,1)
    plt.imshow(x_tr[i,:,:,0],cmap='gray')
    plt.axis(False)
    plt.title('Temp')

    plt.subplot(1,4,2)
    plt.imshow(x_tr[i,:,:,1],cmap='gray')
    plt.axis(False)
    plt.title('Disp')

    plt.subplot(1,4,3)
    res=generator(tf.expand_dims(x_tr[i],axis=0))
    plt.imshow(res[0],cmap='gray')
    plt.axis(False)
    plt.title('Generated')

    plt.subplot(1,4,4)
    plt.imshow(y_tr[i],cmap='gray')
    plt.axis(False)
    plt.title('Ground Truth')
    plt.savefig(f'comparison_plot_1200_img_batch_size_64_img{i+1}.png')
    plt.show()

res=generator(tf.expand_dims(x[1499],axis=0))
plt.imshow(res[0],cmap='gray')




'''














