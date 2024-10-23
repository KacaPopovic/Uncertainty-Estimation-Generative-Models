"""
This file trains a GAN on the Fashion MNIST dataset and generates results.
"""

# Generator definition
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os

# Generator definition
class Generator(nn.Module):
    def __init__(self, noise_channels, image_channels, gen_features):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: Z (noise vector)
            nn.ConvTranspose2d(noise_channels, gen_features * 8, 4, 1, 0, bias=False),  # 1x1 -> 4x4
            nn.BatchNorm2d(gen_features * 8),
            nn.ReLU(True),

            # State size. (gen_features*8) x 4 x 4
            nn.ConvTranspose2d(gen_features * 8, gen_features * 4, 4, 2, 1, bias=False),  # 4x4 -> 8x8
            nn.BatchNorm2d(gen_features * 4),
            nn.ReLU(True),

            # State size. (gen_features*4) x 8 x 8
            nn.ConvTranspose2d(gen_features * 4, gen_features * 2, 4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.BatchNorm2d(gen_features * 2),
            nn.ReLU(True),

            # State size. (gen_features*2) x 16 x 16
            nn.ConvTranspose2d(gen_features * 2, gen_features, 4, 2, 1, bias=False),  # 16x16 -> 32x32
            nn.BatchNorm2d(gen_features),
            nn.ReLU(True),

            # State size. (gen_features) x 32 x 32
            nn.ConvTranspose2d(gen_features, image_channels, 4, 2, 1, bias=False),  # 32x32 -> 64x64
            nn.Tanh()  # Use Tanh to get output in range [-1, 1]
            # Final state size: (image_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self, image_channels, disc_features):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (image_channels) x 64 x 64
            nn.Conv2d(image_channels, disc_features, 4, 2, 1, bias=False),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),

            # State size. (disc_features) x 32 x 32
            nn.Conv2d(disc_features, disc_features * 2, 4, 2, 1, bias=False),  # 32x32 -> 16x16
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # State size. (disc_features*2) x 16 x 16
            nn.Conv2d(disc_features * 2, disc_features * 4, 4, 2, 1, bias=False),  # 16x16 -> 8x8
            nn.BatchNorm2d(disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # State size. (disc_features*4) x 8 x 8
            nn.Conv2d(disc_features * 4, disc_features * 8, 4, 2, 1, bias=False),  # 8x8 -> 4x4
            nn.BatchNorm2d(disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # State size. (disc_features*8) x 4 x 4
            nn.Conv2d(disc_features * 8, 1, 4, 1, 0, bias=False),  # 4x4 -> 1x1
            nn.Sigmoid()  # Use Sigmoid to get a probability score between 0 and 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


# Initialize the weights_CIFAR10
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class GAN(nn.Module):
    def __init__(self, ngpu, gen, disc):
        super(GAN, self).__init__()
        self.ngpu = ngpu
        self.generator = gen
        self.discriminator = disc

    def forward(self, noise):
        if noise.is_cuda and self.ngpu > 1:
            fake_data = nn.parallel.data_parallel(self.generator, noise, range(self.ngpu))
            disc_fake = nn.parallel.data_parallel(self.discriminator, fake_data, range(self.ngpu))
        else:
            fake_data = self.generator(noise)
            disc_fake = self.discriminator(fake_data).unsqueeze(1)

        return disc_fake

    def generate_image(self, noise):
        if noise.is_cuda and self.ngpu > 1:
            fake_data = nn.parallel.data_parallel(self.generator, noise, range(self.ngpu))
        else:
            fake_data = self.generator(noise)

        return fake_data

    def load_generator_state_dict(self, state_dict):
        self.generator.load_state_dict(state_dict)

    def load_discriminator_state_dict(self, state_dict):
        self.discriminator.load_state_dict(state_dict)

    def save_generator_state_dict(self, path):
        torch.save(self.generator.state_dict(), path)

    def save_discriminator_state_dict(self, path):
        torch.save(self.discriminator.state_dict(), path)

    def freeze_except_last_generator_layer(self):
        # Freeze all layers in the generator except the last convolutional layer
        for name, param in self.generator.named_parameters():
            if name.startswith('main.12'):
                continue
            param.requires_grad = False

        # Freeze all layers in the discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False


# Training loop
def main():
    # Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0002
    EPOCHS = 25
    noise_channels = 100
    image_channels = 1  # For MNIST, the image channels are 1 (grayscale)
    gen_features = 64
    disc_features = 64

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transform
    transform = transforms.Compose([
        transforms.Resize(64),  # Resize MNIST to 64x64 for this model
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create generator and discriminator models
    netG = Generator(noise_channels, image_channels, gen_features).to(device)
    netD = Discriminator(image_channels, disc_features).to(device)

    # Apply the weights_init function to randomly initialize all weights_CIFAR10
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss function (Binary Cross Entropy)
    criterion = nn.BCELoss()

    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Fixed noise for consistent evaluation of generator performance
    fixed_noise = torch.randn(64, noise_channels, 1, 1, device=device)

    # Create directory for saving images
    os.makedirs('../experiments/generated_images', exist_ok=True)

    # Training Loop
    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader, 0):
            # Get real images and create real labels
            real_images = data[0].to(device)
            b_size = real_images.size(0)
            real_labels = torch.full((b_size,), 1., device=device)
            fake_labels = torch.full((b_size,), 0., device=device)

            # Train the discriminator on real images
            netD.zero_grad()
            output = netD(real_images)
            lossD_real = criterion(output, real_labels)
            lossD_real.backward()

            # Generate fake images
            noise = torch.randn(b_size, noise_channels, 1, 1, device=device)
            fake_images = netG(noise)

            # Train the discriminator on fake images
            output = netD(fake_images.detach())
            lossD_fake = criterion(output, fake_labels)
            lossD_fake.backward()
            optimizerD.step()

            # Train the generator (want discriminator to think fake images are real)
            netG.zero_grad()
            output = netD(fake_images)
            lossG = criterion(output, real_labels)
            lossG.backward()
            optimizerG.step()

            # Print loss values every 100 steps
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{i}/{len(dataloader)}] "
                      f"Loss D: {lossD_real.item() + lossD_fake.item():.4f}, Loss G: {lossG.item():.4f}")

        # Save generated images at the end of each epoch
        with torch.no_grad():
            fake_images = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake_images, f'../experiments/generated_images/epoch_{epoch}.png', normalize=True)

    print("Training complete.")


if __name__ == "__main__":
    main()


