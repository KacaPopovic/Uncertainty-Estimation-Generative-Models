"""
This file will train the models created in the model file and generate results
"""
# import all libraries
import torch
import torchvision
from torch import nn
from torch import optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os


# from model import Discriminator, Generator  # Ensure you have these classes defined or imported correctly

# Define the Generator and Discriminator classes if not imported
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, noise_channels, image_channels, features):
        super(Generator, self).__init__()
        """
        In this function the generator model will be defined with all of it layers.
        The generator model uses 4 ConvTranspose blocks. Each block containes
        a ConvTranspose2d, BatchNorm2d and ReLU activation.
        """
        # define the model
        self.model = nn.Sequential(
            # Transpose block 1
            nn.ConvTranspose2d(noise_channels, features*16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            # Transpose block 2
            nn.ConvTranspose2d(features*16, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.ReLU(),

            # Transpose block 3
            nn.ConvTranspose2d(features*8, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.ReLU(),

            # Transpose block 4
            nn.ConvTranspose2d(features*4, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(),

            # Last transpose block (different)
            nn.ConvTranspose2d(features*2, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, image_channels, features):
        super(Discriminator, self).__init__()
        """
        This function will define the Discriminator model with all the layers needed.
        The model has 5 Conv blocks. The blocks have Conv2d, BatchNorm and LeakyReLU activation.
        """
        # define the model
        self.model = nn.Sequential(
            # define the first Conv block
            nn.Conv2d(image_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Conv block 2
            nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2),

            # Conv block 3
            nn.Conv2d(features*2, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2),

            # Conv block 4
            nn.Conv2d(features*4, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2),

            # Conv block 5 (different)
            nn.Conv2d(features*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

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
            if name.startswith('model.11'):
                continue
            param.requires_grad = False

        # Freeze all layers in the discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False

def main():

    # Define the hyperparameters and variables
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 256
    IMAGE_SIZE = 28  # Changed to 28 to match Fashion MNIST
    EPOCHS = 250
    image_channels = 1
    noise_channels = 256
    gen_features = 64
    disc_features = 64

    # Set everything to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transform
    data_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Load the dataset
    dataset = MNIST(root="dataset/", train=True, transform=data_transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize Generator and Discriminator
    gen_model = Generator(noise_channels, image_channels, gen_features).to(device)
    disc_model = Discriminator(image_channels, disc_features).to(device)

    # Setup optimizers for both models
    gen_optimizer = optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Define the loss function
    criterion = nn.BCELoss()

    # Make both models train
    gen_model.train()
    disc_model.train()

    # Define labels for fake images and real images for the discriminator
    fake_label = 0
    real_label = 1

    # Define a fixed noise
    fixed_noise = torch.randn(64, noise_channels, 1, 1).to(device)

    # Make the writers for TensorBoard
    writer_real = SummaryWriter(f"runs/fashion/test_real")
    writer_fake = SummaryWriter(f"runs/fashion/test_fake")

    # Define a step
    step = 0

    print("Start training...")

    # Ensure directories for saving models exist
    model_dir = "models/weights_CIFAR10"
    os.makedirs(model_dir, exist_ok=True)

    # Loop over all epochs and all data
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(dataloader):
            # Set the data to device
            data = data.to(device)

            # Get the batch size
            batch_size = data.shape[0]

            # ---------------------
            # Train the Discriminator
            # ---------------------
            disc_model.zero_grad()
            # Labels for real images: use label smoothing (0.9 instead of 1.0)
            label = (torch.ones(batch_size) * 0.9).to(device)
            output = disc_model(data).reshape(-1)
            real_disc_loss = criterion(output, label)
            D_x = output.mean().item()

            # Train on fake images
            noise = torch.randn(batch_size, noise_channels, 1, 1).to(device)
            fake = gen_model(noise)
            # Labels for fake images: use label smoothing (0.1 instead of 0.0)
            label = (torch.ones(batch_size) * 0.1).to(device)
            output = disc_model(fake.detach()).reshape(-1)
            fake_disc_loss = criterion(output, label)

            # Calculate the final discriminator loss
            disc_loss = real_disc_loss + fake_disc_loss

            # Backprop and optimize
            disc_loss.backward()
            disc_optimizer.step()

            # -----------------
            # Train the Generator
            # -----------------
            gen_model.zero_grad()
            # Labels for generator: want discriminator to think these are real
            label = torch.ones(batch_size).to(device)
            output = disc_model(fake).reshape(-1)
            gen_loss = criterion(output, label)
            # Backprop and optimize
            gen_loss.backward()
            gen_optimizer.step()

            # -----------------
            # Logging
            # -----------------
            if batch_idx % 50 == 0:
                step += 1

                # Print losses
                print(
                    f"Epoch: {epoch}/{EPOCHS} ===== Batch: {batch_idx}/{len(dataloader)} ===== Disc loss: {disc_loss:.4f} ===== Gen loss: {gen_loss:.4f} ===== D(x): {D_x:.4f} ===== D(G(z)): {output.mean().item():.4f}"
                )

                ### Test the model
                with torch.no_grad():
                    # Generate fake images
                    fake_images = gen_model(fixed_noise)
                    # Make grid in the TensorBoard
                    img_grid_real = torchvision.utils.make_grid(data[:40], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake_images[:40], normalize=True)

                    # Write the images in TensorBoard
                    writer_real.add_image("Real images", img_grid_real, global_step=step)
                    writer_fake.add_image("Generated images", img_grid_fake, global_step=step)

        # Save the models at the end of each epoch
        torch.save(gen_model.state_dict(), os.path.join(model_dir, f'gen_epoch_{epoch}.pth'))
        torch.save(disc_model.state_dict(), os.path.join(model_dir, f'disc_epoch_{epoch}.pth'))

        print(f"Saved models for epoch {epoch}")

    # Close TensorBoard writers
    writer_real.close()
    writer_fake.close()

    print("Training complete.")

if __name__ == "__main__":
    main()

