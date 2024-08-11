from laplace import Laplace
from GAN_cifar import Generator, Discriminator, GAN
from torch.utils.data import DataLoader, random_split
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_map_model(ngpu=1, device='cpu'):

    generator = Generator(ngpu).to(device)
    discriminator = Discriminator(ngpu).to(device)
    gan = GAN(ngpu, generator, discriminator).to(device)
    # Base directory for weights
    weights_dir = r'D:\Uncertainty-Estimation-Generative-Models\models\weights'

    # Load the weights
    gan.load_generator_state_dict(torch.load(os.path.join(weights_dir, 'netG_epoch_24.pth'), map_location=device))
    gan.load_discriminator_state_dict(torch.load(os.path.join(weights_dir, 'netD_epoch_24.pth'), map_location=device))

    return gan

class NoiseDataset(Dataset):
    def __init__(self, num_samples, noise_dim, device):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            noise_dim (int): Size of the latent vector (noise vector).
            device (torch.device): Device to create the noise tensors on (e.g., 'cpu' or 'cuda').
        """
        self.num_samples = num_samples
        self.noise_dim = noise_dim
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Generate a random noise vector
        noise = torch.randn(self.noise_dim, 1, 1, device=self.device)
        # Assign label 0 to the noise vector
        label = torch.tensor(0, device=self.device)
        return noise, label

def plot_generated_images(images, n_row, n_col):

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    for i, ax in enumerate(axes.flat):
        img = images[i].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # Convert from CHW to HWC format
        img = (img + 1) / 2.0  # Denormalize the image (from [-1, 1] to [0, 1])
        ax.imshow(img)
        ax.axis('off')
    plt.show()

def main():

    # input noise dimension
    noise_dim = 100

    # checking the availability of cuda devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the dataset
    num_samples = 10000
    full_dataset = NoiseDataset(num_samples, noise_dim, device)

    # Define the split ratio
    train_size = int(0.8 * len(full_dataset))  # 80% for training
    val_size = len(full_dataset) - train_size  # the rest for validation

    # Split the dataset into training and validation subsets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Load the combined GAN model
    model = load_map_model(ngpu=1, device='cpu')

    # Check output of generator

    with torch.no_grad():
        noise = torch.randn(8, noise_dim, 1, 1, device=device)  # Generate a batch of 8 noise vectors
        generated_images = model.generate_image(noise)

    plot_generated_images(generated_images, n_row=2, n_col=4)


    # Set subset of weights to just last layer of generator by freezing everything else

    model.freeze_except_last_generator_layer()

    # Verify the requires_grad attribute
    #for name, param in model.generator.named_parameters():
    #    print(f"Generator Parameter: {name}, requires_grad: {param.requires_grad}")

    #for name, param in model.discriminator.named_parameters():
    #    print(f"Discriminator Parameter: {name}, requires_grad: {param.requires_grad}")

    # Generate laplace class
    bayesian_model = Laplace(model, "regression",
                 subset_of_weights="all",
                 hessian_structure="diag")

    # Fit bayesian model to data - approximate covariance matrices of gaussians

    bayesian_model.fit(train_loader)

if __name__ == '__main__':
    main()