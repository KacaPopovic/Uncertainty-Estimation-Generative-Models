from GAN_cifar import Generator, Discriminator, GAN
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from laplace import Laplace


def plot_generated_images(images, n_row, n_col):

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    for i, ax in enumerate(axes.flat):
        img = images[i].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # Convert from CHW to HWC format
        img = (img + 1) / 2.0  # Denormalize the image (from [-1, 1] to [0, 1])
        ax.imshow(img)
        ax.axis('off')
    plt.show()


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


class LaplaceTransformation:
    def __init__(self, weights_path, noise_dim, device):
        self.noise_dim = noise_dim
        self.device = device
        self.weights_dir = weights_path
        self.map_model = None
        self.laplace_model = None

    def load_map_model(self, ngpu=1):
        generator = Generator(ngpu).to(self.device)
        discriminator = Discriminator(ngpu).to(self.device)
        self.map_model = GAN(ngpu, generator, discriminator).to(self.device)
        # Load the weights
        self.map_model.load_generator_state_dict(torch.load(
            os.path.join(self.weights_dir, 'netG_epoch_24.pth'), map_location=self.device, weights_only=True))
        self.map_model.load_discriminator_state_dict(
            torch.load(os.path.join(self.weights_dir, 'netD_epoch_24.pth'),
                       map_location=self.device, weights_only=True))

    def approximate_bayesian_model(self, data_loader, likelihood, subset_of_weights, hessian_structure):
        self.laplace_model = Laplace(self.map_model, likelihood, subset_of_weights, hessian_structure)
        self.laplace_model.fit(data_loader)


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
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    weights_dir = r'D:\Uncertainty-Estimation-Generative-Models\models\weights'

    laplace = LaplaceTransformation(weights_dir, noise_dim, device)
    laplace.load_map_model()

    # Check output of generator
    with torch.no_grad():
        noise = torch.randn(8, noise_dim, 1, 1, device=device)  # Generate a batch of 8 noise vectors
        generated_images = laplace.map_model.generate_image(noise)

    plot_generated_images(generated_images, n_row=2, n_col=4)
    laplace.approximate_bayesian_model(train_loader, "classification",
                                       "all", "diag")
    model = laplace.laplace_model
    state_dict = model.state_dict()
    torch.save(state_dict, "state_dict.bin")


if __name__ == '__main__':
    main()
