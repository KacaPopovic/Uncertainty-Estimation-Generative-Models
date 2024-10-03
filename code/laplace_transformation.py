#from GAN_cifar import Generator, Discriminator, GAN
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from laplace import Laplace
from GAN_MNIST import Generator, Discriminator, GAN


def plot_generated_images(images: torch.Tensor, n_row: int, n_col: int, rgb: bool = True) -> None:
    """
    Plots the generated images in a grid format.

    :param images: A tensor of generated images with shape (N, C, H, W).
                   - For RGB images, C should be 3.
                   - For grayscale images, C should be 1.
    :param n_row: The number of rows in the plot grid.
    :param n_col: The number of columns in the plot grid.
    :param rgb:   If True, treats images as RGB; if False, treats images as grayscale.
    :return:      None
    """
    # Validate the number of images
    N, C, H, W = images.shape
    if rgb and C != 3:
        raise ValueError(f"Expected 3 channels for RGB images, but got {C} channels.")
    if not rgb and C != 1:
        raise ValueError(f"Expected 1 channel for grayscale images, but got {C} channels.")

    # Determine the total number of images to display
    total_images = n_row * n_col
    if total_images > N:
        raise ValueError(f"Grid size {n_row}x{n_col} is larger than the number of images ({N}).")

    # Create subplots
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 2, n_row * 2))
    axes = axes.flatten()  # Flatten in case of single row or column

    for i in range(total_images):
        ax = axes[i]
        img = images[i].detach().cpu().numpy()

        if rgb:
            # Convert from CHW to HWC format
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)
            # Denormalize the image (assuming the image was normalized to [-1, 1])
            img = (img + 1) / 2.0
            img = np.clip(img, 0, 1)  # Ensure the values are within [0, 1]
            ax.imshow(img)
        else:
            # Convert from CHW to HW format
            img = img.squeeze(0)  # (H, W)
            # Denormalize the image (assuming the image was normalized to [-1, 1])
            img = (img + 1) / 2.0
            img = np.clip(img, 0, 1)  # Ensure the values are within [0, 1]
            ax.imshow(img, cmap='gray')

        ax.axis('off')  # Hide axis

    # If there are more subplots than images, hide the extra axes
    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_MAP_and_uncertainty(images: torch.Tensor, image_map: torch.Tensor, rgb=True):
    """
    Plot the MAP image and its uncertainty (variance).

    Args:
        images (torch.Tensor): Generated images tensor of shape (N, C, H, W).
        image_map (torch.Tensor): MAP image tensor of shape (C, H, W).
        rgb (bool): If True, plot RGB variance; else, plot grayscale variance.

    Returns:
        tuple: (variance_rgb, variance_grayscale, total_variance) where variance_rgb is None if rgb=False
    """
    # Compute variance across the N samples
    variance = torch.var(images, dim=0, unbiased=False)

    if rgb:
        # Process RGB variance
        variance_rgb = variance.permute(1, 2, 0).cpu().detach().numpy()  # (H, W, C)
        variance_rgb = normalize_image(variance_rgb)

        # Compute grayscale variance by summing across color channels
        variance_grayscale = variance_rgb.sum(axis=-1)
    else:
        # Process grayscale variance
        variance_grayscale = variance.squeeze().cpu().detach().numpy()  # (H, W)
        variance_grayscale = normalize_image(variance_grayscale)
        variance_rgb = None  # No RGB variance in grayscale case

    # Compute the total variance (sum of all pixel variances)
    total_variance = variance.sum().item()

    # Normalize the image_map
    if rgb:
        image_map_np = image_map.squeeze().permute(1, 2, 0).cpu().detach().numpy()  # (H, W, C)
    else:
        image_map_np = image_map.squeeze().cpu().detach().numpy()  # (H, W)
    image_map_np = normalize_image(image_map_np)

    # Determine the number of subplots based on the mode
    if rgb:
        n_subplots = 3
    else:
        n_subplots = 2

    # Create subplots
    fig, axes = plt.subplots(1, n_subplots, figsize=(15, 5))

    # Plot the MAP image
    if rgb:
        axes[0].imshow(image_map_np)
    else:
        axes[0].imshow(image_map_np, cmap='gray')
    axes[0].set_title('Generated Image Map')
    axes[0].axis('off')  # Hide axis

    # Plot variance
    if rgb:
        # Plot the RGB variance
        axes[1].imshow(variance_rgb)
        axes[1].set_title('Variance of Generated Images (RGB)')
    else:
        # Plot the grayscale variance
        axes[1].imshow(variance_grayscale, cmap='gray')
        axes[1].set_title('Variance of Generated Images (Grayscale)')
    axes[1].axis('off')  # Hide axis

    # If RGB, plot the grayscale variance as the third subplot
    if rgb:
        axes[2].imshow(variance_grayscale, cmap='gray')
        axes[2].set_title('Variance of Generated Images (Grayscale)')
        axes[2].axis('off')  # Hide axis

    # Adjust layout to make space for figtext
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for text

    # Add the total variance text slightly above the bottom
    plt.figtext(
        0.5,
        0.02,  # Adjusted y-position to move text upward
        f'Total Variance: {total_variance:.4f}',
        ha='center',
        fontsize=16,  # Increased font size
        color='black',  # Black text
        backgroundcolor='white'  # White background without a box
    )

    plt.show()

    return variance_rgb, variance_grayscale, total_variance


def normalize_image(image):
    # Normalize the image to the range [0, 1]
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min)


class NoiseDataset(Dataset):
    """This class represents a noise dataset for a generative adversarial network (GAN).

        :param num_samples: Number of samples in the dataset.
        :type num_samples: int
        :param noise_dim: Size of the latent vector (noise vector).
        :type noise_dim: int
        :param device: Device to create the noise tensors on (e.g., 'cpu' or 'cuda').
        :type device: torch.device
    """
    def __init__(self, num_samples: int, noise_dim: int, device: torch.device, conditional = False):
        """
        Initializes an instance of the NoiseDataset class.
        """
        self.num_samples = num_samples
        self.noise_dim = noise_dim
        self.device = device
        self.conditional = conditional

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, index:int) -> (torch.Tensor, float):
        """
        Retrieves the noise vector and label tensor associated with the given index.

        :param index: An integer representing the index of the noise vector and label tensor to retrieve.
        :type index: int
        :return: A tuple containing the noise vector and label tensor associated with the given index.
        The noise vector is of type torch.Tensor, and the label tensor is of type float.
        The label is always 1 since we use NoiseDataset to generate fake images, that we want to be classified as real.
        """
        # Generate a random noise vector
        noise = torch.randn(self.noise_dim, 1, 1, device=self.device)
        # Assign label 0 to the noise vector
        label = torch.tensor(1, device=self.device).unsqueeze(0)
        if self.conditional:
            class_label = torch.randint(0, 10, (1,), device=self.device)
            return noise, label.float(), class_label
        else:
            return noise, label.float()


class LaplaceTransformation:
    """This class represents Laplace transformation in the context of generative adversarial networks (GANs).

    :param weights_path: The path to the directory containing the weights for the map model.
    :type weights_path: str
    :param noise_dim: The dimensionality of the noise for the generator.
    :type noise_dim: int
    :param device: The device to use for training and inference.
    :type device: str
    """
    def __init__(self, weights_path, noise_dim, device, conditional = False):
        """
        Initializes an instance of the LaplaceTransformation class.
        """
        self.noise_dim = noise_dim
        self.device = device
        self.weights_dir = weights_path
        self.map_model = None
        self.laplace_model = None
        self.conditional = conditional

    def load_map_model(self, ngpu=1):
        """
        Loads the map model with generator and discriminator for training GAN.

        :param ngpu: Number of GPUs available.
        :type ngpu: int
        """
        # if not mnist, GEN, DISC should have param ngpu
        generator = Generator().to(self.device)
        discriminator = Discriminator().to(self.device)
        self.map_model = GAN(ngpu, generator, discriminator).to(self.device)
        # Load the weights
        self.map_model.load_generator_state_dict(torch.load(
            os.path.join(self.weights_dir, 'netG_epoch_59.pth'), map_location=self.device, weights_only=True))
        self.map_model.load_discriminator_state_dict(
            torch.load(os.path.join(self.weights_dir, 'netD_epoch_59.pth'),
                       map_location=self.device, weights_only=True))

        self.map_model.freeze_except_last_generator_layer()

    def approximate_bayesian_model(self, data_loader, likelihood, subset_of_weights, hessian_structure):
        """
        Approximates the Bayesian model.

        :param data_loader: The data loader to load the data set.
        :type data_loader: DataLoader
        :param likelihood: The likelihood for the Laplace transformation.
        :type likelihood: str
        :param subset_of_weights: The subset of weights to include in the Laplace transformation.
        :type subset_of_weights: str
        :param hessian_structure: The structure of the Hessian matrix for the Laplace transformation.
        :type hessian_structure: str
        """
        self.laplace_model = Laplace(self.map_model, likelihood, subset_of_weights, hessian_structure)
        self.laplace_model.fit(data_loader, progress_bar=True)

    def load_laplace_model(self, weigts_dir, likelihood, subset_of_weights, hessian_structure):
        """
        Loads the Laplace model.

        :param weigts_dir: The directory where the weights for the Laplace model are stored.
        :type weigts_dir: str
        :param likelihood: The likelihood function for the Laplace transformation.
        :type likelihood: str
        :param subset_of_weights: The subset of weights to include in the Laplace transformation.
        :type subset_of_weights: str
        :param hessian_structure: The structure of the Hessian matrix for the Laplace transformation.
        :type hessian_structure: str
        """
        self.laplace_model = Laplace(self.map_model, likelihood, subset_of_weights, hessian_structure)
        self.laplace_model.load_state_dict(torch.load(weigts_dir))


def main():

    conditional = False

    # input noise dimension
    noise_dim = 256

    # checking the availability of cuda devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the dataset
    num_samples = 60000
    full_dataset = NoiseDataset(num_samples, noise_dim, device, conditional)

    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=2)

    weights_dir = r'D:\Uncertainty-Estimation-Generative-Models\models\weights_FashionMNIST'

    laplace = LaplaceTransformation(weights_dir, noise_dim, device, conditional=conditional)
    laplace.load_map_model()

    # Check output of generator
    with torch.no_grad():
        noise = torch.randn(8, noise_dim, 1, 1, device=device)  # Generate a batch of 8 noise vectors
        if conditional:
            labels = torch.randint(0, 10, (8,), device=device)
            generated_images = laplace.map_model.generate_image(noise, labels)
        else:
            generated_images = laplace.map_model.generate_image(noise)
    plot_generated_images(generated_images,4,2, rgb=False)

    laplace.approximate_bayesian_model(train_loader, "classification", "all", "full")
    weights_dir = "laplace_models/MNIST_new_full.bin"
    state_dict = laplace.laplace_model.state_dict()
    torch.save(state_dict, weights_dir)
    #laplace.load_laplace_model(weights_dir, "classification", "all", "diag")
    model = laplace.laplace_model

    for i in range(20):
        noise = torch.randn(noise_dim, 1, 1, device=device).unsqueeze(0)
        if conditional:
            label = torch.randint(0, 10, (1,), device=device)
            image_map = laplace.map_model.generate_image(noise, label)
            py, images = model((noise,label), pred_type="nn", link_approx="mc", n_samples=100)
        else:
            image_map = laplace.map_model.generate_image(noise)
            py, images = model(noise, pred_type="nn", link_approx="mc", n_samples=100)
        images = images.squeeze(1)  # Removes the second dimension
        plot_generated_images(images, n_row=4, n_col=5, rgb = False)
        plot_MAP_and_uncertainty(images, image_map, rgb = False)


if __name__ == '__main__':
    main()
