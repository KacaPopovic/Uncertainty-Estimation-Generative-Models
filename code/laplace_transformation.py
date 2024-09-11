from GAN_cifar import Generator, Discriminator, GAN
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from laplace import Laplace


def plot_generated_images(images: torch.Tensor, n_row: int, n_col: int) -> None:
    """
    Plots the generated images.

    :param images: A list of generated images.
    :param n_row: The number of rows in the plot grid.
    :param n_col: The number of columns in the plot grid.
    :return: None
    """
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    for i, ax in enumerate(axes.flat):
        img = images[i].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # Convert from CHW to HWC format
        img = (img + 1) / 2.0  # Denormalize the image (from [-1, 1] to [0, 1])
        ax.imshow(img)
        ax.axis('off')
    plt.show()


def plot_images_and_variance(images, variance_rgb, n_row=4, n_col=2):
    fig, axs = plt.subplots(n_row, n_col, figsize=(10, 10))

    # Flatten the axes array for easy iteration
    axs = axs.flatten()

    # Plot the images
    for i in range(n_row * n_col):
        if i < len(images):
            axs[i].imshow(images[i].cpu().detach().permute(1, 2, 0).numpy())
            axs[i].set_title(f'Image {i + 1}')
        else:
            # Plot the variance in the remaining subplot
            axs[i].imshow(variance_rgb)
            axs[i].set_title('Variance (RGB)')
        axs[i].axis('off')  # Hide the axis for better visualization

    plt.tight_layout()
    plt.show()

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
    def __init__(self, num_samples: int, noise_dim: int, device: torch.device):
        """
        Initializes an instance of the NoiseDataset class.
        """
        self.num_samples = num_samples
        self.noise_dim = noise_dim
        self.device = device

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
    def __init__(self, weights_path, noise_dim, device):
        """
        Initializes an instance of the LaplaceTransformation class.
        """
        self.noise_dim = noise_dim
        self.device = device
        self.weights_dir = weights_path
        self.map_model = None
        self.laplace_model = None

    def load_map_model(self, ngpu=1):
        """
        Loads the map model with generator and discriminator for training GAN.

        :param ngpu: Number of GPUs available.
        :type ngpu: int
        """
        generator = Generator(ngpu).to(self.device)
        discriminator = Discriminator(ngpu).to(self.device)
        self.map_model = GAN(ngpu, generator, discriminator).to(self.device)
        # Load the weights
        self.map_model.load_generator_state_dict(torch.load(
            os.path.join(self.weights_dir, 'netG_epoch_40.pth'), map_location=self.device, weights_only=True))
        self.map_model.load_discriminator_state_dict(
            torch.load(os.path.join(self.weights_dir, 'netD_epoch_40.pth'),
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

    # input noise dimension
    noise_dim = 100

    # checking the availability of cuda devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the dataset
    num_samples = 30000
    full_dataset = NoiseDataset(num_samples, noise_dim, device)

    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=2)

    weights_dir = r'D:\Uncertainty-Estimation-Generative-Models\models\weights'

    laplace = LaplaceTransformation(weights_dir, noise_dim, device)
    laplace.load_map_model()

    # Check output of generator
    with torch.no_grad():
        noise = torch.randn(8, noise_dim, 1, 1, device=device)  # Generate a batch of 8 noise vectors
        generated_images = laplace.map_model.generate_image(noise)

    #plot_generated_images(generated_images, n_row=2, n_col=4)
    #laplace.approximate_bayesian_model(train_loader, "classification", "all", "diag")
    weights_dir = "freezed_diag_classification_large.bin"
    #state_dict = laplace.laplace_model.state_dict()
    #torch.save(state_dict, weights_dir)
    laplace.load_laplace_model(weights_dir, "classification", "all", "diag")
    model = laplace.laplace_model

    for i in range(20):
        noise = torch.randn(100, 1, 1, device=device).unsqueeze(0)
        image_map = laplace.map_model.generate_image(noise)
        py, images = model(noise, pred_type="nn", link_approx="mc", n_samples=100)
        images = images.squeeze(1)  # Removes the second dimension
        new_images = torch.cat((image_map, images), dim=0)
        plot_generated_images(images, n_row=4, n_col=5)

        variance = torch.var(images, dim=0, unbiased=False)
        variance_rgb = variance.permute(1, 2, 0).cpu().detach().numpy()
        variance_rgb = normalize_image(variance_rgb)

        # Compute variance_grayscale
        variance_grayscale = variance_rgb.sum(axis=-1)
        variance_grayscale = normalize_image(variance_grayscale)

        # Compute the total variance (sum of all pixel variances)
        total_variance = variance.sum().item()

        # Normalize the image_map
        image_map_np = image_map.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        image_map_np = normalize_image(image_map_np)

        # Plotting the image_map, variance_rgb, and variance_grayscale side by side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the image_map
        axes[0].imshow(image_map_np)
        axes[0].set_title('Generated Image Map')
        axes[0].axis('off')  # Hide axis

        # Plot the variance_rgb
        axes[1].imshow(variance_rgb)
        axes[1].set_title('Variance of Generated Images (RGB)')
        axes[1].axis('off')  # Hide axis

        # Plot the variance_grayscale
        axes[2].imshow(variance_grayscale, cmap='gray')
        axes[2].set_title('Variance of Generated Images (Grayscale)')
        axes[2].axis('off')  # Hide axis

        # Add a text box with the total variance
        plt.figtext(0.5, 0.01, f'Total Variance: {total_variance:.4f}',
                    ha='center', fontsize=12, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

        plt.show()


if __name__ == '__main__':
    main()
