import torch
from torch.utils.data import Dataset, DataLoader
from laplace import Laplace
from GAN_FMNIST import Generator, Discriminator, GAN
from visualization import *
from per_class_helpers import *


class NoiseDataset(Dataset):
    """This class represents a noise dataset for a generative adversarial network (GAN).

        :param num_samples: Number of samples in the dataset.
        :type num_samples: int
        :param noise_dim: Size of the latent vector (noise vector).
        :type noise_dim: int
        :param device: Device to create the noise tensors on (e.g., 'cpu' or 'cuda').
        :type device: torch.device
    """
    def __init__(self, num_samples: int, noise_dim: int, device: device, conditional=False):
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

    def __getitem__(self, index: int) -> (torch.Tensor, float):
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

    :param weights_path: The path to the directory containing the weights_CIFAR10 for the map model.
    :type weights_path: str
    :param noise_dim: The dimensionality of the noise for the generator.
    :type noise_dim: int
    :param device: The device to use for training and inference.
    :type device: str
    """
    def __init__(self, weights_path, noise_dim, device, conditional=False):
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
        Loads Maximum Aposteriori model trained via empirical risk minimization.

        :param ngpu: Number of GPUs available.
        :type ngpu: int
        """
        # Create MAP model
        generator = Generator(256,1,64).to(self.device)
        discriminator = Discriminator(1,64).to(self.device)
        self.map_model = GAN(ngpu, generator, discriminator).to(self.device)

        # Load weights_CIFAR10
        self.map_model.load_generator_state_dict(torch.load(
            os.path.join(self.weights_dir, 'netG_epoch_137.pth'), map_location=self.device, weights_only=True))
        self.map_model.load_discriminator_state_dict(
            torch.load(os.path.join(self.weights_dir, 'netD_epoch_137.pth'),
                       map_location=self.device, weights_only=True))

        self.map_model.freeze_except_last_generator_layer()

    def approximate_bayesian_model(self, data_loader, likelihood, subset_of_weights, hessian_structure):
        """
        Approximates the Bayesian model using Laplace method.

        :param data_loader: The data loader to load the data set.
        :type data_loader: DataLoader
        :param likelihood: The likelihood for the Laplace transformation.
        :type likelihood: str
        :param subset_of_weights: The subset of weights_CIFAR10 to include in the Laplace transformation.
        :type subset_of_weights: str
        :param hessian_structure: The structure of the Hessian matrix for the Laplace transformation.
        :type hessian_structure: str
        """
        self.laplace_model = Laplace(self.map_model, likelihood, subset_of_weights, hessian_structure)
        self.laplace_model.fit(data_loader, progress_bar=True)

    def load_laplace_model(self, weigts_dir, likelihood, subset_of_weights, hessian_structure):
        """
        Loads the Laplace model.

        :param weigts_dir: The directory where the weights_CIFAR10 for the Laplace model are stored.
        :type weigts_dir: str
        :param likelihood: The likelihood function for the Laplace transformation.
        :type likelihood: str
        :param subset_of_weights: The subset of weights_CIFAR10 to include in the Laplace transformation.
        :type subset_of_weights: str
        :param hessian_structure: The structure of the Hessian matrix for the Laplace transformation.
        :type hessian_structure: str
        """
        self.laplace_model = Laplace(self.map_model, likelihood, subset_of_weights, hessian_structure)
        self.laplace_model.load_state_dict(torch.load(weigts_dir, map_location=torch.device('cpu')))


def main():

    # Define parameters of the simulation

    conditional = False
    train = True
    noise_dim = 256
    num_samples = 60000
    weights_MAP = r'../models/MAP_models/weights_FashionMNIST'
    weights_laplace = "../models/laplace_models/final_fashion_MNIST.bin"

    # checking the availability of cuda devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading or training of laplace model

    laplace = LaplaceTransformation(weights_MAP, noise_dim, device, conditional=conditional)
    laplace.load_map_model()

    if train:
        full_dataset = NoiseDataset(num_samples, noise_dim, device, conditional)
        train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=2)
        laplace.approximate_bayesian_model(train_loader, "classification", "all", "diag")
        state_dict = laplace.laplace_model.state_dict()
        torch.save(state_dict, weights_laplace)
    else:
        laplace.load_laplace_model(weights_laplace, "classification", "all", "diag")
    model = laplace.laplace_model

    # Plotting uncertainty

    epistemic_uncertainties = []
    epistemic_uncertainty_maps = []
    epistemic_uncertainty_maps_for_plot = []
    save_path_plots = '../experiments/uncertainties_images'

    for i in range(20):
        if conditional:
            T2 = 256
            noise = torch.randn(T2, noise_dim, 1, 1, device=device)
            label = torch.full((T2,), i, device=device)
            batch_size = 128  # Define the batch size  # Adjust according to your noise dimension
            n_batches = T2 // batch_size  # Calculate the number of batches
            remainder = T2 % batch_size  # If T2 is not perfectly divisible by batch_size

            # Initialize containers for results
            all_images = []
            all_py = []

            # Loop over batches
            for batch_idx in range(n_batches + int(remainder > 0)):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, T2)

                # Get the current batch of noise and labels
                noise_batch = noise[start_idx:end_idx]
                label_batch = label[start_idx:end_idx]

                # Generate images for this batch
                image_map = laplace.map_model.generate_image(noise_batch, label_batch)

                # Perform the model prediction for this batch
                _, images_batch = model((noise_batch, label_batch), pred_type="nn", link_approx="mc",
                                               n_samples=100)

                all_images.append(images_batch)

            # Concatenate the results across all batches
            images = torch.cat(all_images, dim=1)  # Adjust the dimension accordingly
            (epistemic_uncertainty_value,
             epistemic_uncertainty_map,
             epistemic_uncertainty_map_for_plot) = compute_epistemic_uncertainty_per_class(images)
            epistemic_uncertainties.append(epistemic_uncertainty_value)
            epistemic_uncertainty_maps.append(epistemic_uncertainty_map)
            epistemic_uncertainty_maps_for_plot.append(epistemic_uncertainty_map_for_plot)

            print(f"Label {i}: DONE")
            print(f'Uncertainty: {epistemic_uncertainty_value}')
            plot_map_image_uncertainty_maps(
                image_map[0,:,:,:],
                epistemic_uncertainty_map_for_plot,
                epistemic_uncertainty_value,
                i, save_path_plots)
        else:
            noise = torch.randn(noise_dim, 1, 1, device=device).unsqueeze(0)
            image_map = laplace.map_model.generate_image(noise)
            py, images = model(noise, pred_type="nn", link_approx="mc", n_samples=100)
            images = images.squeeze(1)  # Removes the second dimension
            plot_generated_images(images, n_row=4, n_col=5, rgb = False)
            plot_MAP_and_uncertainty(images, image_map, rgb=False)
            plot_images_and_variance(model, noise_dim, device)
            #plot_MAP_and_uncertainty(images, image_map)


if __name__ == '__main__':
    main()
