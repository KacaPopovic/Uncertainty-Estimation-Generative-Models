import os
import torch
import numpy as np
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")


def compute_epistemic_uncertainty_per_class(generated_images):
    """
    Computes the epistemic uncertainty of the generated images, as in paper "Quantifying Generative Model Uncertainty in
    Posterior Sampling Methods for Computational Imaging".
    """

    # T2 is the number of models (100)
    # T1 is the number of images per model (100)
    T2, T1, H, W, C = generated_images.shape

    # Step 2: Compute the overall mean image (mu*) - over ensemble, latent space (t2, t1)
    mu_star = generated_images.mean(dim=(0, 1))  # shape: (32, 32, 3)

    # Step 3: Compute the mean image for each model (mu*_t2) - over latent space (t1)
    mu_t2_star = generated_images.mean(dim=1)    # shape: (100, 32, 32, 3)

    # Step 3: Compute Σ_epistemic
    mu_t2_star_flat = mu_t2_star.view(T2, -1)  # Flatten the spatial dimensions (T2, H*W*C)
    mu_star_flat = mu_star.view(-1)            # Flatten (H*W*C)

    # Compute Σ_epistemic based on the formula
    # Σ_epistemic = (1 / T2) * Σ(μ_t2* μ_t2*^T) - μ* μ*^T
    epistemic_covariance = (1 / T2) * torch.einsum('ti,tj->ij', mu_t2_star_flat, mu_t2_star_flat) \
                           - torch.outer(mu_star_flat, mu_star_flat)  # Shape: (H*W*C, H*W*C)

    if torch.isnan(epistemic_covariance).any():
        print("NaN detected in STEP 3")
    if torch.isinf(epistemic_covariance).any():
        print("Inf detected in STEP 3")

    # Step 4: Extract diagonal of Σ_epistemic and reshape to image size
    diag_epistemic = epistemic_covariance.diag().view(H, W, C)
    diag_epistemic = torch.clamp(diag_epistemic, min=0)

    if torch.isnan(diag_epistemic).any():
        print("NaN detected in STEP 4")
    if torch.isinf(diag_epistemic).any():
        print("Inf detected in STEP 4")

    # Step 5: Take the square root of the diagonal entries for the final uncertainty map
    epistemic_uncertainty_map = torch.sqrt(diag_epistemic)

    if torch.isnan(epistemic_uncertainty_map).any():
        print("NaN detected in STEP 5")
    if torch.isinf(epistemic_uncertainty_map).any():
        print("Inf detected in STEP 5")

    # Step 6: Normalize the uncertainty map to [0, 1] for better visualization

    # Get the min and max values for normalization
    min_val = epistemic_uncertainty_map.min().item()
    max_val = epistemic_uncertainty_map.max().item()

    # If min and max are very close, we can scale the contrast
    if max_val - min_val < 1e-5:
        # Enhance contrast by amplifying the range
        epistemic_uncertainty_map_normalized = (epistemic_uncertainty_map - min_val) * 1000
    else:
        # Standard normalization
        epistemic_uncertainty_map_normalized = (epistemic_uncertainty_map - min_val) / (max_val - min_val)

    # Ensure the values are within [0, 1] range
    epistemic_uncertainty_map_normalized = torch.clamp(epistemic_uncertainty_map_normalized, 0, 1)

    if torch.isnan(epistemic_uncertainty_map_normalized).any():
        print("NaN detected in STEP 6")
    if torch.isinf(epistemic_uncertainty_map_normalized).any():
        print("Inf detected in STEP 6")

    # Step 7: Sum over the uncertainty map to get final uncertainty
    epistemic_uncertainty = epistemic_uncertainty_map.sum()

    return epistemic_uncertainty, epistemic_uncertainty_map, epistemic_uncertainty_map_normalized

def plot_map_image_uncertainty_maps(map_image, uncertainty_map, uncertainty_value, label, output_dir):
    """
    Plots the image that the pretrained model gives, the uncertainty map as an RGB and BW image.
    """
    # Move the tensors to CPU and convert to NumPy arrays
    map_image = map_image.cpu().detach().numpy()
    map_image = np.moveaxis(map_image, 0, -1)
    uncertainty_map = uncertainty_map.cpu().detach().numpy()
    uncertainty_map = np.moveaxis(uncertainty_map,0, -1)

    # Plot the image_map, variance_rgb, and variance_grayscale side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the generated image map
    axes[0].imshow(map_image)
    axes[0].set_title('Generated Image Map')
    axes[0].axis('off')

    # Plot the RGB uncertainty map
    axes[1].imshow(uncertainty_map)
    axes[1].set_title('Variance of Generated Images (RGB)')
    axes[1].axis('off')

    # Plot the grayscale variance map
    axes[2].imshow(np.sum(uncertainty_map, axis=-1), cmap='gray')
    axes[2].set_title('Variance of Generated Images (Grayscale)')
    axes[2].axis('off')  # Hide axis

    # Add a text box with the total variance
    plt.figtext(0.5, 0.05, f'Total Variance: {uncertainty_value:.4f}',
                ha='center', fontsize=16, color='black', backgroundcolor='white')

    os.makedirs(output_dir, exist_ok=True)

    # Save the plot as an image in the specified directory with the label in the filename
    plt.savefig(os.path.join(output_dir, f"{label}_uncertainties.png"), bbox_inches='tight')

    plt.show()


def compute_epistemic_uncertainty_new(generated_images):
    """
    Computes the epistemic uncertainty of the generated images by:
    1. Computing mean over the latent space dimension (dim=1).
    2. Computing variance over the model dimension (dim=0).
    """
    # Stack the list into a single tensor
    #generated_images = torch.stack(generated_images)  # shape: (num_models, num_samples_per_model, H, W, C)

    # Step 1: Compute the mean over latent space (dim=1)
    mean_latent_space = generated_images.mean(dim=1)  # shape: (num_models, H, W, C)

    # Step 2: Compute variance over the model ensemble (dim=0)
    epistemic_variance = mean_latent_space.var(dim=0)  # shape: (H, W, C)

    # Step 3: Return the square root of the variance as epistemic uncertainty (for visualization purposes)
    epistemic_uncertainty_map = torch.sqrt(epistemic_variance)  # shape: (H, W, C)

    # Step 4: Compute total variance
    epistemic_uncertainty = epistemic_variance.sum()

    # Step 5: Normalize the uncertainty map to [0, 1] for better visualization

    # Get the min and max values for normalization
    min_val = epistemic_uncertainty_map.min().item()
    max_val = epistemic_uncertainty_map.max().item()

    # If min and max are very close, we can scale the contrast
    if max_val - min_val < 1e-5:
        # Enhance contrast by amplifying the range
        epistemic_uncertainty_map_normalized = (epistemic_uncertainty_map - min_val) * 1000
    else:
        # Standard normalization
        epistemic_uncertainty_map_normalized = (epistemic_uncertainty_map - min_val) / (max_val - min_val)

    # Ensure the values are within [0, 1] range
    epistemic_uncertainty_map_normalized = torch.clamp(epistemic_uncertainty_map_normalized, 0, 1)


    return epistemic_uncertainty, epistemic_uncertainty_map, epistemic_uncertainty_map_normalized