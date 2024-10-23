import matplotlib.pyplot as plt
import torch
import numpy as np
def plot_generated_images(images: torch.Tensor, n_row: int, n_col: int, rgb: bool = True, save_path=None) -> None:
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
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_MAP_and_uncertainty(images: torch.Tensor, image_map: torch.Tensor, rgb=True, save_path=None):
    """
    Plot the MAP image and its uncertainty (variance), and optionally save the plot.

    Args:
        images (torch.Tensor): Generated images tensor of shape (N, C, H, W).
        image_map (torch.Tensor): MAP image tensor of shape (C, H, W).
        rgb (bool): If True, plot RGB variance; else, plot grayscale variance.
        save_path (str): File path to save the plot. If None, the plot will not be saved.

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
        axes[1].imshow(variance_grayscale, cmap='hot')
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

    # Save the plot if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    # Display the plot
    plt.show()

    return variance_rgb, variance_grayscale, total_variance



def normalize_image(image):
    # Normalize the image to the range [0, 1]
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min)


def plot_images_and_variance(model, noise_dim, device, num_plots=10, selected_samples=5):
    """
    Plots randomly selected images and their variance from a model generating multiple samples from noise.

    Args:
    - model: The PyTorch model used to generate images.
    - noise_dim: The dimension of the noise vector input to the model.
    - device: The device ('cpu' or 'cuda') on which to run the model.
    - num_plots: Number of different noise samples to generate and plot. Default is 10.
    - selected_samples: Number of random samples to select from the generated images. Default is 5.

    """
    all_images = []

    # Generating images and variances
    for i in range(num_plots):
        noise = torch.randn(noise_dim, 1, 1, device=device).unsqueeze(0)
        py, images = model(noise, pred_type="nn", link_approx="mc", n_samples=100)

        # Calculate variance
        variance = torch.var(images, dim=0, unbiased=False)
        variance_grayscale = variance.squeeze().cpu().detach().numpy()

        # Store selected samples and variance image
        all_images.append(images[:selected_samples].cpu().detach())  # Selected samples (5 images)
        all_images.append(variance_grayscale)  # Variance image

    # Plotting: flipping rows and columns
    fig, axs = plt.subplots(nrows=selected_samples + 1, ncols=num_plots, figsize=(15, 2 * (selected_samples + 1)))

    # Plotting each set of images
    for i in range(num_plots):
        # First plot the selected 5 samples in each column
        for j in range(selected_samples):
            sample_image = all_images[i * 2][
                j].squeeze().cpu().detach().numpy()  # Fetch the image, remove batch dimension
            axs[j, i].imshow(sample_image, cmap='gray')
            axs[j, i].axis('off')

        # Then plot the variance image as the last row in each column
        variance_image = all_images[i * 2 + 1]  # The variance image was stored as the second element for each plot
        axs[selected_samples, i].imshow(variance_image, cmap='hot')
        axs[selected_samples, i].axis('off')

    # Tight layout for better spacing
    plt.tight_layout()
    plt.show()

