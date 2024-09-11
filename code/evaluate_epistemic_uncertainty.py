import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy import linalg
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
from laplace_transformation import LaplaceTransformation
from PIL import Image
import os
import matplotlib.pyplot as plt
from pytorch_fid import fid_score
import shutil

def compute_fid(real_images_path, generated_images_path, batch_size, device):
    """
    Computes the Frechet Inception Distance (FID) between two sets of images.
    """
    fid = fid_score.calculate_fid_given_paths([real_images_path, generated_images_path],
                                              batch_size=batch_size, device=device, dims=2048)
    return fid

def save_image(image_tensor, filename):
    # Squeeze the batch dimension and rearrange from CHW to HWC format
    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()

    # Denormalize the image: assuming the range is [-1, 1], bring it to [0, 1]
    image_np = (image_np + 1) / 2.0

    # Convert to [0, 255] and ensure the type is uint8 for saving as an image
    image_np = (image_np * 255).astype(np.uint8)

    # Save using PIL
    Image.fromarray(image_np).save(filename)

# Function to check the number of images in a directory
def check_real_images_folder(real_images_path, required_count=5000):
    # Check if the directory exists, and count the number of images
    if os.path.exists(real_images_path):
        image_files = [f for f in os.listdir(real_images_path) if f.endswith('.png')]
        return len(image_files) >= required_count
    return False

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory to save images
    output_dir = './generated_images'
    os.makedirs(output_dir, exist_ok=True)

    # Generate 25,000 images and compute their variance
    noise_dim = 100
    total_images = 25
    images_with_variance = []

    # Load Laplace model

    weights_dir_MAP = r'D:\Uncertainty-Estimation-Generative-Models\models\weights'
    laplace = LaplaceTransformation(weights_dir_MAP, noise_dim, device)
    laplace.load_map_model()
    weights_dir_Laplace = "laplace_models/freezed_diag_classification_large.bin"
    laplace.load_laplace_model(weights_dir_Laplace, "classification", "all", "diag")
    model = laplace.laplace_model
    new_model = False

    for i in range(total_images):
        noise = torch.randn(1, noise_dim, 1, 1, device=device)
        image_map = laplace.map_model.generate_image(noise)

        # Save the generated image
        image_filename = os.path.join(f'image_{i}.png')
        save_image(image_map, image_filename)

        py, images = model(noise, pred_type="nn", link_approx="mc", n_samples=100)
        images = images.squeeze(1)

        variance = torch.var(images, dim=0, unbiased=False)
        total_variance = variance.sum().item()

        # Store the filename and its variance
        images_with_variance.append((image_filename, total_variance))

    # Sort images by variance
    images_with_variance.sort(key=lambda x: x[1])

    # Split images into 5 bins of 5000 images each
    bins = [images_with_variance[i:i + 5000] for i in range(0, total_images, 5000)]

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Define a folder path for saving real images from CIFAR-10
    real_images_path = './real_images'
    os.makedirs(real_images_path, exist_ok=True)

    # Check if there are already 5000 images in the folder
    if check_real_images_folder(real_images_path):
        print(f'Found {len(os.listdir(real_images_path))} images, skipping CIFAR-10 image saving...')
    else:
        print('Saving CIFAR-10 images to real_images_path...')

        # Load CIFAR-10 dataset (real images)
        cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        cifar10_loader = DataLoader(cifar10_dataset, batch_size=64, shuffle=True, num_workers=2)

        # Save some CIFAR-10 images to real_images_path for FID calculation
        image_count = 0
        for idx, (images, _) in enumerate(cifar10_loader):
            for img_idx, image in enumerate(images):
                save_image(image, os.path.join(real_images_path, f'real_image_{image_count}.png'))
                image_count += 1
                if image_count >= 5000:  # Save only 5000 real images
                    break
            if image_count >= 5000:
                break

    # Calculate FID scores for each bin
    fid_scores = []
    for i, image_bin in enumerate(bins):
        bin_folder = f'./generated_images_bin_{i + 1}'
        os.makedirs(bin_folder, exist_ok=True)

        # Copy the images of the current bin to the new folder
        for image_filename, _ in image_bin:
            image_path = os.path.join(output_dir, image_filename)
            shutil.copy(image_path, bin_folder)

        # Calculate FID score for the current bin
        fid_score = compute_fid(real_images_path, bin_folder, batch_size=2, device=device)
        fid_scores.append(fid_score)
        print(f'FID Score for Bin {i + 1}: {fid_score:.4f}')

    variance_bins = [1, 2, 3, 4, 5]  # Assuming the variance of the bin corresponds to these indices

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(variance_bins, fid_scores, marker='o')
    plt.xlabel('Tubes with increasing uncertainty')
    plt.ylabel('FID Score')
    plt.title('FID Score vs. Uncertainty of the Tube')
    plt.ylim(80, max(fid_scores) + 5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()

