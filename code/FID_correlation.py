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

# Function to calculate FID score
def calculate_fid(real_features, generated_features):
    # Calculate mean and covariance of real features
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    # Calculate mean and covariance of generated features
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)

    # Calculate the FID score
    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)

    # Numerical stability check
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

# Function to extract features using InceptionV3
def extract_features(images, model):
    model.eval()
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()

def save_image(image_tensor, filename):
    # Squeeze the batch dimension and rearrange from CHW to HWC format
    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()

    # Denormalize the image: assuming the range is [-1, 1], bring it to [0, 1]
    image_np = (image_np + 1) / 2.0

    # Convert to [0, 255] and ensure the type is uint8 for saving as an image
    image_np = (image_np * 255).astype(np.uint8)

    # Save using PIL
    Image.fromarray(image_np).save(filename)


def plot_fid_scores(fid_scores):
    bins = range(1, len(fid_scores) + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(bins, fid_scores, color='skyblue')
    plt.xlabel('Bins (sorted by variance)')
    plt.ylabel('FID Score')
    plt.title('FID Scores for Each Bin of Generated Images')
    plt.xticks(bins, [f'Bin {i}' for i in bins])
    plt.show()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory to save images
    output_dir = './generated_images'
    os.makedirs(output_dir, exist_ok=True)

    # Generate 25,000 images and compute their variance
    noise_dim = 100
    total_images = 25000
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
        image_filename = os.path.join(output_dir, f'image_{i}.png')
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
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=64, shuffle=True, num_workers=2)

    # Load InceptionV3 model pre-trained on ImageNet
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()

    # Extract features from CIFAR-10 dataset
    real_features = []
    for images, _ in cifar10_loader:
        images = images.to(device)
        features = extract_features(images, inception)
        real_features.append(features)
    real_features = np.concatenate(real_features, axis=0)

    # Calculate FID scores for each bin
    fid_scores = []
    for i, image_bin in enumerate(bins):
        generated_features = []
        for image_filename, _ in image_bin:
            image = Image.open(image_filename).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            features = extract_features(image, inception)
            generated_features.append(features)
        generated_features = np.concatenate(generated_features, axis=0)

        fid_score = calculate_fid(real_features, generated_features)
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

