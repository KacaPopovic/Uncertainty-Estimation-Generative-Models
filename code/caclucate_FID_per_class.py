from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy import linalg
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
from laplace_transformation import LaplaceTransformation

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


def load_or_calculate_features(cifar10_loader, model, device, feature_file='real_features.npy', max_batches=5000):
    # Check if the feature file exists
    if os.path.exists(feature_file):
        print(f"Loading features from {feature_file}...")
        real_features = np.load(feature_file)
    else:
        print(f"{feature_file} not found. Calculating features...")
        real_features = []
        i = 0
        for images, _ in cifar10_loader:
            i += 1
            images = images.to(device)
            features = extract_features(images, model)
            real_features.append(features)

            # Stop after processing max_batches
            if i > max_batches:
                break

        # Concatenate the feature arrays
        real_features = np.concatenate(real_features, axis=0)

        # Save the features to a file
        np.save(feature_file, real_features)
        print(f"Features saved to {feature_file}")

    return real_features


def load_or_calculate_features_per_class(cifar10_loader, model, device, feature_dir='./real_features', num_classes=10,
                                         max_batches=5000):
    # Check if the directory exists
    os.makedirs(feature_dir, exist_ok=True)

    real_features_per_class = {}
    for class_id in range(num_classes):
        feature_file = os.path.join(feature_dir, f'real_features_class_{class_id}.npy')

        # Check if the feature file exists
        if os.path.exists(feature_file):
            print(f"Loading features for class {class_id} from {feature_file}...")
            real_features_per_class[class_id] = np.load(feature_file)
        else:
            print(f"{feature_file} not found. Calculating features for class {class_id}...")
            real_features = []
            i = 0
            for images, labels in cifar10_loader:
                # Only keep images of the current class
                class_images = images[labels == class_id]
                if len(class_images) > 0:
                    class_images = class_images.to(device)
                    features = extract_features(class_images, model)
                    real_features.append(features)

                # Stop after processing max_batches
                i += 1
                if i > max_batches:
                    break

            # Concatenate the feature arrays and save them to a file
            real_features = np.concatenate(real_features, axis=0)
            real_features_per_class[class_id] = real_features
            np.save(feature_file, real_features)
            print(f"Features for class {class_id} saved to {feature_file}")

    return real_features_per_class

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Directory to save images
    output_dir = './generated_images_per_class'
    os.makedirs(output_dir, exist_ok=True)

    # Generate 25,000 images and compute their variance
    noise_dim = 100
    total_images = 25000
    num_classes = 10
    im_per_class = total_images // num_classes

    # Load Laplace model

    weights_dir_MAP = r'D:\Uncertainty-Estimation-Generative-Models\models\weights_conditional'
    laplace = LaplaceTransformation(weights_dir_MAP, noise_dim, device)
    laplace.load_map_model()
    weights_dir_Laplace = "../models/laplace_models/freezed_diag_classification_large_conditional1.bin"
    laplace.load_laplace_model(weights_dir_Laplace, "classification", "all", "diag")
    model = laplace.laplace_model

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
    real_features_per_class = load_or_calculate_features_per_class(cifar10_loader, inception, device, max_batches=5000)

    # Calculate FID score for each class
    fid_scores = []
    for i in range(num_classes):
        generated_features = []
        label = torch.full((1,), i, device=device)
        for j in range(im_per_class):
            noise = torch.randn(1, noise_dim, 1, 1, device=device)
            image_map = laplace.map_model.generate_image(noise, label)
            image_map_resized = torch.nn.functional.interpolate(image_map, size=(299, 299), mode='bilinear')
            image = image_map_resized.to(device)

            features = extract_features(image, inception)
            generated_features.append(features)

        # Concatenate the generated features for the current class
        generated_features = np.concatenate(generated_features, axis=0)

        # Calculate FID score for the current class
        fid_score = calculate_fid(real_features_per_class[i], generated_features)
        fid_scores.append(fid_score)
        print(f'FID Score for Class {i + 1}: {fid_score:.4f}')

    variance_bins = range(1, num_classes + 1)
    uncertainties = [876.1741, 867.83, 836.0944, 841.2355, 855.9167, 854.0078, 852.1121, 861.603, 849.5162,  847.4067]

    sorted_indices = np.argsort(uncertainties)
    sorted_fid_scores = np.array(fid_scores)[sorted_indices]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(variance_bins, sorted_fid_scores, marker='o')
    plt.xlabel('Tubes with increasing uncertainty')
    plt.ylabel('FID Score')
    plt.title('FID Score vs. Uncertainty of the Class')
    plt.ylim(80, max(fid_scores) + 5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()

