from laplace import Laplace
from GAN_cifar import Generator, Discriminator, GAN
import torch
from torchvision import datasets as dset, transforms
from torch.utils.data import DataLoader, random_split, Subset
from laplace.utils import ModuleNameSubnetMask


def load_map_model(ngpu=1, device='cpu'):
    generator = Generator(ngpu).to(device)
    discriminator = Discriminator(ngpu).to(device)
    gan = GAN(ngpu, generator, discriminator).to(device)
    gan.load_generator_state_dict(torch.load('weights/netG_epoch_24.pth'))
    gan.load_discriminator_state_dict(torch.load('weights/netD_epoch_24.pth'))
    return gan

def main():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # number of gpu's available
    ngpu = 1
    # input noise dimension
    nz = 100
    # number of generator filters
    ngf = 64
    # number of discriminator filters
    ndf = 64

    # checking the availability of cuda devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the dataset
    full_dataset = dset.CIFAR10(root="./data", train=True, download=True, transform=transform)

    # Define the split ratio
    train_size = int(0.8 * len(full_dataset))  # 80% for training
    val_size = len(full_dataset) - train_size  # the rest for validation

    # Split the dataset into training and validation subsets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Pre-trained model
    model = load_map_model(ngpu, device)

    # Define subnetwork for laplace method - only generator should be bayesian
    subnetwork_mask = ModuleNameSubnetMask(model, module_names=["generator"])
    subnetwork_mask.select()
    subnetwork_indices = subnetwork_mask.indices

    # User-specified LA flavor
    la = Laplace(model, "regression",
                 subset_of_weights="all",
                 hessian_structure="diag",
                 subnetwork_indices=subnetwork_indices)
    la.fit(train_loader)
    la.optimize_prior_precision(
        method="gridsearch",
        pred_type="glm",
        link_approx="probit",
        val_loader=val_loader
    )
    # User-specified predictive approx.
    # pred = la(x, pred_type="glm", link_approx="probit")

if __name__ == '__main__':
    main()