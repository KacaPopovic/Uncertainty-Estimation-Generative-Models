# Imports and setup
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os

# Random seed for reproducibility
import random
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Define constants
nc = 3  # Number of image channels
nz = 100  # Size of input noise vector
ngf = 64  # Generator filters in first layer
ndf = 64  # Discriminator filters in first layer
num_classes = 10  # CIFAR-10 has 10 classes

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ngpu = 1  # Number of GPUs to use

# Custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Conditional Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        input = torch.cat((noise, label_emb), 1)
        return self.main(input)

# Conditional Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.main = nn.Sequential(
            nn.Conv2d(nc + num_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_emb = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.expand(labels.size(0), num_classes, img.size(2), img.size(3))
        input = torch.cat((img, label_emb), 1)
        return self.main(input)

class GAN(nn.Module):
    def __init__(self, ngpu, gen, disc):
        super(GAN, self).__init__()
        self.ngpu = ngpu
        self.generator = gen
        self.discriminator = disc

    def forward(self, noise, labels):

        fake_data = self.generator(noise, labels)
        disc_fake = self.discriminator(fake_data, labels).unsqueeze(1)
        return disc_fake

    def generate_image(self, noise):
        if noise.is_cuda and self.ngpu > 1:
            fake_data = nn.parallel.data_parallel(self.generator, noise, range(self.ngpu))
        else:
            fake_data = self.generator(noise)

        return fake_data

    def load_generator_state_dict(self, state_dict):
        self.generator.load_state_dict(state_dict)

    def load_discriminator_state_dict(self, state_dict):
        self.discriminator.load_state_dict(state_dict)

    def save_generator_state_dict(self, path):
        torch.save(self.generator.state_dict(), path)

    def save_discriminator_state_dict(self, path):
        torch.save(self.discriminator.state_dict(), path)

    def freeze_except_last_generator_layer(self):
        # Freeze all layers in the generator except the last convolutional layer
        for name, param in self.generator.named_parameters():
            if name.startswith('main.12'):
                continue
            param.requires_grad = False

        # Freeze all layers in the discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False

def main():
    # Loading the dataset
    dataset = dset.CIFAR10(root="./data", download=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    nc = 3  # Number of image channels

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                             shuffle=True, num_workers=2)

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    # Optionally load pre-trained weights
    # netG.load_state_dict(torch.load('D:/Uncertainty-Estimation-Generative-Models/models/weights/netG_epoch_24.pth', map_location=torch.device('cpu')))

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    # Optionally load pre-trained weights
    # netD.load_state_dict(torch.load('D:/Uncertainty-Estimation-Generative-Models/models/weights/netD_epoch_24.pth', map_location=torch.device('cpu')))

    criterion = nn.BCELoss()

    # Setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(128, nz, 1, 1, device=device)
    fixed_labels = torch.randint(0, 10, (128,), device=device)  # Generate fixed labels for evaluation
    real_label = 1.0
    fake_label = 0.0

    niter = 2
    g_loss = []
    d_loss = []

    # Ensure the output directory exists
    output_dir = 'D:\\Uncertainty-Estimation-Generative-Models\\outputs\\output_coditional_cifar10\\'
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()

            real_cpu, labels = data  # Get the real images and labels
            real_cpu = real_cpu.to(device)
            labels = labels.to(device)
            batch_size = real_cpu.size(0)

            # Train with real images
            label = torch.full((batch_size,), real_label, device=device)
            output = netD(real_cpu, labels)
            output = output.view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake images
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise, labels)  # Generate fake images with the same labels as real ones
            label.fill_(fake_label)
            output = netD(fake.detach(), labels)
            output = output.view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = netD(fake, labels)
            output = output.view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
                epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save the output images
            if i % 100 == 0:
                print('Saving the output')
                vutils.save_image(real_cpu, f'{output_dir}/real_samples.png', normalize=True)
                fake = netG(fixed_noise, fixed_labels)  # Generate images with fixed labels
                vutils.save_image(fake.detach(), f'{output_dir}/fake_samples_epoch_{epoch + 25:03d}.png', normalize=True)

        # Checkpointing after every epoch
        torch.save(netG.state_dict(), f'{output_dir}/netG_epoch_{epoch}.pth')
        torch.save(netD.state_dict(), f'{output_dir}/netD_epoch_{epoch}.pth')

if __name__ == "__main__":
    main()