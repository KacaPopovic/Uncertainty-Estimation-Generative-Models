# import all libraries
from __future__ import print_function
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch
import torchvision
from torch import nn
from torch import optim
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


cudnn.benchmark = True

# set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
nc = 1
# checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
# number of discriminator filters
ndf = 64


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        output = self.main(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


class GAN(nn.Module):
    def __init__(self, ngpu, gen, disc):
        super(GAN, self).__init__()
        self.ngpu = ngpu
        self.generator = gen
        self.discriminator = disc

    def forward(self, noise):
        if noise.is_cuda and self.ngpu > 1:
            fake_data = nn.parallel.data_parallel(self.generator, noise, range(self.ngpu))
            disc_fake = nn.parallel.data_parallel(self.discriminator, fake_data, range(self.ngpu))
        else:
            fake_data = self.generator(noise)
            disc_fake = self.discriminator(fake_data).unsqueeze(1)

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
    # define the hyperparameters and variables
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 256
    IMAGE_SIZE = 64
    EPOCHS = 250
    image_channels = 1
    noise_channels = 256
    gen_features = 64
    disc_features = 64

    # set everything to GPU
    device = torch.device("cuda")

    # define the transform
    data_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # load the dataset
    dataset = FashionMNIST(root="dataset/", train=True, transform=data_transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # load models
    gen_model = Generator(ngpu).to(device)
    disc_model = Discriminator(ngpu).to(device)

    # setup optimizers for both models
    gen_optimizer = optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # define the loss function
    criterion = nn.BCELoss()

    # make both models train
    gen_model.train()
    disc_model.train()

    # deifne labels for fake images and real images for the discriminator
    fake_label = 0
    real_label = 1

    # define a fixed noise
    fixed_noise = torch.randn(64, noise_channels, 1, 1).to(device)

    # make the writers for tensorboard
    writer_real = SummaryWriter(f"runs/fashion/test_real")
    writer_fake = SummaryWriter(f"runs/fashion/test_fake")

    # define a step
    step = 0

    print("Start training...")

    # loop over all epochs and all data
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(dataloader):
            # set the data to cuda
            data = data

            # get the batch size
            batch_size = data.shape[0]

            # Train the discriminator model on real data
            disc_model.zero_grad()
            label = (torch.ones(batch_size) * 0.9).to(device)
            output = disc_model(data).reshape(-1)
            real_disc_loss = criterion(output, label)

            # train the disc model on fake (generated) data
            noise = torch.randn(batch_size, noise_channels, 1, 1).to(device)
            fake = gen_model(noise)
            label = (torch.ones(batch_size) * 0.1).to(device)
            output = disc_model(fake.detach()).reshape(-1)
            fake_disc_loss = criterion(output, label)

            # calculate the final discriminator loss
            disc_loss = real_disc_loss + fake_disc_loss

            # apply the optimizer and gradient
            disc_loss.backward()
            disc_optimizer.step()

            # train the generator model
            gen_model.zero_grad()
            label = torch.ones(batch_size).to(device)
            output = disc_model(fake).reshape(-1)
            gen_loss = criterion(output, label)
            # apply the optimizer and gradient
            gen_loss.backward()
            gen_optimizer.step()

            # print losses in console and tensorboard
            if batch_idx % 50 == 0:
                step += 1

                # print everything
                print(
                    f"Epoch: {epoch} ===== Batch: {batch_idx}/{len(dataloader)} ===== Disc loss: {disc_loss:.4f} ===== Gen loss: {gen_loss:.4f}"
                )

                # test the model
                with torch.no_grad():
                    # generate fake images
                    fake_images = gen_model(fixed_noise)
                    # make grid in the tensorboard
                    img_grid_real = torchvision.utils.make_grid(data[:40], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake_images[:40], normalize=True)

                    # write the images in tensorbaord
                    writer_real.add_image(
                        "Real images", img_grid_real, global_step=step
                    )
                    writer_fake.add_image(
                        "Generated images", img_grid_fake, global_step=step
                    )


if __name__ == "__main__":
    main()
