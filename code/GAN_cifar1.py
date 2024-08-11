from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

cudnn.benchmark = True

# set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# loading the dataset
dataset = dset.CIFAR10(root="./data", download=True,
                       transform=transforms.Compose([
                           transforms.Resize(64),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))
nc = 3

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=True, num_workers=2)

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


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.convtrans2d_1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.batchnorm2d_1 = nn.BatchNorm2d(ngf * 8)
        self.relu_1 = nn.ReLU(True)

        self.convtrans2d_2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.batchnorm2d_2 = nn.BatchNorm2d(ngf * 4)
        self.relu_2 = nn.ReLU(True)

        self.convtrans2d_3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.batchnorm2d_3 = nn.BatchNorm2d(ngf * 2)
        self.relu_3 = nn.ReLU(True)

        self.convtrans2d_4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.batchnorm2d_4 = nn.BatchNorm2d(ngf)
        self.relu_4 = nn.ReLU(True)

        self.convtrans2d_5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input):
        x = self.convtrans2d_1(input)
        x = self.batchnorm2d_1(x)
        x = self.relu_1(x)

        x = self.convtrans2d_2(x)
        x = self.batchnorm2d_2(x)
        x = self.relu_2(x)

        x = self.convtrans2d_3(x)
        x = self.batchnorm2d_3(x)
        x = self.relu_3(x)

        x = self.convtrans2d_4(x)
        x = self.batchnorm2d_4(x)
        x = self.relu_4(x)

        x = self.convtrans2d_5(x)
        output = self.tanh(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.conv2d_1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.leakyrelu_1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2d_2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.batchnorm2d_1 = nn.BatchNorm2d(ndf * 2)
        self.leakyrelu_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2d_3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.batchnorm2d_2 = nn.BatchNorm2d(ndf * 4)
        self.leakyrelu_3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2d_4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.batchnorm2d_3 = nn.BatchNorm2d(ndf * 8)
        self.leakyrelu_4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2d_5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.conv2d_1(input)
        x = self.leakyrelu_1(x)

        x = self.conv2d_2(x)
        x = self.batchnorm2d_1(x)
        x = self.leakyrelu_2(x)

        x = self.conv2d_3(x)
        x = self.batchnorm2d_2(x)
        x = self.leakyrelu_3(x)

        x = self.conv2d_4(x)
        x = self.batchnorm2d_3(x)
        x = self.leakyrelu_4(x)

        x = self.conv2d_5(x)
        output = self.sigmoid(x)

        return output.view(-1, 1).squeeze(1)



class GAN(nn.Module):
    def __init__(self, ngpu, gen, disc):
        super(GAN, self).__init__()
        self.ngpu = ngpu
        self.generator = gen
        self.discriminator = disc

    def forward(self, noise, real_data):
        fake_data = self.generator(noise)
        disc_fake = self.discriminator(fake_data)
        disc_real = self.discriminator(real_data)

        return fake_data, disc_fake, disc_real

    def load_generator_state_dict(self, state_dict):
        self.generator.load_state_dict(state_dict)

    def load_discriminator_state_dict(self, state_dict):
        self.discriminator.load_state_dict(state_dict)

    def save_generator_state_dict(self, path):
        torch.save(self.generator.state_dict(), path)

    def save_discriminator_state_dict(self, path):
        torch.save(self.discriminator.state_dict(), path)


import torch.nn as nn


class CombinedGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(CombinedGAN, self).__init__()
        self.ngpu = generator.ngpu

        # Flatten all layers from generator and discriminator into a single list
        self.layers = nn.ModuleList()

        # Add generator layers
        self.layers.append(generator.convtrans2d_1)
        self.layers.append(generator.batchnorm2d_1)
        self.layers.append(generator.relu_1)
        self.layers.append(generator.convtrans2d_2)
        self.layers.append(generator.batchnorm2d_2)
        self.layers.append(generator.relu_2)
        self.layers.append(generator.convtrans2d_3)
        self.layers.append(generator.batchnorm2d_3)
        self.layers.append(generator.relu_3)
        self.layers.append(generator.convtrans2d_4)
        self.layers.append(generator.batchnorm2d_4)
        self.layers.append(generator.relu_4)
        self.layers.append(generator.convtrans2d_5)
        self.layers.append(generator.tanh)

        # Add discriminator layers
        self.layers.append(discriminator.conv2d_1)
        self.layers.append(discriminator.leakyrelu_1)
        self.layers.append(discriminator.conv2d_2)
        self.layers.append(discriminator.batchnorm2d_1)
        self.layers.append(discriminator.leakyrelu_2)
        self.layers.append(discriminator.conv2d_3)
        self.layers.append(discriminator.batchnorm2d_2)
        self.layers.append(discriminator.leakyrelu_3)
        self.layers.append(discriminator.conv2d_4)
        self.layers.append(discriminator.batchnorm2d_3)
        self.layers.append(discriminator.leakyrelu_4)
        self.layers.append(discriminator.conv2d_5)
        self.layers.append(discriminator.sigmoid)

    def forward(self, input, mode='generator'):
        x = input
        if mode == 'generator':
            for layer in self.layers[:15]:  # Generator layers
                x = layer(x)
        elif mode == 'discriminator':
            for layer in self.layers[15:]:  # Discriminator layers
                x = layer(x)
            x = x.view(-1, 1).squeeze(1)
        return x


def main():
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    # load weights to test the model
    # netG.load_state_dict(torch.load('weights/netG_epoch_24.pth'))
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    # load weights to test the model
    # netD.load_state_dict(torch.load('weights/netD_epoch_24.pth'))
    print(netD)

    criterion = nn.BCELoss()

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(128, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    niter = 25
    g_loss = []
    d_loss = []

    for epoch in range(niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
            epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # save the output
            if i % 100 == 0:
                print('saving the output')
                vutils.save_image(real_cpu, 'output/real_samples.png', normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), 'output/fake_samples_epoch_%03d.png' % (epoch), normalize=True)

        # Check pointing for every epoch
        torch.save(netG.state_dict(), 'weights/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), 'weights/netD_epoch_%d.pth' % (epoch))


if __name__ == "__main__":
    main()