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
import os

cudnn.benchmark = True

# set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
nc = 3
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

    def forward(self, input):
        output = self.main(input)
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

    def forward(self, input):

        output = self.main(input)
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

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    # load weights to test the model
    netG.load_state_dict(torch.load('D:/Uncertainty-Estimation-Generative-Models/models/weights/netG_epoch_24.pth', map_location=torch.device('cpu')))
    #print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    # load weights to test the model
    netD.load_state_dict(torch.load('D:/Uncertainty-Estimation-Generative-Models/models/weights/netD_epoch_24.pth', map_location=torch.device('cpu')))
    #print(netD)

    criterion = nn.BCELoss()

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(128, nz, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0

    niter = 5
    g_loss = []
    d_loss = []

    # Ensure the output directory exists
    output_dir = 'D:\\Uncertainty-Estimation-Generative-Models\\code\\output\\'
    os.makedirs(output_dir, exist_ok=True)

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
                vutils.save_image(fake.detach(), 'output/fake_samples_epoch_%03d.png' % (epoch + 25), normalize=True)

        # Check pointing for every epoch
        torch.save(netG.state_dict(), 'D:/Uncertainty-Estimation-Generative-Models/models/weights/netG_epoch_%d.pth' % (epoch+25))
        torch.save(netD.state_dict(), 'D:/Uncertainty-Estimation-Generative-Models/models/weights/netD_epoch_%d.pth' % (epoch+25))


if __name__ == "__main__":
    main()