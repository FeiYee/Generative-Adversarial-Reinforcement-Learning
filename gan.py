import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generator Definition
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# 2. Discriminator Definition
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 3. Training Function
def train_gan(generator, discriminator, data_loader, epochs, device):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for i, real_data in enumerate(data_loader):
            # Train Discriminator
            discriminator.zero_grad()
            real_data = real_data.to(device)
            real_labels = torch.ones(real_data.size(0), 1).to(device)
            fake_labels = torch.zeros(real_data.size(0), 1).to(device)
            outputs = discriminator(real_data)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            noise = torch.randn(real_data.size(0), noise_dim).to(device)
            fake_data = generator(noise)
            outputs = discriminator(fake_data.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()

            optimizer_d.step()

            # Train Generator
            generator.zero_grad()
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{epochs}], d_loss: {d_loss_real.item() + d_loss_fake.item()}, g_loss: {g_loss.item()}")

# 4. Main Execution
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    noise_dim = 100
    data_dim = 784  # Example for MNIST dataset
    batch_size = 64
    epochs = 50

    generator = Generator(input_dim=noise_dim, output_dim=data_dim).to(device)
    discriminator = Discriminator(input_dim=data_dim).to(device)

    # Load data
    # train
    # train_gan(generator, discriminator, data_loader, epochs, device)
