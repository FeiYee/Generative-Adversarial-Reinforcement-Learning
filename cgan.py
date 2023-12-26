import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子以确保结果可重现
torch.manual_seed(0)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_size),
            nn.Tanh()
        )

    def forward(self, x, labels):
        x = torch.cat([x, labels], 1) # 将噪声和标签合并
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = torch.cat([x, labels], 1) # 将数据和标签合并
        return self.fc(x)

def train_cgan(data_loader, generator, discriminator, g_optimizer, d_optimizer, criterion, num_epochs, device):
    for epoch in range(num_epochs):
        for i, (real_samples, labels) in enumerate(data_loader):
            # 训练鉴别器
            real_samples = real_samples.to(device)
            labels = labels.to(device)
            d_optimizer.zero_grad()
            real_labels = torch.ones((real_samples.size(0), 1)).to(device)
            fake_labels = torch.zeros((real_samples.size(0), 1)).to(device)

            outputs = discriminator(real_samples, labels)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            noise = torch.randn(real_samples.size(0), 100).to(device)
            fake_samples = generator(noise, labels)
            outputs = discriminator(fake_samples, labels)
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            noise = torch.randn(real_samples.size(0), 100).to(device)
            fake_samples = generator(noise, labels)
            outputs = discriminator(fake_samples, labels)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

def main():
    # 超参数设置
    batch_size = 64
    lr = 0.0002
    num_epochs = 50
    sample_dim = 100 # 噪声维度
    label_dim = 10 # 假设有10个不同的标签
    g_hidden_dim = 256
    d_hidden_dim = 256
    fake_data_dim = 784 # 生成数据的维度

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    generator = Generator(sample_dim + label_dim, g_hidden_dim, fake_data_dim).to(device)
    discriminator = Discriminator(fake_data_dim + label_dim, d_hidden_dim, 1).to(device)

    # 优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # 损失函数
    criterion = nn.BCELoss()

    # 加载数据（这里需要替换为您自己的数据加载逻辑）
    # 假设 data 和 labels 是您的数据和标签
    data = torch.randn(1000, fake_data_dim) # 示例数据
    labels = torch.randint(0, label_dim, (1000, 1)) # 示例标签
    dataset = TensorDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练模型
    train_cgan(data_loader, generator, discriminator, g_optimizer, d_optimizer, criterion, num_epochs, device)

if __name__ == "__main__":
    main()
