import torch
import torch.nn as nn
import torch.optim as optim

# 设定隐向量大小、输入图像尺寸等
latent_dim = 100
conditional_dim = 10  # 条件向量的维度
image_size = 784  # 假设是28x28的图像，展平后的尺寸

# 生成器定义
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + conditional_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, z, c):
        input = torch.cat([z, c], 1)
        return self.model(input)

# 鉴别器定义
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1 + conditional_dim),
            nn.Sigmoid()
        )

    def forward(self, img):
        output = self.model(img)
        validity = output[:, 0]
        label = output[:, 1:]
        return validity, label

# 初始化网络和优化器
netG = Generator()
netD = Discriminator()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 生成随机噪声和条件向量的函数
def generate_noise_and_conditions(batch_size):
    z = torch.randn(batch_size, latent_dim)
    c = torch.randn(batch_size, conditional_dim)
    return z, c

# 训练循环
num_epochs = 50  # 或者根据需要调整
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # 配置输入
        real_imgs = imgs.view(imgs.size(0), -1)
        batch_size = real_imgs.size(0)

        # 生成随机噪声和条件向量
        z, c = generate_noise_and_conditions(batch_size)

        # 生成假图像
        fake_imgs = netG(z, c)

        # -----------------
        #  训练鉴别器
        # -----------------
        optimizerD.zero_grad()

        # 真实图像的损失
        real_pred, _ = netD(real_imgs)
        d_real_loss = torch.nn.functional.binary_cross_entropy(real_pred, torch.ones_like(real_pred))

        # 假图像的损失
        fake_pred, _ = netD(fake_imgs.detach())
        d_fake_loss = torch.nn.functional.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))

        # 总损失
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizerD.step()

        # -----------------
        #  训练生成器
        # -----------------
        optimizerG.zero_grad()

        # 生成器的损失
        validity, _ = netD(fake_imgs)
        g_loss = torch.nn.functional.binary_cross_entropy(validity, torch.ones_like(validity))

        g_loss.backward()
        optimizerG.step()

        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
