import torch
import torch.nn as nn

class DenoisingModule(nn.Module):
    def __init__(self, channel):
        super(DenoisingModule, self).__init__()
        self.theta_embedding = nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1)
        self.phi_embedding = nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1)
        self.conv_11 = nn.Conv2d(channel, channel, kernel_size=1, stride=1)

    def forward(self, x):
        # non-local mean
        channel, height, width = x.size(1), x.size(2), x.size(3)
        theta = self.theta_embedding(x)
        phi = self.phi_embedding(x)
        x = torch.reshape(x, (-1, channel, height * width))
        theta = torch.reshape(theta, (-1, channel // 2, height * width))
        phi = torch.reshape(phi, (-1, channel // 2, height * width))
        f = torch.matmul(theta.transpose(1, 2), phi) / (channel ** 0.5)
        f = nn.functional.softmax(f)
        denoised_result = torch.matmul(f, x.transpose(1, 2)).transpose(1, 2)
        denoised_result = torch.reshape(denoised_result, (-1, channel, height, width))
        # 1*1 conv
        conv_result = self.conv_11(denoised_result)
        return denoised_result + conv_result

class FD(nn.Module):
    """
    backbone should contain multiple instances of nn.Sequential
    feature denoising modules are added at the end of sequentials
    """
    def __init__(self, backbone, indices, dims):
        super(FD, self).__init__()
        self.backbone = backbone
        self.install_denoising_modules(indices, dims)

    def forward(self, x):
        return self.backbone(x)

    def install_denoising_modules(self, indices, dims):
        modules = list(self.backbone.children())
        device = next(self.backbone.parameters()).device
        for i, (name, module) in enumerate(self.backbone._modules.items()):
            if i in indices:
                self.backbone._modules[name] = nn.Sequential(
                    self.backbone._modules[name],
                    DenoisingModule(dims[i]).to(device)
                )

if __name__ == '__main__':
    import torchvision
    model = torchvision.models.vgg16(pretrained=False)
    denoised_model = FD(model, [0, 1], [512, 512])
    print(denoised_model)
    x = torch.randn(1, 3, 224, 224)
    print(denoised_model(x).size())

