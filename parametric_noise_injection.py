import torch
import torch.nn as nn

class NoisyConv(nn.Module):
    def __init__(self, module):
        super(NoisyConv, self).__init__()
        assert isinstance(module, nn.Conv2d)
        with torch.no_grad():
            std = module.weight.std()
            self.noise = torch.randn_like(module.weight) * std
        self.alpha = nn.Parameter(torch.randn_like(std), requires_grad=True)
        self.core = module

    def forward(self, x):
        return nn.functional.conv2d(x, self.core.weight + self.alpha * self.noise, self.core.bias,
                                    self.core.stride, self.core.padding, self.core.dilation, self.core.groups)

class NoisyLinear(nn.Module):
    def __init__(self, module):
        super(NoisyLinear, self).__init__()
        assert isinstance(module, nn.Linear)
        with torch.no_grad():
            std = module.weight.std()
            self.noise = torch.randn_like(module.weight) * std
        self.alpha = nn.Parameter(torch.randn_like(std), requires_grad=True)
        self.core = module

    def forward(self, x):
        return nn.functional.linear(x, self.core.weight + self.alpha * self.noise, self.core.bias)

class PNI(nn.Module):
    def __init__(self, backbone):
        super(PNI, self).__init__()
        self.backbone = backbone
        self.noise_injection()

    def forward(self, x):
        return self.backbone(x)

    def noise_injection(self):
        def module_wise_noise_injection(module):
            for name, submodule in module._modules.items():
                if len(submodule._modules.items()) > 0:
                    module_wise_noise_injection(submodule)
                elif isinstance(submodule, nn.Conv2d):
                    module._modules[name] = NoisyConv(submodule)
                elif isinstance(submodule, nn.Linear):
                    module._modules[name] = NoisyLinear(submodule)
        module_wise_noise_injection(self.backbone)

if __name__ == '__main__':
    import torchvision
    model = torchvision.models.vgg16_bn(pretrained=False)
    pni_model = PNI(model)
    print(pni_model)
    x = torch.randn(1, 3, 224, 224)
    print(pni_model(x).size())

