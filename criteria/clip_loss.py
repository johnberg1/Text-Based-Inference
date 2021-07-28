import torch.nn as nn

class CLIPLoss(nn.Module):    
    def __init__(self, model, styleGAN_size=256):
        super(CLIPLoss, self).__init__()
        self.model = model
        self.upsample = nn.Upsample(scale_factor=7)
        self.avg_pool = nn.AvgPool2d(kernel_size=styleGAN_size // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image,text)[0] / 100
        return similarity