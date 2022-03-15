import torch.nn as nn
import torch   
		
class GeneratorLoss(nn.Module):
    def __init__(self, discriminator):
        super(GeneratorLoss, self).__init__()
        self.discriminator = discriminator
    
    def forward(self, x):
        fake_logit = self.discriminator(x, None)
        gloss = nn.functional.softplus(-fake_logit)
        return gloss