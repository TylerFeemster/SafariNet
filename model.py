import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
  def __init__(self, num_channels, reduction=16):
    super(SqueezeExcitation, self).__init__()
    self.squeeze = nn.AdaptiveAvgPool2d(1)
    self.excitation = nn.Sequential(
        nn.Linear(num_channels, num_channels//reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(num_channels//reduction, num_channels, bias=False),
        nn.Sigmoid()
    )

  def forward(self, x):
    bs, num_channels, _, _ = x.shape
    y = self.squeeze(x).view(bs, num_channels)
    y = self.excitation(y).view(bs, num_channels, 1, 1)
    return x * y.expand_as(x)


class VGG16Counting(nn.Module):
    def __init__(self, num_classes: int = 6, add_se: bool = False):
        super(VGG16Counting, self).__init__()
        self.including_se = add_se

        # bs x 3 x 128 x 128
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, 5),
                                    nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.Conv2d(64, 64, 5),
                                    nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.MaxPool2d(2))
        # bs x 64 x 60 x 60
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, 5),
                                    nn.BatchNorm2d(128), nn.ReLU(),
                                    nn.Conv2d(128, 128, 5),
                                    nn.BatchNorm2d(128), nn.ReLU(),
                                    nn.MaxPool2d(2))
        # bs x 128 x 26 x 26
        if self.including_se:
            self.se = SqueezeExcitation(128)
        
        self.branches, self.outputs = nn.ModuleList(), nn.ModuleList()
        for _ in range(2):
            # bs x 128 x 26 x 26
            self.branches.append(nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                               nn.BatchNorm2d(128), nn.ReLU(),
                                               nn.Conv2d(128, 128, 3, padding=1),
                                               nn.BatchNorm2d(128), nn.ReLU(),
                                               nn.MaxPool2d(2)))

            #    two inputs: bs x 128 x 13 x 13 
            # -> two inputs: bs x (128 * 13 * 13) 
            # -> one input:  bs x (2 * 128 * 13 * 13)
            self.outputs.append(nn.Sequential(nn.Linear(2 * 128 * 13 * 13, 256),
                                              nn.ReLU(), nn.Dropout(0.2),
                                              nn.Linear(256, num_classes)))

    def forward(self, x):
      x = self.block1(x)
      x = self.block2(x)
      
      if self.including_se:
        x = self.se(x)

      x = torch.cat([torch.flatten(branch(x), start_dim=1)
                    for branch in self.branches], 1)
      outputs = [output(x) for output in self.outputs]

      return outputs
