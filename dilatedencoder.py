def c2_xavier_fill(module: nn.Module):
  """
  Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
  Also initializes `module.bias` to 0.
  Args:
      module (torch.nn.Module): module to initialize.
  """
  # Caffe2 implementation of XavierFill in fact
  # corresponds to kaiming_uniform_ in PyTorch
  nn.init.kaiming_uniform_(module.weight, a=1)
  if module.bias is not None:
    nn.init.constant_(module.bias, 0)


class Bottleneck(nn.Module):

  def __init__(self,
    in_channels: int = 512,
    mid_channels: int = 128,
    dilation: int = 1):

    super(Bottleneck, self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
      nn.BatchNorm2d(mid_channels),
      nn.ReLU()
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=dilation, dilation=dilation),
      nn.BatchNorm2d(mid_channels),
      nn.ReLU()
    )
    self.conv3 = nn.Sequential(
      nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
      nn.BatchNorm2d(in_channels),
      nn.ReLU()
    )
      
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = out + identity
    return out

class DilatedEncoder(nn.Module):
  def __init__(self,
    in_channels,
    mid_channels,
    dilation_growth_factor=2,
    num_residual_blocks=4):
    """Residual Block for DarkNet.
    This module has the dowsample layer (optional),
    1x1 conv layer and 3x3 conv layer.
    """
    super(DilatedEncoder, self).__init__()

    self.lateral_conv = nn.Conv2d(in_channels,
    mid_channels,
    kernel_size=1)
    self.lateral_norm = nn.BatchNorm2d(mid_channels)

    self.fpn_conv = nn.Conv2d(mid_channels,
    mid_channels,
    kernel_size=3,
    padding=1)        
    self.fpn_norm = nn.BatchNorm2d(mid_channels)

    encoder_blocks = []
    for i in range(1, num_residual_blocks + 1):
      dilation = i * dilation_growth_factor
      encoder_blocks.append(
      Bottleneck(
      mid_channels,
      in_channels,
      dilation=dilation
      )
      )

    self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)
    self._init_weight()

  def _init_weight(self):
    c2_xavier_fill(self.lateral_conv)
    c2_xavier_fill(self.fpn_conv)
    for m in [self.lateral_norm, self.fpn_norm]:
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
    for m in self.dilated_encoder_blocks.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias, 0)

      if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
                
  def forward(self, feature: torch.Tensor) -> torch.Tensor:
    out = self.lateral_norm(self.lateral_conv(feature))
    out = self.fpn_norm(self.fpn_conv(out))
    return self.dilated_encoder_blocks(out)
