import torch
import torch as th
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.silu(out)
        out += residual
        out = self.conv2(out)
        out = self.silu(out)
        return out

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class SimpleControlNet(nn.Module):   ## ToDo 
    def __init__(
        self,
        image_size,
        in_channels,   #4
        model_channels,# 320
        hint_channels,
        dims=2
        ):
        super().__init__()
        self.input_blocks = nn.ModuleList(
            [
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
            ]
        ) # 320 64
        ch = model_channels
        self.conv_ms = nn.ModuleList([nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0)])
        self.conv_ss = nn.ModuleList([nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0)])

        self.input_blocks.append(ResidualBlock(ch, ch, 1, 1, 0)) # 320 64
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))


        self.input_blocks.append(ResidualBlock(ch, ch, 1, 1, 0)) # 320 64
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))

        self.input_blocks.append(ResidualBlock(ch, ch*2, 1, 1, 0)) # 640 64
        ch = ch * 2
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))


        self.input_blocks.append(conv_nd(dims, ch, ch, 3, stride=2, padding=1)) # 640 32
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))

        self.input_blocks.append(ResidualBlock(ch, ch, 1, 1, 0)) # 640 32
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))

        self.input_blocks.append(ResidualBlock(ch, ch*2, 1, 1, 0)) # 1280 32
        ch = ch * 2
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))

        self.input_blocks.append(conv_nd(dims, ch, ch, 3, stride=2, padding=1)) # 1280 16
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))

        self.input_blocks.append(ResidualBlock(ch, ch, 1, 1, 0)) # 1280 16
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))

        self.input_blocks.append(ResidualBlock(ch, ch, 1, 1, 0)) # 1280 16
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))

        self.input_blocks.append(conv_nd(dims, ch, ch, 3, stride=2, padding=1)) # 1280 8
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))

        self.input_blocks.append(ResidualBlock(ch, ch, 1, 1, 0)) # 1280 8
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))

        self.input_blocks.append(ResidualBlock(ch, ch, 1, 1, 0)) # 1280 8
        self.conv_ms.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))
        self.conv_ss.append(nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0))

    def forward(self, hint):

        outs = []

        h = hint.type(torch.float32)
        for module, conv_m, conv_s in zip(self.input_blocks, self.conv_ms, self.conv_ss):
            h = module(h)
            outs.append([conv_m(h), conv_s(h), h])

        return outs