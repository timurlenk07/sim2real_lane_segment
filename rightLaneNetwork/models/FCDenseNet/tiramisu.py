from torch.autograd import Function
from torch.nn import functional as F

from .layers import *


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class FCDenseNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                                 growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        self.featureChannels = cur_channels_count

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        return out

    def getFeatureChannels(self):
        return self.featureChannels


class FCDenseNetClassifier(nn.Module):
    def __init__(self, in_channels, n_classes, temperature=0.05):
        super().__init__()
        self.finalConv = nn.Conv2d(in_channels=in_channels, out_channels=n_classes,
                                   kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.T = temperature

    def forward(self, x, reverseGrad=False, useSoftmax=True):
        if reverseGrad:
            x = grad_reverse(x)
        x = F.normalize(x)
        x = self.finalConv(x)
        x = x / self.T
        if useSoftmax:
            x = self.softmax
        return x


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super().__init__()

        self.featureExtractor = FCDenseNetFeatureExtractor(
            in_channels=in_channels, down_blocks=down_blocks,
            up_blocks=up_blocks, bottleneck_layers=bottleneck_layers,
            growth_rate=growth_rate, out_chans_first_conv=out_chans_first_conv
        )
        self.classifier = FCDenseNetClassifier(
            in_channels=self.featureExtractor.getFeatureChannels(), n_classes=n_classes
        )

    def forward(self, x):
        x = self.featureExtractor(x)
        x = self.classifier(x)
        return x


def FCDenseNet57(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet67(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet103(n_classes):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 5, 7, 10, 12),
        up_blocks=(12, 10, 7, 5, 4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet57Base():
    return FCDenseNetFeatureExtractor(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48
    )


def FCDenseNet57Classifier(n_classes):
    return FCDenseNetClassifier(FCDenseNet57Base().getFeatureChannels(), n_classes)
