import torch.nn as nn
import torchvision.models as models

class ConvModule(nn.Module):
    """Conv Layer consisting of Conv, BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels):
        """
        Args: 
           in_channels: Number of input channels
           out_channels: Number of output channels
        """
        
        super(ConvModule, self).__init__()
        layer = [nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                           stride=1, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        return x
    
class EncoderModule(nn.Module):
    """Encoder component of the segnet"""
    
    def __init__(self, in_channels, out_channels, num_blocks):
        """
        Args: 
           in_channels: Number of input channels
           out_channels: Number of output channels
           num_blocks: Number of conv layers present in this module.
        """

        super(EncoderModule, self).__init__()
        layers = [ConvModule(in_channels, out_channels)]
        for index in range(0, num_blocks - 1):
            layers += [ConvModule(out_channels, out_channels)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            
    def forward(self, x):
        x = self.features(x)
        output_size = x.size()
        x, indices = self.pool(x)
        return x, indices, output_size
    
class DecoderModule(nn.Module):
    """Decoder component of the segnet"""
    
    def __init__(self, in_channels, out_channels, num_blocks):
        """
        Args: 
           in_channels: Number of input channels
           out_channels: Number of output channels
           num_blocks: Number of conv layers present in this module.
        """

        super(DecoderModule, self).__init__()
        layers = []
        for index in range(0, num_blocks - 1):
            layers += [ConvModule(in_channels, in_channels)]
        layers += [ConvModule(in_channels, out_channels)]
        self.features = nn.Sequential(*layers)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
            
    def forward(self, x, indices, size):
        x = self.unpool(input=x, indices=indices, output_size=size)
        x = self.features(x)
        return x
    
class Segnet(nn.Module):
    """Segnet class with encoder and decoder components followed by classifier"""
    
    def __init__(self, num_classes, in_channels=3):
        super(Segnet, self).__init__()
        filter_config = (64, 128, 256, 512, 512)  # Channels for each encoder module
        
        encoder_num_blocks = (2, 2, 3, 3, 3)  # Blocks in each encoder module
        encoder_filter_config = (in_channels,) + filter_config 
        
        decoder_num_blocks = (3, 3, 3, 2, 2)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)
        
        self.encoders = nn.ModuleList()
        for i in range(0, 5):
            self.encoders.append(EncoderModule(encoder_filter_config[i], 
                                               encoder_filter_config[i + 1], 
                                               encoder_num_blocks[i]))
            
                                 
        self.decoders = nn.ModuleList()
        for i in range(0, 5):
            self.decoders.append(DecoderModule(decoder_filter_config[i], 
                                               decoder_filter_config[i + 1], 
                                               decoder_num_blocks[i]))

        # Conv layer for classification
        self.classifier = nn.Conv2d(decoder_filter_config[-1], num_classes, 3, 1, 1)
        # Initialize the encoder parameters with vgg16
        self.init_vgg16_params()
        
    def forward(self, x):
        indices = []
        sizes = []
        
        for i in range(0, 5):
            x, index, size = self.encoders[i](x)
            indices.append(index)
            sizes.append(size)
            
        for i in range(0, 5):
            x = self.decoders[i](x, indices[4 - i], sizes[4 - i])
            
        return self.classifier(x)

    def init_vgg16_params(self):
        # Initializes the encoder layer with vgg16 parameters
        vgg16 = models.vgg16(pretrained=True)
        vgg_layers = []
        for layer in list(vgg16.features.children()):
            if isinstance(layer, nn.Conv2d):
                vgg_layers.append(layer)

        segnet_layers = []
        for encoder in self.encoders:
            for conv_layer in encoder.features:
                for inner_layer in conv_layer.features:
                    if isinstance(inner_layer, nn.Conv2d):
                        segnet_layers.append(inner_layer)
                        
        assert len(vgg_layers) == len(segnet_layers)

        for first, second in zip(vgg_layers, segnet_layers):
            if isinstance(first, nn.Conv2d) and isinstance(second, nn.Conv2d):
                assert first.weight.size() == second.weight.size()
                assert first.bias.size() == second.bias.size()
                second.weight.data = first.weight.data
                second.bias.data = first.bias.data
