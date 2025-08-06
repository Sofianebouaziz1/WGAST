import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numpy as np


NUM_BANDS = 1
SCALE_FACTOR = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(ConvBlock, self).__init__()

        # Sequential block containing reflection padding and convolution
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding),  # Adds reflection padding to preserve edge details
            nn.Conv2d(input_size, output_size, kernel_size, stride, 0, bias=bias)  # 2D convolution
        )

        # LeakyReLU activation function to introduce non-linearity
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)  # Apply convolution with padding
        return self.act(out)  # Apply LeakyReLU activation


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)

class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super(ResBlock, self).__init__()

        # Define the layers for the residual block
        residual = [
            nn.ReflectionPad2d(padding),  # Add reflection padding to preserve spatial dimensions
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 0, bias=bias),  # Convolution with input and output channels equal (identity mapping for the residual connection)
            nn.Dropout(0.5),  # Regularization to prevent overfitting
            nn.LeakyReLU(inplace=True),  # Apply LeakyReLU activation
            nn.ReflectionPad2d(padding),  # Add padding again before the second convolution
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 0, bias=bias),  # Second convolution
            nn.Dropout(0.5),  # Another dropout layer for regularization
        ]
        self.residual = nn.Sequential(*residual)  # Combine layers into a sequential module

    def forward(self, inputs):
        trunk = self.residual(inputs)  # Pass input through the residual block
        return trunk + inputs  # Add the original input to the output (residual connection)


class FeatureExtract(torch.nn.Module):
    def __init__(self, in_channels=NUM_BANDS):  # Constructor with default input channels (NUM_BANDS)
        super(FeatureExtract, self).__init__()
        channels = (16, 32, 64, 128, 256)  # Defining a tuple for the number of channels at each layer
        
        # First convolutional block (conv1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 1, 3),  # 7x7 kernel, stride 1, padding 3
            ResBlock(channels[0]),  # Apply ResBlock after convolution (with 16 output channels)
        )
        
        # Second convolutional block (conv2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2, 1),  # 3x3 kernel, stride 2, padding 1
            ResBlock(channels[1]),  # Apply ResBlock after convolution (with 32 output channels)
        )
        
        # Third convolutional block (conv3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),  # 3x3 kernel, stride 2, padding 1
            ResBlock(channels[2]),  # Apply ResBlock after convolution (with 64 output channels)
        )
        
        # Fourth convolutional block (conv4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2, 1),  # 3x3 kernel, stride 2, padding 1
            ResBlock(channels[3]),  # Apply ResBlock after convolution (with 128 output channels)
        )
        
        # Fifth convolutional block (conv5)
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, 2, 1),  # 3x3 kernel, stride 2, padding 1
            ResBlock(channels[4]),  # Apply ResBlock after convolution (with 256 output channels)
        )

    def forward(self, inputs):  # Define the forward pass of the model
        # Pass input through each convolutional block
        l1 = self.conv1(inputs)  # First layer output
        l2 = self.conv2(l1)  # Second layer output
        l3 = self.conv3(l2)  # Third layer output
        l4 = self.conv4(l3)  # Fourth layer output
        l5 = self.conv5(l4)  # Fifth layer output
        
        # Return the outputs from all layers as a list
        return [l1, l2, l3, l4, l5]


class SignificanceExtraction(nn.Module):
    def __init__(self, in_channels, ifattention=True, iftwoinput=False, outputM=False):
        super(SignificanceExtraction, self).__init__()

        # Flags to determine whether to use attention, two inputs, and whether to output M1
        self.attention = ifattention  
        self.twoinput = iftwoinput  
        self.outputM = outputM  

        if self.attention:  # If attention mechanism is enabled
            # First 1x1 convolution layer with batch normalization
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels)
            )

            # Second 1x1 convolution layer with batch normalization
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels)
            )

            # Attention mask generation: produces a single-channel attention map
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(1),
                nn.Sigmoid()  # Outputs values between 0 and 1 for attention weighting
            )

    def forward(self, inputs):
        if self.attention:  # If attention is enabled

            if self.twoinput:  # If using two distinct inputs
                input1 = self.conv1(inputs[0])  # Process first input
                input2 = self.conv2(inputs[2])  # Process third input
            else:  # Default case, using first and second input
                input1 = self.conv1(inputs[0]) # modis t2
                input2 = self.conv2(inputs[1]) # modis t1

            x = input1 - input2  # Compute difference between feature maps
            M1 = self.conv(x)  # Generate attention mask

            # Weighted sum: Enhances important features while suppressing others
            result = inputs[0] * M1 + inputs[2] * (1 - M1)
        else:
            # If no attention, take an equal-weighted average
            result = 0.5 * inputs[0] + 0.5 * inputs[2]

        # Return the result and optionally the attention mask
        return (result, M1) if self.outputM else result


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)  # Ensure input has 4 dimensions (N, C, H, W)
    N, C = size[:2]  # Extract batch size (N) and channels (C)
    
    # Compute variance across spatial dimensions (H, W), adding small epsilon for numerical stability
    feat_var = feat.view(N, C, -1).var(dim=2) + eps  
    feat_std = feat_var.sqrt().view(N, C, 1, 1)  # Compute standard deviation and reshape for broadcasting
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)  # Compute mean and reshape for broadcasting
    
    return feat_mean, feat_std  # Return mean and std per channel


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])  # Ensure same batch size and channels
    size = content_feat.size()  # Get input size

    # Compute mean and std for style and content features
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    # Normalize content features and re-scale using style statistics
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)  # Apply style statistics



class SimilarityFeatureRefiner(nn.Module):
    def __init__(self, method='cosine'):
        """
        Refines Landsat LST features using similarity with Sentinel index features.
        Args:
            method (str): 'cosine' or 'corr' similarity.
        """
        super(SimilarityFeatureRefiner, self).__init__()
        assert method in ['cosine', 'corr']
        self.method = method

    def forward(self, HS_feat, HS_indices_feat, SS_feat):
        """
        Args:
            HS_feat: (B, C, H, W) - Landsat LST features (encoded).
            HS_indices_feat: (B, C, H, W) - Landsat indices features (encoded).
            SS_feat: (B, C, H, W) - Sentinel indices features (encoded).

        Returns:
            refined_HS_feat: (B, C, H, W) - Refined Landsat LST features.
        """
        if self.method == 'cosine':
            norm_HS = F.normalize(HS_indices_feat, p=2, dim=1)
            norm_SS = F.normalize(SS_feat, p=2, dim=1)
            similarity = (norm_HS * norm_SS).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        elif self.method == 'corr':
            HS_centered = HS_indices_feat - HS_indices_feat.mean(dim=1, keepdim=True)
            SS_centered = SS_feat - SS_feat.mean(dim=1, keepdim=True)

            numerator = (HS_centered * SS_centered).sum(dim=1, keepdim=True)
            denominator = torch.sqrt((HS_centered**2).sum(dim=1, keepdim=True) * (SS_centered**2).sum(dim=1, keepdim=True) + 1e-6)
            similarity = numerator / denominator

        #similarity = similarity.clamp(min=0.0, max=1.0)

        # Refine Landsat LST features
        refined = HS_feat * similarity
        return refined


class CombinFeatureGenerator(nn.Module):
    def __init__(self, NUM_BANDS=NUM_BANDS, ifAdaIN=True, ifAttention=True, ifTwoInput=False, outputM=False):
        super(CombinFeatureGenerator, self).__init__()

        # Initialize module parameters
        self.ifAdaIN = ifAdaIN  # Use Adaptive Instance Normalization
        self.ifAttention = ifAttention  # Use Attention Mechanism
        self.ifTwoInput = ifTwoInput  # Use two input sources
        self.outputM = outputM  # Output attention map M

        # Feature extraction networks for Landsat and Hyperspectral inputs
        self.indices_SNet = FeatureExtract(in_channels = 3)
        self.MODIS_SNet = FeatureExtract(in_channels = 1)
        self.Landsat_SNet = FeatureExtract(in_channels = 1)

        # Define channels at different levels
        channels = (16, 32, 64, 128, 256)

        # Create a list of significance extraction modules for different feature levels
        self.SignE_List = nn.ModuleList([
            SignificanceExtraction(in_channels=ch, ifattention=self.ifAttention,
                                   iftwoinput=self.ifTwoInput, outputM=self.outputM).cuda()
            for ch in channels
        ])

        self.similarity_refiner = SimilarityFeatureRefiner(method='cosine')

        # Define decoder with deconvolution and residual blocks
        self.conv1 = nn.Sequential(
            DeconvBlock(channels[4] * 2, channels[3], 4, 2, 1, bias=True),
            ResBlock(channels[3]),
        )
        self.conv2 = nn.Sequential(
            DeconvBlock(channels[3] * 2, channels[2], 4, 2, 1, bias=True),
            ResBlock(channels[2]),
        )
        self.conv3 = nn.Sequential(
            DeconvBlock(channels[2] * 2, channels[1], 4, 2, 1, bias=True),
            ResBlock(channels[1]),
        )
        self.conv4 = nn.Sequential(
            DeconvBlock(channels[1] * 2, channels[0], 4, 2, 1, bias=True),
            #nn.MaxPool2d(kernel_size=2, stride=1),  # Spatial pooling
            ResBlock(channels[0]),
        )
        self.conv5 = nn.Sequential(
            ResBlock(channels[0] * 2),
            nn.Conv2d(channels[0] * 2, channels[0], 1, 1, 0),  # 1x1 conv to reduce channels
            ResBlock(channels[0]),
            nn.Conv2d(channels[0], NUM_BANDS, 1, 1, 0),  # Final output layer
        )

    def forward(self, inputs):
        # Split Landsat input into LST (first channel) and spectral indices (remaining 3 channels)
        landsat_LST = inputs[1][:, 0:1, :, :]        # (B, 1, h, w)
        landsat_indices = inputs[1][:, 1:, :, :]     # (B, 3, h, w)

        # Get Sentinel spatial size
        target_H, target_W = inputs[2].shape[-2:]

        # Upsample Landsat LST and indices to match Sentinel resolution
        landsat_LST = F.interpolate(landsat_LST, size=(target_H, target_W), mode='bicubic', align_corners=False)
        landsat_indices = F.interpolate(landsat_indices, size=(target_H, target_W), mode='bicubic', align_corners=False)


        # Extract features from each input using the corresponding sub-networks

        LS2_List = self.MODIS_SNet(inputs[0])  # Modis image 1
        HS_List = self.Landsat_SNet(landsat_LST)  # Landsat image
        HS_indices_LIST =  self.indices_SNet(landsat_indices)  # Landsat image
        SS1_List = self.indices_SNet(inputs[2])  # Sentinel image

        LS1_List = self.MODIS_SNet(inputs[3])  # Modis image 2

        # Prepare to construct fused output
        new_10mHS_list = [] 

        # For each level of multi-scale features, compute similarity between Landsat and Sentinel indices.
        # Then use it to enhance the corresponding Landsat LST features.

        for HS1, HS1_indices, SS1 in zip(HS_List, HS_indices_LIST, SS1_List):
            refined_hs1 = self.similarity_refiner(HS1, HS1_indices, SS1)
            new_10mHS_list.append(refined_hs1)
     

        # Apply Adaptive Instance Normalization (AdaIN)
        SpecFeature_List = [
            adaptive_instance_normalization(HS, LS1) if self.ifAdaIN else HS
            for LS1, HS in zip(LS1_List, new_10mHS_list)
        ]

        # Perform significance extraction and fusion
        FusionFeature_List, M = [], []
        if not self.outputM:
            for SignE, SpecFeature, LS1, LS2 in zip(self.SignE_List, SpecFeature_List, LS1_List, LS2_List):
                FusionFeature_List.append(SignE([LS1, LS2, SpecFeature]))
        else:
            for SignE, SpecFeature, LS1, LS2 in zip(self.SignE_List, SpecFeature_List, LS1_List, LS2_List):
                SignE_output0, SignE_output1 = SignE([LS1, LS2, SpecFeature])
                FusionFeature_List.append(SignE_output0)
                M.append(SignE_output1)

        # Decoder: progressively reconstruct output image
        l5 = self.conv1(torch.cat((FusionFeature_List[4], LS1_List[4]), dim=1))
        l4 = self.conv2(torch.cat((FusionFeature_List[3], l5), dim=1))
        l3 = self.conv3(torch.cat((FusionFeature_List[2], l4), dim=1))
        l2 = self.conv4(torch.cat((FusionFeature_List[1], l3), dim=1))
        l1 = self.conv5(torch.cat((FusionFeature_List[0], l2), dim=1))
        
        if self.outputM == False:
            return l1
        else:
            return l1,M


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat  # Flag to return intermediate features
        self.n_layers = n_layers  # Number of convolutional layers

        kw = 4  # Kernel size for convolutions (4x4)
        padw = int(np.ceil((kw-1.0)/2))  # Padding to maintain spatial dimensions
        
        # First convolutional layer (no normalization)
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf  # Number of filters
        
        # Add intermediate convolutional layers
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)  # Double filters each time, capped at 512
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),  # Apply normalization
                nn.LeakyReLU(0.2, True)  # Activation function
            ]]
        
        # Final convolutional layer before classification
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),  # Stride 1 to maintain size
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        # Output layer (producing a 1-channel real/fake prediction map)
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        
        # Apply sigmoid activation if specified
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        # Store layers either as separate modules (for feature extraction) or as one sequential model
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))  # Save as individual sub-modules
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)  # Combine into one model

    def forward(self, input):

        if self.getIntermFeat:
            res = [input]  # Store input for intermediate feature extraction
            for n in range(self.n_layers+2):  # Iterate through stored models
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))  # Apply each model sequentially
            return res[1:]  # Return all intermediate features
        else:
            return self.model(input)  # If intermediate features are not needed, return final output
