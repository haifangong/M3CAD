import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from dataset import idx2ama


def random_mask_and_shift(x, mask_prob=0.1, shift_range=(1, 20)):
    """
    Randomly masks and shifts elements in the input tensor x.

    Parameters:
        x (torch.Tensor): Input tensor of shape [batch_size, sequence_length].
                          Contains integer values from 0 to 21.
        mask_prob (float, optional): Probability of shifting each eligible position. Default is 0.1 (10%).
        shift_range (tuple, optional): Range of integers to shift to, inclusive. Default is (1, 20).

    Returns:
        torch.Tensor: Shifted tensor of shape [batch_size, sequence_length].
    """
    # Ensure x is of type long
    x = x.long()

    # Step 1: Create a mask for eligible positions (values not equal to 0 or 21)
    eligible_mask = (x != 0) & (x != 21)  # Shape: [batch_size, sequence_length]

    # Step 2: Generate a random mask based on the specified masking probability
    random_probs = torch.rand_like(x, dtype=torch.float)  # Uniformly distributed between 0 and 1
    shift_mask = (random_probs < mask_prob) & eligible_mask  # Shape: [batch_size, sequence_length]

    # Step 3: Generate random values within the specified shift range
    # These values will replace the original values at masked positions
    random_values = torch.randint(
        low=shift_range[0],
        high=shift_range[1] + 1,  # +1 because the upper bound in torch.randint is exclusive
        size=x.shape,
        device=x.device
    )  # Shape: [batch_size, sequence_length]

    # Step 4: Apply the shifts using torch.where
    # Wherever shift_mask is True, replace x with random_values; else, keep original x
    x_shifted = torch.where(shift_mask, random_values, x)  # Shape: [batch_size, sequence_length]

    return x_shifted


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion)
            )

    def forward(self, x):
        shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes, embedding_dim=16):
        super().__init__()
        self.csn = nn.Linear(num_classes, 1)
        self.in_channels = embedding_dim
        self.conv1 = nn.Conv3d(3, self.in_channels, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3))
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.layer1 = self._make_layer(block, self.in_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, self.in_channels, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_channels, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.in_channels, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.im_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(3, expand=False, center=None)
        ])

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x, y, debug=False):
        out_list = []
        # y_ = torch.sigmoid(self.csn(y))
        y_ = self.csn(y)
        voxel = x

        # add conditional information
        y_ = y_.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        voxel = voxel * y_

        # adding random noisy
        # voxel = voxel + (0.1 ** 0.5) * torch.randn(voxel.shape).cuda()
        # for i in range(voxel.shape[0]):
        #     voxel[i, ...] = self.im_aug(voxel[i])
        # print(voxel.shape)
        out = self.conv1(voxel)
        out = self.bn1(out)
        out = self.relu(out)
        out_list.append(out)
        if debug:
            print("shape1:", out.shape)
        out = self.max_pool(out)
        out = self.layer1(out)
        out_list.append(out)
        if debug:
            print("shape2:", out.shape)
        out = self.layer2(out)
        out_list.append(out)
        if debug:
            print("shape3:", out.shape)
        out = self.layer3(out)
        out_list.append(out)
        if debug:
            print("shape4:", out.shape)
        out = self.layer4(out)
        if debug:
            print("shape5:", out.shape)
        out_list.append(out)

        out = self.avg_pool(out)
        out_list.append(out)

        if debug:
            print("shape7:", out.shape)
            for item in out_list:
                print(item.shape)
        return out_list


def resnet26(num_classes, embedding_dim):
    return ResNet3D(Bottleneck, [1, 2, 4, 1], num_classes=num_classes, embedding_dim=embedding_dim)


def resnet50(num_classes, embedding_dim):
    return ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, embedding_dim=embedding_dim)


def resnet101(num_classes, embedding_dim):
    return ResNet3D(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, embedding_dim=embedding_dim)


def resnet152(num_classes, embedding_dim):
    return ResNet3D(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, embedding_dim=embedding_dim)


class VoxDecoder(nn.Module):
    def __init__(self, latent_dim=32, use_encoder_features=True):
        super().__init__()
        self.use_encoder_features = use_encoder_features
        # Define the channel dimensions based on latent_dim
        ch1 = latent_dim  # 32
        ch2 = latent_dim * 2  # 64
        ch3 = latent_dim * 4  # 128
        ch4 = latent_dim * 8  # 256
        ch5 = latent_dim * 16  # 512
        ch6 = latent_dim * 16  # 512

        if self.use_encoder_features:
            # Define convolutional blocks that include encoder features
            self.conv5 = self._conv_block(ch6 + ch5, ch5)
            self.conv4 = self._conv_block(ch5 + ch4, ch4)
            self.conv3 = self._conv_block(ch4 + ch3, ch3)
            self.conv2 = self._conv_block(ch3 + ch2, ch2)
            self.conv1 = self._conv_block(ch2 + ch1, ch1)
        else:
            # Define convolutional blocks without encoder features
            self.conv5 = self._conv_block(ch6, ch5)
            self.conv4 = self._conv_block(ch5, ch4)
            self.conv3 = self._conv_block(ch4, ch3)
            self.conv2 = self._conv_block(ch3, ch2)
            self.conv1 = self._conv_block(ch2, ch1)

        # Output convolution to get back to 3 channels
        self.out_conv = nn.Conv3d(ch1, 3, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        """Defines a convolutional block with two Conv3D layers."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, conditional_feature, encoder_feature=None):
        if self.use_encoder_features:
            x1, x2, x3, x4, x5, x6 = encoder_feature
            # Add the condition to x6
            x6 = x6 + conditional_feature.reshape(x6.shape[0], x6.shape[1], 1, 1, 1)
            # Decoder path with upsampling and skip connections
            x = F.interpolate(x6, size=x5.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat((x, x5), dim=1)  # Concatenate with x5
            x = self.conv5(x)

            x = F.interpolate(x, size=x4.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat((x, x4), dim=1)  # Concatenate with x4
            x = self.conv4(x)

            x = F.interpolate(x, size=x3.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat((x, x3), dim=1)  # Concatenate with x3
            x = self.conv3(x)

            x = F.interpolate(x, size=x2.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat((x, x2), dim=1)  # Concatenate with x2
            x = self.conv2(x)

            x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat((x, x1), dim=1)  # Concatenate with x1
            x = self.conv1(x)
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
            x = self.out_conv(x)  # Output layer
            return x
        else:
            # Use only conditional_feature to reconstruct the feature map
            x = conditional_feature
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

            x = self.conv5(x)
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)  # Upsample

            x = self.conv4(x)
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)  # Upsample

            x = self.conv3(x)
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)  # Upsample

            x = self.conv2(x)
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)  # Upsample

            x = self.conv1(x)
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)  # Upsample

            x = self.out_conv(x)  # Output layer
            return x


class SeqFCEncoder(nn.Module):
    def __init__(self, input_size=50, fc_hidden_size=256, embedding_dim=64, rnn_input_size=20, rnn_num_layers=1,
                 bidirectional=True):
        super().__init__()
        # Fully connected layers for processing the sequence
        self.seq_fc = nn.Sequential(
            nn.Linear(input_size, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, embedding_dim)
        )

        # Adjust the final output size based on bidirectionality
        self.output_size = embedding_dim * 2 if bidirectional else embedding_dim

    def forward(self, seq):
        """
        Args:
            seq: Tensor of shape (batch_size, 1, 50)
        Returns:
            seq_encoded: Tensor of shape (batch_size, output_size)
        """
        # Fully connected processing
        for i in range(seq.shape[0]):
            for j in range(50):
                if seq[i, 0, j] == -1:
                    break
                if random.random() < 0.2:
                    seq[i, 0, j] = 0
        seq_features = self.seq_fc(seq).squeeze(1)  # Shape: (batch_size, 64)
        return seq_features


class SeqFCDecoder(nn.Module):
    def __init__(self, in_dim=32, hid_dim=256, out_dim=50):
        super().__init__()

        self.seq_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, seq):
        seq_f = self.seq_fc(seq)
        return seq_f


class SeqEncoder(nn.Module):
    def __init__(
            self,
            num_classes=22,  # 20 amino acids + 1 for padding
            sequence_length=30,  # Maximum sequence length
            embedding_dim=64,  # Dimension of the final embedding
            gru_hidden_size=32,  # Hidden size for GRU
            gru_num_layers=2,  # Number of GRU layers
            bidirectional=False,  # Use bidirectional GRU
            dropout=0.2  # Dropout rate for GRU layers (only if num_layers > 1)
    ):
        """
        Initializes the SeqEncoder module.

        Args:
            num_classes (int): Number of classes for one-hot encoding (default: 21).
            sequence_length (int): Maximum length of the input sequences (default: 50).
            embedding_dim (int): Dimension of the output feature embedding (default: 64).
            gru_hidden_size (int): Number of features in the hidden state of the GRU (default: 128).
            gru_num_layers (int): Number of GRU layers (default: 1).
            bidirectional (bool): If True, becomes a bidirectional GRU (default: True).
            dropout (float): Dropout probability for GRU layers (default: 0.2).
        """
        super(SeqEncoder, self).__init__()

        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        # GRU layer configuration
        self.gru = nn.GRU(
            input_size=self.num_classes,  # One-hot vector size
            hidden_size=gru_hidden_size,  # GRU hidden state size
            num_layers=gru_num_layers,  # Number of stacked GRU layers
            batch_first=True,  # Input and output tensors are provided as (batch, seq, feature)
            bidirectional=bidirectional,  # Bidirectional GRU
            dropout=dropout if gru_num_layers > 1 else 0  # Apply dropout only if num_layers > 1
        )

        # Fully connected layer to map GRU output to embedding_dim
        gru_output_size = gru_hidden_size * 2 if bidirectional else gru_hidden_size
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, embedding_dim),
            nn.ReLU()
        )

    def forward(self, seq):
        """
        Forward pass of the SeqEncoder.

        Args:
            seq (torch.Tensor): Input tensor of shape (batch_size, 1, 50), containing integer values from -1 to 20.
                                -1 denotes padding, and values 1-20 correspond to amino acids.

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, embedding_dim).
        """
        batch_size = seq.size(0)

        # Remove the singleton dimension: (batch_size, 1, 50) -> (batch_size, 50)
        seq = seq.squeeze(1).long()  # Shape: (batch_size, 50)

        # Apply one-hot encoding: (batch_size, 50) -> (batch_size, 50, 21)
        one_hot = F.one_hot(seq, num_classes=self.num_classes).float()  # Shape: (batch_size, 50, 21)

        # Random Masking: Zero out entire amino acid positions with 20% probability
        # Create a mask tensor with shape (batch_size, 50, 1)
        mask_prob = 0.3
        mask = (torch.rand(batch_size, self.sequence_length, 1, device=seq.device) > mask_prob).float()
        one_hot = one_hot * mask  # Shape: (batch_size, 50, 21)

        # Pass through GRU: (batch_size, 50, 21) -> (batch_size, 50, hidden_size * num_directions)
        gru_out, hidden = self.gru(one_hot)

        if self.gru.bidirectional:
            # Concatenate the final forward and backward hidden states
            # hidden shape: (num_layers * 2, batch_size, hidden_size)
            hidden_forward = hidden[0:hidden.size(0):2]  # Shape: (num_layers, batch_size, hidden_size)
            hidden_backward = hidden[1:hidden.size(0):2]  # Shape: (num_layers, batch_size, hidden_size)
            hidden_cat = torch.cat((hidden_forward, hidden_backward),
                                   dim=2)  # Shape: (num_layers, batch_size, hidden_size * 2)
            # Take the last layer's hidden state
            hidden_final = hidden_cat[-1]  # Shape: (batch_size, hidden_size * 2)
        else:
            # hidden shape: (num_layers, batch_size, hidden_size)
            hidden_final = hidden[-1]  # Shape: (batch_size, hidden_size)
        # print('gru_out.shape', gru_out.shape)
        # Pass through the fully connected layer and activation
        seq_encoded = self.fc(hidden_final)  # Shape: (batch_size, embedding_dim)

        return seq_encoded, gru_out


class SeqDecoder(nn.Module):
    def __init__(
            self,
            embedding_dim=32,  # Dimension of the condition embedding
            gru_hidden_size=32,  # Hidden size for GRU
            gru_num_layers=1,  # Number of GRU layers
            output_dim=22,  # Number of classes (22 classes)
            sequence_length=30,  # Sequence length
            dropout=0.2  # Dropout rate for GRU layers (if num_layers > 1)
    ):
        """
        Initializes the SeqDecoder module with condition injection.

        Args:
            embedding_dim (int): Dimension of the condition embedding.
            gru_hidden_size (int): Hidden size for GRU.
            gru_num_layers (int): Number of GRU layers.
            output_dim (int): Number of output classes (e.g., 22 classes).
            sequence_length (int): Length of the input/output sequences.
            dropout (float): Dropout probability for GRU layers (only if num_layers > 1).
        """
        super(SeqDecoder, self).__init__()

        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.output_dim = output_dim
        self.sequence_length = sequence_length

        # Fully Connected Layer to map condition embedding to match seq_one_hot shape
        self.condition_proj = nn.Linear(embedding_dim, sequence_length * output_dim)

        # GRU Layer
        self.gru = nn.GRU(
            input_size=output_dim,  # One-hot input dimension
            hidden_size=self.gru_hidden_size,  # Hidden state size
            num_layers=self.gru_num_layers,  # Number of GRU layers
            batch_first=True,  # Input and output tensors are (batch, seq, feature)
            dropout=dropout if gru_num_layers > 1 else 0  # Apply dropout only if num_layers > 1
        )

        # Fully Connected Layer to map GRU outputs to output classes
        self.fc = nn.Linear(self.gru_hidden_size, self.output_dim)

        # Linear layer to project condition embedding to initial hidden state
        self.hidden_init = nn.Linear(embedding_dim, self.gru_hidden_size)

    def forward(self, seq_one_hot, embedding):
        """
        Forward pass of the SeqDecoder with condition injection.

        Args:
            seq_one_hot (torch.Tensor): One-hot encoded input sequences of shape (batch_size, sequence_length, output_dim).
            embedding (torch.Tensor): Condition embedding of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_dim),
                          representing the logits for each class at each position.
        """
        batch_size = embedding.size(0)
        # Project the condition embedding to match the seq_one_hot shape
        # Shape after projection: (batch_size, sequence_length * output_dim)
        cond_proj = torch.tanh(self.condition_proj(embedding))

        # Reshape to (batch_size, sequence_length, output_dim)
        cond_proj = cond_proj.view(batch_size, self.sequence_length, self.output_dim)

        # Inject condition into the input sequence embedding via element-wise addition
        # Shape: (batch_size, sequence_length, output_dim)
        input_emb = seq_one_hot + cond_proj

        # Initialize hidden state from condition embedding
        # Shape after projection: (batch_size, gru_hidden_size)
        hidden = torch.tanh(self.hidden_init(embedding))

        # Reshape to (num_layers, batch_size, gru_hidden_size)
        hidden = hidden.unsqueeze(0).repeat(self.gru_num_layers, 1, 1)

        # Pass through GRU
        # gru_output shape: (batch_size, sequence_length, gru_hidden_size)
        gru_output, _ = self.gru(input_emb, hidden)

        # Pass through the fully connected layer to get logits
        # seq_logits shape: (batch_size, sequence_length, output_dim)
        seq_logits = self.fc(gru_output)

        return seq_logits


class SEQVAE(nn.Module):
    """
    Sequence-based Variational Autoencoder (SEQVAE).
    
    This model encodes peptide sequences into a latent space and reconstructs them, 
    conditioned on properties (classes). It uses a sequence encoder and decoder 
    with a classification head.
    """
    def __init__(self, classes, latent_size=32, sequence_length=30, use_augmentation=True, mask_prob=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.sequence_length = sequence_length
        self.use_augmentation = use_augmentation
        self.mask_prob = mask_prob
        # Initialize the sequential encoder
        self.seq_encoder = SeqEncoder(sequence_length=sequence_length, embedding_dim=latent_size)
        self.cls_head = nn.Sequential(
            nn.Linear(sequence_length, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size * 4),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(latent_size * 4, classes),
        )

        # Multi-modal fusion adjusted to exclude voxel embedding dimensions
        self.multi_modal_fusion = nn.Sequential(
            nn.Linear(latent_size + classes, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size),
        )

        # Initialize the sequential decoder
        self.seq_decoder = SeqDecoder(sequence_length=sequence_length)

        # Conditional network adjusted to exclude voxel embedding dimensionsnn.LSTM
        self.csn = nn.Linear(latent_size + classes, latent_size)

        # Layers to compute mean and log variance for the latent space
        self.linear_means = nn.Linear(latent_size, latent_size)
        self.linear_log_var = nn.Linear(latent_size, latent_size)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample z from N(mu, var).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c):
        """
        Generate sequences from latent vector z and condition c.
        
        Args:
            z (torch.Tensor): Latent vector.
            c (torch.Tensor): Condition vector (e.g., target properties).
            
        Returns:
            tuple: (Indices of generated sequence, String representation, Raw indices tensor)
        """
        batch_size = z.size(0)
        device = z.device

        # Compute condition embedding
        cond_f = self.csn(torch.cat((z, c), dim=-1))  # Shape: (batch_size, embedding_dim)

        # allowed_indices = torch.tensor([1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], device=device)
        # allowed_indices = torch.tensor([1, 5, 8, 9, 10, 15, 18, 19, 20], device=device)
        allowed_indices = torch.tensor([1, 5, 8, 9, 10, 11, 13, 14, 15, 18, 19, 20], device=device)
        num_allowed = allowed_indices.size(0)

        # Define minimum and maximum sequence lengths
        min_len = self.sequence_length // 2
        max_len = self.sequence_length - 1

        # Randomly generate sequence lengths for the batch
        lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)  # Shape: (batch_size,)

        # Sample amino acid indices for all sequences up to max_len
        sampled = torch.randint(0, num_allowed, (batch_size, max_len), device=device)
        sampled_aas = allowed_indices[sampled]  # Shape: (batch_size, max_len)

        # Initialize sequences with PAD (0)
        sequences = torch.zeros(batch_size, self.sequence_length, dtype=torch.long, device=device)

        # Create a mask to determine where to place sampled AAs
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, max_len)
        mask = positions < lengths.unsqueeze(1)

        # Assign sampled AAs to the sequences
        sequences[:, :max_len][mask] = sampled_aas[mask]

        # Insert END token (21) at the position right after the last AA if there's space
        end_positions = lengths
        batch_indices = torch.arange(batch_size, device=device)
        sequences[batch_indices, end_positions] = 21  # END token

        # Convert numerical sequences to string representations
        sequences_strings = [
            ''.join([idx2ama[idx] for idx in seq[:lengths[0]-1].tolist()]) for seq in sequences
        ]

        # One-hot encode the sequences
        seq_one_hot = F.one_hot(sequences, num_classes=22).float()  # Shape: (batch_size, sequence_length, 22)

        # Pass the one-hot encoded sequences and condition embeddings to the seq_decoder
        recon_seq = self.seq_decoder(seq_one_hot, cond_f)  # Shape: (batch_size, sequence_length, output_dim)
        recon_seq_indices = torch.argmax(recon_seq, dim=-1)  # Shape: (batch_size, sequence_length)

        return recon_seq_indices, sequences_strings, sequences


    def forward(self, x, y):
        """
        Forward pass for training.
        
        Args:
            x (tuple): Input data (voxel, seq). voxel is ignored in SEQVAE.
            y (torch.Tensor): Ground truth labels/properties.
            
        Returns:
            tuple: (Reconstructed sequence, Classification result, Mean, Log Var, Condition feature)
        """
        batch_size = x[1].shape[0]
        # Encode the sequential features
        seq_feature, gru_out = self.seq_encoder(x[1])
        classify_result = self.cls_head(x[1].squeeze(1))

        # Apply multi-modal fusion
        fusion_input = torch.cat((seq_feature, y), dim=1)  # Shape: (batch_size, latent_size + classes)
        fusion_feature = self.multi_modal_fusion(fusion_input)  # Shape: (batch_size, latent_size * 4)

        # Compute mean and log variance for the latent space
        mean = self.linear_means(fusion_feature.view(batch_size, -1))  # Shape: (batch_size, latent_size * 4)
        var = self.linear_log_var(fusion_feature.view(batch_size, -1))  # Shape: (batch_size, latent_size * 4)

        # Reparameterize to obtain latent vector z
        z = self.reparameterize(mean, var)  # Shape: (batch_size, latent_size * 4)

        # Compute conditional feature
        cond_f = self.csn(torch.cat((z, y), dim=-1))  # Shape: (batch_size, latent_size)

        # Decode the sequential reconstruction with sequence
        if self.use_augmentation:
            x_noisy = random_mask_and_shift(x[1].squeeze(1).long(), mask_prob=self.mask_prob)
        else:
            x_noisy = x[1].squeeze(1).long()
        seq_one_hot = F.one_hot(x_noisy, num_classes=22).float()  # Shape: (batch_size, 50, 21)
        seq_rec = self.seq_decoder(seq_one_hot, cond_f)  # Shape depends on SeqDecoder implementation

        return seq_rec, classify_result, mean, var, cond_f


class MMVAE(nn.Module):
    """
    Multimodal Variational Autoencoder (MMVAE).
    
    This model utilizes both voxel (3D structure) and sequence data. It encodes both modalities 
    into a shared latent space and reconstructs both, conditioned on properties.
    """
    def __init__(self, classes, latent_size=32, use_encoder_feature=True, sequence_length=30, use_augmentation=True, mask_prob=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.sequence_length = sequence_length
        self.use_augmentation = use_augmentation
        self.mask_prob = mask_prob
        self.vox_embedding_dim = latent_size * 16
        self.vox_encoder = resnet26(classes, embedding_dim=latent_size)
        self.seq_encoder = SeqEncoder(sequence_length=sequence_length, embedding_dim=latent_size)
        self.cls_head = nn.Sequential(
            nn.Linear(sequence_length, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size * 4),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(latent_size * 4, classes),
        )

        self.multi_modal_fusion = nn.Sequential(
            nn.Linear(self.vox_embedding_dim + latent_size + classes, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size),
        )
        if use_encoder_feature:
            self.vox_decoder = VoxDecoder(use_encoder_features=True)
        else:
            self.vox_decoder = VoxDecoder(use_encoder_features=False)

        self.seq_decoder = SeqDecoder(sequence_length=sequence_length)
        self.csn = nn.Linear(latent_size + classes, self.vox_embedding_dim + latent_size)
        self.linear_means = nn.Linear(latent_size, latent_size)
        self.linear_log_var = nn.Linear(latent_size, latent_size)

    def reparameterize(self, mu, log_var):
        """Reparameterize to sample z."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c):
        """
        Generate sequences from latent vector z and condition c.
        
        Args:
            z (torch.Tensor): Latent vector.
            c (torch.Tensor): Condition vector.
            
        Returns:
            tuple: (Indices of generated sequence, String representation, Raw indices tensor)
        """
        batch_size = z.size(0)
        device = z.device

        # Compute condition embedding
        cond_f = self.csn(torch.cat((z, c), dim=-1))  # Shape: (batch_size, embedding_dim)
        vox_cond, seq_cond = cond_f[..., :self.vox_embedding_dim], cond_f[..., self.vox_embedding_dim:]

        # Define allowed amino acid indices (excluding D=15 and E=16)
        # allowed_indices = torch.tensor([1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], device=device)
        # allowed_indices = torch.tensor([1, 5, 8, 9, 10, 15, 18, 19, 20], device=device)
        allowed_indices = torch.tensor([1, 5, 8, 9, 10, 11, 13, 14, 15, 18, 19, 20], device=device)
        num_allowed = allowed_indices.size(0)

        # Define minimum and maximum sequence lengths
        min_len = self.sequence_length // 2
        max_len = self.sequence_length - 1

        # Randomly generate sequence lengths for the batch
        lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)  # Shape: (batch_size,)

        # Sample amino acid indices for all sequences up to max_len
        sampled = torch.randint(0, num_allowed, (batch_size, max_len), device=device)
        sampled_aas = allowed_indices[sampled]  # Shape: (batch_size, max_len)

        # Initialize sequences with PAD (0)
        sequences = torch.zeros(batch_size, self.sequence_length, dtype=torch.long, device=device)

        # Create a mask to determine where to place sampled AAs
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, max_len)
        mask = positions < lengths.unsqueeze(1)

        # Assign sampled AAs to the sequences
        sequences[:, :max_len][mask] = sampled_aas[mask]

        # Insert END token (21) at the position right after the last AA if there's space
        end_positions = lengths
        batch_indices = torch.arange(batch_size, device=device)
        sequences[batch_indices, end_positions] = 21  # END token

        # Convert numerical sequences to string representations
        sequences_strings = [
            ''.join([idx2ama[idx] for idx in seq[:lengths[0]-1].tolist()]) for seq in sequences
        ]

        # One-hot encode the sequences
        seq_one_hot = F.one_hot(sequences, num_classes=22).float()  # Shape: (batch_size, sequence_length, 22)

        # Pass the one-hot encoded sequences and condition embeddings to the seq_decoder
        recon_seq = self.seq_decoder(seq_one_hot, seq_cond)  # Shape: (batch_size, sequence_length, output_dim)
        recon_seq_indices = torch.argmax(recon_seq, dim=-1)  # Shape: (batch_size, sequence_length)

        return recon_seq_indices, sequences_strings, sequences

    def inference_template(self, z, c, template):
        """
        Generate sequences based on a template.
        """
        batch_size = z.size(0)
        device = z.device

        # Compute condition embedding
        cond_f = self.csn(torch.cat((z, c), dim=-1))  # Shape: (batch_size, embedding_dim)
        vox_cond, seq_cond = cond_f[..., :self.vox_embedding_dim], cond_f[..., self.vox_embedding_dim:]

        sampled_aas = template.repeat(batch_size, 1)

        # Define minimum and maximum sequence lengths
        min_len = self.sequence_length // 2
        max_len = self.sequence_length - 1

        # Randomly generate sequence lengths for the batch
        lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)  # Shape: (batch_size,)

        # Initialize sequences with PAD (0)
        sequences = torch.zeros(batch_size, self.sequence_length, dtype=torch.long, device=device)

        # Create a mask to determine where to place sampled AAs
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, max_len)
        mask = positions < lengths.unsqueeze(1)

        # Assign sampled AAs to the sequences
        sequences[:, :max_len][mask] = sampled_aas[mask]

        # Insert END token (21) at the position right after the last AA if there's space
        end_positions = lengths
        batch_indices = torch.arange(batch_size, device=device)
        sequences[batch_indices, end_positions] = 21  # END token

        # Convert numerical sequences to string representations
        sequences_strings = [
            ''.join([idx2ama[idx] for idx in seq[:lengths[0]-1].tolist()]) for seq in sequences
        ]

        # One-hot encode the sequences
        seq_one_hot = F.one_hot(sequences, num_classes=22).float()  # Shape: (batch_size, sequence_length, 22)

        # Pass the one-hot encoded sequences and condition embeddings to the seq_decoder
        recon_seq = self.seq_decoder(seq_one_hot, seq_cond)  # Shape: (batch_size, sequence_length, output_dim)
        recon_seq_indices = torch.argmax(recon_seq, dim=-1)  # Shape: (batch_size, sequence_length)

        return recon_seq_indices, sequences_strings, sequences

    def forward(self, x, y):
        """
        Forward pass for MMVAE.
        
        Args:
            x (tuple): (voxel, seq)
            y (torch.Tensor): Labels
            
        Returns:
            tuple: ((Vox Recon, Seq Recon), Classification, Mean, Var, Condition)
        """
        batch_size = x[0].shape[0]
        vox_feature = self.vox_encoder(x[0], y)
        seq_feature, _ = self.seq_encoder(x[1])
        vox4mm = vox_feature[-1].view(batch_size, -1)

        fusion_feature_cls = torch.cat((vox4mm, seq_feature), dim=1)
        classify_result = self.cls_head(x[1].squeeze(1))

        fusion_feature_rec = torch.cat((vox4mm, seq_feature, y), dim=1)

        fusion_feature = self.multi_modal_fusion(fusion_feature_rec)

        mean = self.linear_means(fusion_feature.view(batch_size, -1))
        var = self.linear_log_var(fusion_feature.view(batch_size, -1))
        z = self.reparameterize(mean, var)
        cond_f = self.csn(torch.cat((z, y), dim=-1))
        vox_cond, seq_cond = cond_f[..., :self.vox_embedding_dim], cond_f[..., self.vox_embedding_dim:]
        vox_rec = self.vox_decoder(vox_cond.view(batch_size, self.vox_embedding_dim, 1, 1, 1), vox_feature)

        # Decode the sequential reconstruction with sequence
        if self.use_augmentation:
            x_noisy = random_mask_and_shift(x[1].squeeze(1).long(), mask_prob=self.mask_prob)
        else:
            x_noisy = x[1].squeeze(1).long()
        seq_one_hot = F.one_hot(x_noisy, num_classes=22).float()  # Shape: (batch_size, 50, 21)
        seq_rec = self.seq_decoder(seq_one_hot, seq_cond)  # Shape depends on SeqDecoder implementation
        return (vox_rec, seq_rec), classify_result, mean, var, cond_f


class SEQWAE(nn.Module):
    """
    Sequence-based Wasserstein Autoencoder (SEQWAE).
    
    Similar to SEQVAE but uses Wasserstein distance for the generative loss.
    """
    def __init__(self, classes, latent_size=32, sequence_length=30, use_augmentation=True, mask_prob=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.use_augmentation = use_augmentation
        self.mask_prob = mask_prob

        # Initialize the sequential encoder
        self.seq_encoder = SeqEncoder(sequence_length=sequence_length, embedding_dim=latent_size)

        # Multi-modal fusion adjusted to exclude voxel embedding dimensions
        self.multi_modal_fusion = nn.Sequential(
            nn.Linear(latent_size + classes, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size * 4),
        )

        # Initialize the sequential decoder
        self.seq_decoder = SeqDecoder(sequence_length=sequence_length)

        # Conditional network adjusted to exclude voxel embedding dimensions
        self.csn = nn.Linear(latent_size * 4 + classes, latent_size)

    def inference(self, z, c):
        """
        Generate sequences from latent vector z and condition c.
        """
        batch_size = z.size(0)
        device = z.device

        # Compute condition embedding
        cond_f = self.csn(torch.cat((z, c), dim=-1))  # Shape: (batch_size, embedding_dim)

        allowed_indices = torch.tensor([1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], device=device)
        num_allowed = allowed_indices.size(0)

        # Define minimum and maximum sequence lengths
        min_len = self.sequence_length // 2
        max_len = self.sequence_length - 1

        # Randomly generate sequence lengths for the batch
        lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)  # Shape: (batch_size,)

        # Sample amino acid indices for all sequences up to max_len
        sampled = torch.randint(0, num_allowed, (batch_size, max_len), device=device)
        sampled_aas = allowed_indices[sampled]  # Shape: (batch_size, max_len)

        # Initialize sequences with PAD (0)
        sequences = torch.zeros(batch_size, self.sequence_length, dtype=torch.long, device=device)

        # Create a mask to determine where to place sampled AAs
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, max_len)
        mask = positions < lengths.unsqueeze(1)

        # Assign sampled AAs to the sequences
        sequences[:, :max_len][mask] = sampled_aas[mask]

        # Insert END token (21) at the position right after the last AA if there's space
        end_positions = lengths
        batch_indices = torch.arange(batch_size, device=device)
        sequences[batch_indices, end_positions] = 21  # END token

        # Convert numerical sequences to string representations
        sequences_strings = [
            ''.join([idx2ama[idx] for idx in seq[:lengths[0]-1].tolist()]) for seq in sequences
        ]

        # One-hot encode the sequences
        seq_one_hot = F.one_hot(sequences, num_classes=22).float()  # Shape: (batch_size, sequence_length, 22)

        # Pass the one-hot encoded sequences and condition embeddings to the seq_decoder
        recon_seq = self.seq_decoder(seq_one_hot, cond_f)  # Shape: (batch_size, sequence_length, output_dim)
        recon_seq_indices = torch.argmax(recon_seq, dim=-1)  # Shape: (batch_size, sequence_length)

        return recon_seq_indices, sequences_strings, sequences

    def forward(self, x, y):
        """
        Forward pass for SEQWAE.
        
        Args:
            x (tuple): (voxel, seq)
            y (torch.Tensor): Labels
            
        Returns:
            tuple: (Seq Rec, Seq Feature, Latent z)
        """
        batch_size = x[1].shape[0]
        # Encode the sequential features
        seq_feature, _ = self.seq_encoder(x[1])

        # Fusion feature: Concatenate sequential feature with conditional input
        fusion_input = torch.cat((seq_feature, y), dim=1)  # Shape: (batch_size, latent_size + classes)
        # Apply multi-modal fusion
        fusion_feature = self.multi_modal_fusion(fusion_input)  # Shape: (batch_size, latent_size * 4)
        # Latent representation z
        z = fusion_feature.view(batch_size, -1)

        # Compute conditional feature
        cond_f = self.csn(torch.cat((z, y), dim=-1))  # Shape: (batch_size, latent_size)

        # Decode the sequential reconstruction with sequence
        if self.use_augmentation:
            x_noisy = random_mask_and_shift(x[1].squeeze(1).long(), mask_prob=self.mask_prob)
        else:
            x_noisy = x[1].squeeze(1).long()
        seq_one_hot = F.one_hot(x_noisy, num_classes=22).float()  # Shape: (batch_size, 50, 21)
        seq_rec = self.seq_decoder(seq_one_hot, cond_f)  # Shape depends on SeqDecoder implementation

        # For MMD computation, we'll need z and prior samples
        return seq_rec, seq_feature, z


class MMWAE(nn.Module):
    """
    Multimodal Wasserstein Autoencoder (MMWAE).
    
    Combines voxel and sequence data using WAE framework.
    """
    def __init__(self, classes, latent_size=32, use_encoder_feature=True, sequence_length=30, use_augmentation=True, mask_prob=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.vox_embedding_dim = latent_size * 16
        self.use_augmentation = use_augmentation
        self.mask_prob = mask_prob
        self.vox_encoder = resnet26(classes, embedding_dim=latent_size)
        self.seq_encoder = SeqEncoder(sequence_length=sequence_length, embedding_dim=latent_size)
        self.multi_modal_fusion = nn.Sequential(
            nn.Linear(self.vox_embedding_dim + latent_size + classes, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size * 4),
        )
        if use_encoder_feature:
            self.vox_decoder = VoxDecoder(use_encoder_features=True)
        else:
            self.vox_decoder = VoxDecoder(use_encoder_features=False)

        self.seq_decoder = SeqDecoder(sequence_length=sequence_length)

        self.csn = nn.Linear(latent_size * 4 + classes, self.vox_embedding_dim + latent_size)

    def inference(self, z, c):
        """
        Generate sequences from latent vector z and condition c.
        """
        batch_size = z.size(0)
        device = z.device

        # Compute condition embedding
        cond_f = self.csn(torch.cat((z, c), dim=-1))  # Shape: (batch_size, embedding_dim)

        # Define allowed amino acid indices (excluding D=15 and E=16)
        allowed_indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20], device=device)
        num_allowed = allowed_indices.size(0)

        # Define minimum and maximum sequence lengths
        min_len = self.sequence_length // 2
        max_len = self.sequence_length - 1

        # Randomly generate sequence lengths for the batch
        lengths = torch.randint(min_len, max_len + 1, (batch_size,), device=device)  # Shape: (batch_size,)

        # Sample amino acid indices for all sequences up to max_len
        sampled = torch.randint(0, num_allowed, (batch_size, max_len), device=device)
        sampled_aas = allowed_indices[sampled]  # Shape: (batch_size, max_len)

        # Initialize sequences with PAD (0)
        sequences = torch.zeros(batch_size, self.sequence_length, dtype=torch.long, device=device)

        # Create a mask to determine where to place sampled AAs
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, max_len)
        mask = positions < lengths.unsqueeze(1)

        # Assign sampled AAs to the sequences
        sequences[:, :max_len][mask] = sampled_aas[mask]

        # Insert END token (21) at the position right after the last AA if there's space
        end_positions = lengths
        batch_indices = torch.arange(batch_size, device=device)
        sequences[batch_indices, end_positions] = 21  # END token

        # Convert numerical sequences to string representations
        sequences_strings = [
            ''.join([idx2ama[idx] for idx in seq[:lengths[0]-1].tolist()]) for seq in sequences
        ]

        # One-hot encode the sequences
        seq_one_hot = F.one_hot(sequences, num_classes=22).float()  # Shape: (batch_size, sequence_length, 22)

        # Pass the one-hot encoded sequences and condition embeddings to the seq_decoder
        recon_seq = self.seq_decoder(seq_one_hot, cond_f)  # Shape: (batch_size, sequence_length, output_dim)
        recon_seq_indices = torch.argmax(recon_seq, dim=-1)  # Shape: (batch_size, sequence_length)

        return recon_seq_indices, sequences_strings, sequences


    def forward(self, x, y):
        """
        Forward pass for MMWAE.
        
        Args:
            x (tuple): (voxel, seq)
            y (torch.Tensor): Labels
            
        Returns:
            tuple: ((Vox Rec, Seq Rec), Fusion Feature, Latent z)
        """
        batch_size = x[0].shape[0]
        # Encode voxel data
        vox_feature = self.vox_encoder(x[0], y)
        # Encode sequential data
        seq_feature, _ = self.seq_encoder(x[1])

        vox4mm = vox_feature[-1].view(batch_size, -1)
        # Concatenate features and class labels
        fusion_feature_cls = torch.cat((vox4mm, seq_feature, y), dim=1)
        # Fuse features
        fusion_feature = self.multi_modal_fusion(fusion_feature_cls)
        # Latent representation z
        z = fusion_feature.view(batch_size, -1)

        # Compute conditional features for decoder
        cond_f = self.csn(torch.cat((z, y), dim=-1))
        vox_cond, seq_cond = cond_f[..., :self.vox_embedding_dim], cond_f[..., self.vox_embedding_dim:]

        # Decode voxel data
        vox_rec = self.vox_decoder(
            vox_cond.view(batch_size, self.vox_embedding_dim, 1, 1, 1), vox_feature
        )
        # Decode the sequential reconstruction with sequence
        if self.use_augmentation:
            x_noisy = random_mask_and_shift(x[1].squeeze(1).long(), mask_prob=self.mask_prob)
        else:
            x_noisy = x[1].squeeze(1).long()
        seq_one_hot = F.one_hot(x_noisy, num_classes=22).float()  # Shape: (batch_size, 50, 21)
        seq_rec = self.seq_decoder(seq_one_hot, seq_cond)  # Shape depends on SeqDecoder implementation

        # For MMD computation, we'll need z and prior samples
        return (vox_rec, seq_rec), fusion_feature_cls, z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GOOD')
    parser.add_argument('--gen_model', type=str, default='vae',
                        help='vae wae')
    parser.add_argument('--model', type=str, default='mm_unet',
                        help='model structure')
    args = parser.parse_args()

    number_classes = 2
    gt = torch.zeros((2, number_classes)).cuda()
    input_voxel = torch.zeros((2, 3, 64, 64, 64)).cuda()
    input_seq = torch.zeros((2, 1, 50)).cuda()

    if args.gen_model == 'vae':
        if args.model == 'seq':
            model = SEQVAE(number_classes).cuda()
        elif args.model == 'mm_unet':
            model = MMVAE(number_classes, use_encoder_feature=True).cuda()
        elif args.model == 'mm_mt':
            model = MMVAE(number_classes, use_encoder_feature=False).cuda()

    elif args.gen_model == 'wae':
        if args.model == 'seq':
            model = SEQWAE(number_classes).cuda()
        elif args.model == 'mm_unet':
            model = MMWAE(number_classes, use_encoder_feature=True).cuda()
        elif args.model == 'mm_mt':
            model = MMWAE(number_classes, use_encoder_feature=False).cuda()

    if args.gen_model == 'vae':
        if args.model == 'seq':
            seq_rec, fusion_feature_cls, mean, var, cond_f = model((input_voxel, input_seq), gt)
        else:
            (seq_rec, vox_rec), fusion_feature_cls, mean, var, cond_f = model((input_voxel, input_seq), gt)
            print(vox_rec.shape)
        print(seq_rec.shape)
    else:
        if args.model == 'seq':
            seq_rec, fusion_feature_cls, z = model((input_voxel, input_seq), gt)
        else:
            (seq_rec, vox_rec), fusion_feature_cls, z = model((input_voxel, input_seq), gt)
            print(vox_rec.shape)
        print(seq_rec.shape)

