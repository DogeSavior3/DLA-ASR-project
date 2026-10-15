from torch import nn

from .blocks import ConvBlock, RNNGRUBlock


class DeepSpeech2Model(nn.Module):
    def __init__(self, n_feats, n_tokens, conv_params, rnn_gru_params):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            conv_params: parameters from Hydra for convolutional layers.
            rnn_gru_params: parameters from Hydra for recurrent GRU layers.
        """
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.rnn_layers = nn.ModuleList()
        in_channels = 1
        freq = n_feats

        for cfg in conv_params:
            self.conv_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=cfg.out_channels,
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride,
                    padding=cfg.padding,
                    dilation=cfg.dilation,
                )
            )
            in_channels = cfg.out_channels
            kernel = cfg.kernel_size[0]
            stride = cfg.stride[0]
            padding = cfg.padding[0]
            dilation = cfg.dilation[0]
            freq = (freq + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
            in_channels = cfg.out_channels

        rnn_input_size = in_channels * freq
        self.rnn_layers.append(
            RNNGRUBlock(
                input_size=rnn_input_size,
                hidden_size=rnn_gru_params.hidden_size,
                normalization=rnn_gru_params.normalization,
            )
        )

        for _ in range(1, rnn_gru_params.num_layers):
            self.rnn_layers.append(
                RNNGRUBlock(
                    input_size=2 * rnn_gru_params.hidden_size,
                    hidden_size=rnn_gru_params.hidden_size,
                    normalization=rnn_gru_params.normalization,
                )
            )
        self.fc = nn.Linear(2 * rnn_gru_params.hidden_size, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        output_lengths = spectrogram_length
        # spectogram: (B, F, L))
        x = spectrogram.unsqueeze(1)
        # x: (B, 1, F, L) 1 = C we always do one channel in dataset

        for block in self.conv_layers:
            x = block(x)
            conv = block.conv
            kernel = conv.kernel_size[1]
            stride = conv.stride[1]
            padding = conv.padding[1]
            dilation = conv.dilation[1]

            output_lengths = (
                output_lengths + 2 * padding - dilation * (kernel - 1) - 1
            ) // stride + 1

        # gru takes (B, L, input_size = F * C)
        B, C, F, L = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, L, C * F)

        for block in self.rnn_layers:
            x = block(x)

        logits = self.fc(x)
        log_probs = logits.log_softmax(dim=-1)

        return {"log_probs": log_probs, "log_probs_length": output_lengths}
