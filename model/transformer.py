import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        """
        Initialize the Transformer model.

        Args:
            input_dim (int): The size of the input vocabulary.
            output_dim (int): The size of the output vocabulary.
            d_model (int): The number of expected features in the encoder/decoder inputs.
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of encoder layers.
            num_decoder_layers (int): The number of decoder layers.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(Transformer, self).__init__()

        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.trg_embedding = nn.Embedding(output_dim, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )

        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, trg):
        """
        Forward pass through the Transformer model.

        Args:
            src (Tensor): The source sequence.
            trg (Tensor): The target sequence.

        Returns:
            Tensor: The output sequence.
        """
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        trg = self.trg_embedding(trg) * math.sqrt(self.d_model)

        src = self.positional_encoding(src)
        trg = self.positional_encoding(trg)

        output = self.transformer(src, trg)

        output = self.fc_out(output)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize the Positional Encoding module.

        Args:
            d_model (int): The number of expected features in the input.
            dropout (float): The dropout value.
            max_len (int): The maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass through the Positional Encoding module.

        Args:
            x (Tensor): The input sequence.

        Returns:
            Tensor: The input sequence with positional encodings added.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
