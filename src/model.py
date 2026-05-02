import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Adds a fixed position-dependent vector to token embeddings.
    """

    def __init__(self, hidden_dim: int, max_len: int = 80):
        super().__init__()

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * (-math.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: [1, max_len, hidden_dim]
        pe = pe.unsqueeze(0)

        # Register as buffer so it moves with model.to(device),
        # but is not trained.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Tensor of same shape with positional encoding added.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class DyckTransformerClassifier(nn.Module):
    """
    Small BERT-style Transformer encoder for binary Dyck error detection.

    Input:
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]

    Output:
        logits: [batch_size, 2]
    """

    def __init__(
        self,
        vocab_size: int = 7,
        max_len: int = 80,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * hidden_dim

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.num_classes = num_classes

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_encoding = SinusoidalPositionalEncoding(
            hidden_dim=hidden_dim,
            max_len=max_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialise embeddings and classifier weights.
        Transformer layers use PyTorch defaults.
        """
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        """
        Args:
            input_ids:
                Tensor of shape [batch_size, seq_len].
            attention_mask:
                Tensor of shape [batch_size, seq_len].
                1 = real token, 0 = padding.
            return_hidden:
                If True, also return the final hidden states.

        Returns:
            logits, or (logits, hidden_states) if return_hidden=True.
        """
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        x = self.dropout(x)

        src_key_padding_mask = None

        if attention_mask is not None:
            # PyTorch Transformer expects True for positions to ignore.
            src_key_padding_mask = attention_mask == 0

        hidden_states = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )

        # [CLS] is the first token.
        cls_representation = hidden_states[:, 0, :]

        logits = self.classifier(self.dropout(cls_representation))

        if return_hidden:
            return logits, hidden_states

        return logits


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Small sanity check.
    batch_size = 4
    seq_len = 80
    vocab_size = 7

    model = DyckTransformerClassifier(
        vocab_size=vocab_size,
        max_len=seq_len,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    logits = model(input_ids, attention_mask)

    print("Logits shape:", logits.shape)
    print("Number of trainable parameters:", count_parameters(model))

    assert logits.shape == (batch_size, 2)

    print("Model sanity check passed.")