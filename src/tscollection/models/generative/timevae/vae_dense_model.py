from torch import nn

from tscollection.models.generative.timevae.vae_base import BaseVariationalAutoencoder, Sampling


class DenseEncoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim):
        super().__init__()
        input_size = seq_len * feat_dim

        encoder_layers = []
        encoder_layers.append(nn.Flatten())

        for M_out in hidden_layer_sizes:
            encoder_layers.append(nn.Linear(input_size, M_out))
            encoder_layers.append(nn.ReLU())
            input_size = M_out

        self.encoder = nn.Sequential(*encoder_layers)
        self.z_mean = nn.Linear(input_size, latent_dim)
        self.z_log_var = nn.Linear(input_size, latent_dim)
        self.sampling = Sampling()

    def forward(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class DenseDecoder(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes, latent_dim):
        super().__init__()
        decoder_layers = []
        input_size = latent_dim
        self.seq_len = seq_len
        self.feat_dim = feat_dim

        for M_out in hidden_layer_sizes:
            decoder_layers.append(nn.Linear(input_size, M_out))
            decoder_layers.append(nn.ReLU())
            input_size = M_out

        decoder_layers.append(nn.Linear(input_size, seq_len * feat_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, z):
        decoder_output = self.decoder(z)
        reshaped_output = decoder_output.view(-1, self.seq_len, self.feat_dim)
        return reshaped_output


class VariationalAutoencoderDense(BaseVariationalAutoencoder):
    model_name = 'VAE_Dense'

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        latent_dim: int,
        reconstruction_wt: float = 3.0,
        learning_rate: float = 1e-3,
        hidden_layer_sizes: list | None = None,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            feat_dim=feat_dim,
            latent_dim=latent_dim,
            reconstruction_wt=reconstruction_wt,
            learning_rate=learning_rate,
        )
        self.save_hyperparameters()

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes

        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_encoder(self) -> nn.Module:
        return DenseEncoder(self.seq_len, self.feat_dim, self.hidden_layer_sizes, self.latent_dim)

    def _get_decoder(self) -> nn.Module:
        return DenseDecoder(
            self.seq_len, self.feat_dim, list(reversed(self.hidden_layer_sizes)), self.latent_dim
        )
