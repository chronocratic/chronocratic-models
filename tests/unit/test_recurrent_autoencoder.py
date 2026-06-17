"""Smoke and unit tests for the RecurrentAutoEncoder recurrent autoencoder."""

from __future__ import annotations

import math

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from chronocratic.models.recurrent.recurrentae.model import RecurrentAutoEncoder


class _RawTensorDataset(Dataset):
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


@pytest.fixture(params=["LSTM", "GRU", "RNN"])
def rnn_type(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[return-value]


@pytest.fixture
def model(rnn_type: str) -> RecurrentAutoEncoder:
    return RecurrentAutoEncoder(n_features=3, layers=[8], recurrent_cell_type=rnn_type)


class TestRecurrentAutoEncoderInstantiation:
    def test_instantiates_lstm(self) -> None:
        m = RecurrentAutoEncoder(n_features=5, layers=[16])
        assert isinstance(m, RecurrentAutoEncoder)

    def test_instantiates_gru(self) -> None:
        m = RecurrentAutoEncoder(n_features=5, layers=[16], recurrent_cell_type="GRU")
        assert isinstance(m, RecurrentAutoEncoder)

    def test_instantiates_rnn(self) -> None:
        m = RecurrentAutoEncoder(n_features=5, layers=[16], recurrent_cell_type="RNN")
        assert isinstance(m, RecurrentAutoEncoder)

    def test_instantiates_multilayer(self) -> None:
        m = RecurrentAutoEncoder(n_features=5, layers=[64, 32, 16])
        assert isinstance(m, RecurrentAutoEncoder)

    def test_encoder_is_nn_module(self) -> None:
        from torch import nn

        m = RecurrentAutoEncoder(n_features=3, layers=[8])
        assert isinstance(m.encoder, nn.Module)

    def test_decoder_is_nn_module(self) -> None:
        from torch import nn

        m = RecurrentAutoEncoder(n_features=3, layers=[8])
        assert isinstance(m.decoder, nn.Module)


class TestEncoderOutputShape:
    def test_encoder_returns_full_sequence(self, model: RecurrentAutoEncoder) -> None:
        x = torch.randn(4, 20, 3)
        with torch.no_grad():
            out = model.encoder(x)
        assert out.shape == (4, 20, 8), f"Got {out.shape}"

    def test_encoder_output_batch_dim(self, model: RecurrentAutoEncoder) -> None:
        x = torch.randn(7, 15, 3)
        with torch.no_grad():
            out = model.encoder(x)
        assert out.shape[0] == 7

    def test_encoder_output_seq_dim(self, model: RecurrentAutoEncoder) -> None:
        x = torch.randn(2, 30, 3)
        with torch.no_grad():
            out = model.encoder(x)
        assert out.shape[1] == 30

    def test_encoder_output_latent_dim(self, model: RecurrentAutoEncoder) -> None:
        x = torch.randn(2, 10, 3)
        with torch.no_grad():
            out = model.encoder(x)
        assert out.shape[2] == 8

    def test_multilayer_encoder_output_dim(self) -> None:
        m = RecurrentAutoEncoder(n_features=5, layers=[64, 32])
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            out = m.encoder(x)
        assert out.shape == (2, 10, 32)


class TestDecoderOutputShape:
    def test_decoder_reconstructs_input_shape(self, model: RecurrentAutoEncoder) -> None:
        x = torch.randn(4, 20, 3)
        with torch.no_grad():
            encoded = model.encoder(x)
            decoded = model.decoder(torch.flip(encoded, dims=[1]))
        assert decoded.shape == x.shape

    def test_forward_output_shape_matches_input(self, model: RecurrentAutoEncoder) -> None:
        x = torch.randn(3, 12, 3)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape


class TestTSRCCompatibility:
    """Verify the student interface contract: encoder returns (B, T, D)."""

    def test_encoder_output_is_3d(self) -> None:
        model = RecurrentAutoEncoder(n_features=5, layers=[16])
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            r2 = model.encoder(x)
        assert r2.dim() == 3

    def test_last_timestep_indexing_works(self) -> None:
        model = RecurrentAutoEncoder(n_features=5, layers=[16])
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            r2 = model.encoder(x)
        repr_vec = r2[:, -1, :]
        assert repr_vec.shape == (2, 16)

    def test_flip_then_decode_matches_input_shape(self) -> None:
        model = RecurrentAutoEncoder(n_features=5, layers=[16])
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            r2 = model.encoder(x)
            x_hat = model.decoder(torch.flip(r2, dims=[1]))
        assert x_hat.shape == x.shape


class TestLightningTraining:
    def _train(self, model: RecurrentAutoEncoder, n_steps: int = 3) -> list[float]:
        data = torch.randn(n_steps * 4, 20, model.n_features)
        loader = DataLoader(_RawTensorDataset(data), batch_size=4)

        losses: list[float] = []
        orig = model.training_step

        def _capture(*args: object, **kwargs: object) -> torch.Tensor:
            loss = orig(*args, **kwargs)
            if loss is not None:
                losses.append(loss.detach().item())  # type: ignore[union-attr]
            return loss  # type: ignore[return-value]

        model.training_step = _capture  # type: ignore[method-assign]

        trainer = pl.Trainer(
            accelerator="cpu",
            max_steps=n_steps,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(model, train_dataloaders=loader)
        return losses

    def test_training_produces_finite_losses(self) -> None:
        model = RecurrentAutoEncoder(n_features=3, layers=[8])
        losses = self._train(model)
        assert len(losses) == 3
        for loss in losses:
            assert math.isfinite(loss), f"Non-finite loss: {loss}"

    def test_training_loss_is_scalar(self) -> None:
        model_local = RecurrentAutoEncoder(n_features=3, layers=[8])
        batch = torch.randn(4, 20, 3)
        loss = model_local.training_step(batch, 0)
        assert loss.ndim == 0

    def test_mae_loss_trains(self) -> None:
        model = RecurrentAutoEncoder(n_features=3, layers=[8], loss="mae")
        losses = self._train(model)
        for loss in losses:
            assert math.isfinite(loss)


class TestPostprocess:
    def test_postprocess_returns_last_timestep(self) -> None:
        model = RecurrentAutoEncoder(n_features=3, layers=[8])
        fake_output = torch.randn(2, 10, 8)
        result = model._postprocess(fake_output)
        assert result.shape == (2, 8)
        assert torch.equal(result, fake_output[:, -1, :])
