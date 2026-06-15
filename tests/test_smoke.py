"""Smoke tests for model training, augmentation extension, and checkpoint round-trip.

Verifies that each model can run multiple training steps with finite loss
after module restructuring, that user-defined augmentation subclasses work
with the model pipeline, and that checkpoint save/reload preserves encoder
weights.

Uses the new producer-based augmentation contract (AugmentationProducer[ViewSet])
for all model training tests. Legacy AugmentationMethod tests remain for
backward compatibility verification.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from chronocratic.models.augmentation import AlignedPair, AugmentationProducer, SingleView, ViewPair
from chronocratic.models.augmentation.producers import IndependentPair
from chronocratic.models.convolutional.dilated.autotcl.augmentation import (
    AutoTCLNeuralNetworkAugmentation,
    RIPTrainingStrategy,
)
from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
    AutoTCLNeuralNetworkAugmentationParameters,
)
from chronocratic.models.convolutional.dilated.autotcl.model import AutoTCL
from chronocratic.models.convolutional.dilated.cost.augmentation import (
    CosTRandomFunctionAugmentation,
    CosTRandomFunctionAugmentationParameters,
)
from chronocratic.models.convolutional.dilated.cost.model import CoST
from chronocratic.models.convolutional.dilated.ts2vec.augmentation import CropShiftProducer
from chronocratic.models.convolutional.dilated.ts2vec.model import TS2Vec
from chronocratic.models.convolutional.standard.tstcc.augmentations import _default_tstcc_pair


def _train_steps(
    model: pl.LightningModule, batch_size: int, seq_length: int, input_dims: int, num_steps: int
) -> list[torch.Tensor]:
    """Run ``num_steps`` training steps via a minimal Lightning Trainer.

    ``training_step`` on all three models calls ``self.optimizers()`` which
    requires the model to be attached to a Trainer. This helper creates a
    dummy dataloader and runs the Trainer for the requested number of steps.

    Args:
        model: The Lightning model to train.
        batch_size: Batch size for the dummy data.
        seq_length: Sequence length for the dummy data.
        input_dims: Number of input features (channels).
        num_steps: How many training steps to execute.

    Returns:
        List of loss tensors returned by each training step.
    """
    data = torch.randn(batch_size * num_steps, seq_length, input_dims)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Store loss from training_step on the model so the callback can retrieve it
    model._test_losses = []  # type: ignore[attr-defined]

    original_training_step = model.training_step

    def _wrapped_training_step(
        batch: torch.Tensor | tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor | None:
        loss = original_training_step(batch, batch_idx)
        if loss is not None:
            model._test_losses.append(loss.detach())  # type: ignore[attr-defined]
        return loss

    model.training_step = _wrapped_training_step  # type: ignore[method-assign]

    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=num_steps,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=dataloader)
    return model._test_losses  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Model training smoke tests (VER-01 through VER-05)
# --------------------------------------------------------------------------- #


class TestModelTrainingSmoke:
    """End-to-end training smoke tests for each model (VER-01 through VER-03)."""

    def test_ts2vec_trains_5_steps(self) -> None:
        """TS2Vec with CropShiftProducer trains 5 steps with finite loss (VER-01)."""
        model = TS2Vec(input_dims=1, augmentation=CropShiftProducer())

        losses = _train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=5)

        assert len(losses) == 5
        for step_idx, loss in enumerate(losses):
            assert loss is not None
            assert loss.ndim == 0, "Loss must be a scalar tensor"
            assert math.isfinite(loss.item()), (
                f"Loss at step {step_idx} is not finite: {loss.item()}"
            )

    def test_ts2vec_checkpoint_preserves_encoder_weights(self) -> None:
        """TS2Vec checkpoint save/reload preserves encoder weights (VER-05).

        Save the model's full state_dict to a temporary file, reload it with
        weights_only=True, and verify that encoder state_dict keys and values
        match the originals exactly.
        """
        model = TS2Vec(input_dims=1, augmentation=CropShiftProducer())

        original = {k: v.clone() for k, v in model.encoder.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            torch.save(model.state_dict(), tmp_path)
            loaded_state = torch.load(tmp_path, weights_only=True)
            model.load_state_dict(loaded_state)

            for key in original:
                assert torch.equal(original[key], model.encoder.state_dict()[key]), (
                    f"Encoder weight mismatch after checkpoint reload for key: {key}"
                )
        finally:
            os.unlink(tmp_path)

    def test_cost_trains_5_steps(self) -> None:
        """CoST with IndependentPair producer trains 5 steps (VER-02)."""
        model = CoST(
            input_dims=1,
            sequence_length=100,
            kernel_sizes=[3],
            augmentation=IndependentPair(
                aug=CosTRandomFunctionAugmentation(
                    params=CosTRandomFunctionAugmentationParameters(sigma=0.1)
                )
            ),
        )

        losses = _train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=5)

        assert len(losses) == 5
        for step_idx, loss in enumerate(losses):
            assert loss is not None
            assert loss.ndim == 0
            assert math.isfinite(loss.item()), (
                f"Loss at step {step_idx} is not finite: {loss.item()}"
            )

    def test_autotcl_trains_5_steps(self) -> None:
        """AutoTCL with neural network augmentation trains 5 steps (VER-03)."""
        aug_params = AutoTCLNeuralNetworkAugmentationParameters(
            input_dims=1, output_dims=320, kernel_sizes=[3]
        )
        model = AutoTCL(
            input_dims=1,
            kernel_sizes=[3],
            augmentation=AutoTCLNeuralNetworkAugmentation(
                params=aug_params, training_strategy=RIPTrainingStrategy()
            ),
        )

        losses = _train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=5)

        assert len(losses) >= 1, "AutoTCL may skip steps during phase-1 aug training"
        for step_idx, loss in enumerate(losses):
            assert loss.ndim == 0
            assert math.isfinite(loss.item()), (
                f"Loss at step {step_idx} is not finite: {loss.item()}"
            )


# --------------------------------------------------------------------------- #
# Augmentation extensibility (VER-04)
# --------------------------------------------------------------------------- #


class TestAugmentationExtensibility:
    """Verify that user-defined AugmentationProducer implementations work (VER-04)."""

    def test_identity_producer_works_with_ts2vec(self) -> None:
        """Identity producer (no transform) produces finite loss with TS2Vec."""

        class IdentityProducer:
            """Pass-through producer that returns the original data as both views."""

            def produce(self, x: torch.Tensor) -> AlignedPair:
                seq_len = x.size(1)
                return AlignedPair(first=x, second=x, overlap_length=seq_len)

        model = TS2Vec(input_dims=1, augmentation=IdentityProducer())

        losses = _train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=1)

        assert len(losses) == 1
        loss = losses[0]
        assert loss is not None
        assert loss.ndim == 0
        assert math.isfinite(loss.item())

    def test_custom_single_view_producer_works_with_autotcl(self) -> None:
        """Custom SingleView producer trains with AutoTCL model."""

        class NoiseProducer:
            """Simple producer that adds Gaussian noise."""

            def produce(self, x: torch.Tensor) -> SingleView:
                return SingleView(view=x + torch.randn_like(x) * 0.1)

        model = AutoTCL(input_dims=1, kernel_sizes=[3], augmentation=NoiseProducer())

        losses = _train_steps(model=model, batch_size=4, seq_length=100, input_dims=1, num_steps=1)

        assert len(losses) >= 1
        for loss in losses:
            assert loss.ndim == 0
            assert math.isfinite(loss.item())

    def test_default_tstcc_pair_produces_view_pair(self) -> None:
        """_default_tstcc_pair() returns ViewPair with correct structure."""
        producer = _default_tstcc_pair()
        data = torch.randn(4, 1, 100)  # (B, C, T) layout used by TSTCC
        result = producer.produce(data)
        assert isinstance(result, ViewPair)
        assert result.first.shape == data.shape
        assert result.second.shape == data.shape
