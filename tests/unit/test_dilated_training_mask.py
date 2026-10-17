"""Dilated encoders must draw the same training masks as their upstream code."""

from collections import Counter
from collections.abc import Callable

import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from chronocratic.models import AutoTCL, CoST, TS2Vec
import chronocratic.models.convolutional.dilated.encoders.encoders as encoders_module
from chronocratic.models.convolutional.dilated.encoders.masking import MaskMode

type DilatedModel = TS2Vec | AutoTCL | CoST


def _record_masks(
    *, monkeypatch: pytest.MonkeyPatch, model: DilatedModel
) -> tuple[Counter[MaskMode], Counter[MaskMode]]:
    """Train ``model`` on two batches, then encode; return the masks drawn in each."""
    drawn: Counter[MaskMode] = Counter()
    generate_mask = encoders_module.generate_mask

    def recording_generate_mask(*, x: torch.Tensor, mask_mode: MaskMode) -> torch.Tensor:
        drawn[MaskMode(mask_mode)] += 1
        return generate_mask(x=x, mask_mode=mask_mode)

    monkeypatch.setattr(encoders_module, "generate_mask", recording_generate_mask)
    data = torch.randn(8, 64, 2)
    pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    ).fit(model=model, train_dataloaders=DataLoader(TensorDataset(data), batch_size=4))
    training = drawn.copy()
    drawn.clear()
    model.encode(data, batch_size=4, num_workers=0)
    return training, drawn


@pytest.mark.parametrize(
    ("build", "expected_training"),
    [
        # Upstream TS2Vec: forward(mask=None) applies mask_mode while training.
        (lambda: TS2Vec(input_dim=2), {MaskMode.BINOMIAL}),
        # Upstream AutoTCL: the encoder step passes mask=None (binomial); the
        # augmentation step passes 'all_true' with the encoder in eval.
        (lambda: AutoTCL(input_dim=2), {MaskMode.BINOMIAL, MaskMode.ALL_TRUE}),
        # Upstream CoST: forward defaults to mask='all_true' and training passes none.
        (lambda: CoST(input_dim=2, sequence_length=64), {MaskMode.ALL_TRUE}),
    ],
    ids=["ts2vec", "autotcl", "cost"],
)
def test_training_mask_matches_upstream(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], DilatedModel],
    expected_training: set[MaskMode],
) -> None:
    """Training draws the upstream mask modes; encoding never masks."""
    training, encoding = _record_masks(monkeypatch=monkeypatch, model=build())

    assert set(training) == expected_training
    assert set(encoding) == {MaskMode.ALL_TRUE}
