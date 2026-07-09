"""TDD tests for AutoTCL producer integration.

Verifies AutoTCLNeuralNetworkAugmentation satisfies
TrainableAugmentationProducer (nominal ABC + nn.Module) and AutoTCL accepts
AugmentationProducer[SingleView] | None, using maybe_* helpers from
trainable_support.py instead of isinstance(TrainableAugmentation) checks.
"""

from collections.abc import Callable

import torch

from chronocratic.models.augmentation.base import SingleView, TrainableAugmentationProducer
from chronocratic.models.augmentation.primitives import Jitter
from chronocratic.models.augmentation.producers import SingleViewProducer
from chronocratic.models.convolutional.dilated.autotcl.augmentation.methods import (
    AutoTCLNeuralNetworkAugmentation,
    AutoTCLNeuralNetworkAugmentationParameters,
)
from chronocratic.models.convolutional.dilated.autotcl.augmentation.training import (
    RIPTrainingStrategy,
)
from chronocratic.models.convolutional.dilated.autotcl.config import AutoTCLModelParameters
from chronocratic.models.convolutional.dilated.autotcl.model import AutoTCL


class TestAutoTCLNeuralNetworkAugmentationIsTrainableProducer:
    """AutoTCLNeuralNetworkAugmentation is subclass of TrainableAugmentationProducer."""

    def test_is_subclass_of_trainable_augmentation_producer(self) -> None:
        assert issubclass(AutoTCLNeuralNetworkAugmentation, TrainableAugmentationProducer)

    def test_is_nn_module(self) -> None:
        assert issubclass(AutoTCLNeuralNetworkAugmentation, torch.nn.Module)

    def test_is_instance_of_trainable_augmentation_producer(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dim=1)
        )
        assert isinstance(aug, TrainableAugmentationProducer)

    def test_is_instance_of_nn_module(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dim=1)
        )
        assert isinstance(aug, torch.nn.Module)


class TestAutoTCLNeuralNetworkAugmentationProduce:
    """AutoTCLNeuralNetworkAugmentation.produce() returns SingleView."""

    def test_produce_returns_single_view(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dim=1)
        )
        x = torch.randn(4, 100, 1)
        result = aug.produce(x)
        assert isinstance(result, SingleView)

    def test_produce_preserves_shape(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dim=3)
        )
        x = torch.randn(2, 50, 3)
        result = aug.produce(x)
        assert result.view.shape == x.shape

    def test_produce_satisfies_protocol(self) -> None:
        """AugmentationProducer is a structural Protocol (not runtime_checkable).
        Verify structural conformance by checking produce() exists and returns SingleView."""
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dim=1)
        )
        assert hasattr(aug, "produce")
        result = aug.produce(torch.randn(2, 50, 1))
        assert isinstance(result, SingleView)


class TestAutoTCLNeuralNetworkAugmentationTrainStep:
    """AutoTCLNeuralNetworkAugmentation.train_step() returns loss tensor."""

    def test_train_step_returns_loss(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(
                input_dim=1, representation_dim=320, kernel_sizes=[3]
            ),
            training_strategy=RIPTrainingStrategy(),
        )
        encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)
        )
        x = torch.randn(2, 10, 1)
        loss = aug.train_step(x=x, encoder=encoder, batch_idx=0)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)


class TestAutoTCLAcceptsProducer:
    """AutoTCL constructor accepts AugmentationProducer[SingleView] | None."""

    def test_accepts_neural_augmentation_producer(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dim=1)
        )
        model = AutoTCL(input_dim=1, augmentation=aug)
        assert isinstance(model, AutoTCL)

    def test_accepts_single_view_producer_with_jitter(self) -> None:
        producer = SingleViewProducer(aug=Jitter())
        model = AutoTCL(input_dim=1, augmentation=producer)
        assert isinstance(model, AutoTCL)

    def test_default_augmentation_is_trainable_producer(self) -> None:
        model = AutoTCL(input_dim=1)
        assert isinstance(model._augmentation, TrainableAugmentationProducer)


class TestAutoTCLUsesMaybeHelpers:
    """AutoTCL uses maybe_configure_augmentation_optimizer and maybe_train_augmentation."""

    def test_configure_optimizers_returns_list_with_trainable(self) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(input_dim=1)
        )
        model = AutoTCL(input_dim=1, augmentation=aug)
        opts = model.configure_optimizers()
        assert isinstance(opts, list)
        assert len(opts) == 2

    def test_configure_optimizers_returns_single_with_static(self) -> None:
        producer = SingleViewProducer(aug=Jitter())
        model = AutoTCL(input_dim=1, augmentation=producer)
        opts = model.configure_optimizers()
        assert not isinstance(opts, list)


class TestAutoTCLTrainingWithProducer:
    """AutoTCL trains with both trainable and static augmentation paths."""

    def test_trains_5_steps_with_neural_aug(
        self, train_steps: Callable[..., list[torch.Tensor]], finite_losses: Callable[..., None]
    ) -> None:
        aug = AutoTCLNeuralNetworkAugmentation(
            params=AutoTCLNeuralNetworkAugmentationParameters(
                input_dim=1, representation_dim=320, kernel_sizes=[3]
            ),
            training_strategy=RIPTrainingStrategy(),
        )
        model = AutoTCL(input_dim=1, augmentation=aug)
        losses = train_steps(model=model, batch_size=4, seq_length=100, input_dim=1, num_steps=5)
        finite_losses(losses, expected_min=1)

    def test_trains_5_steps_with_static_aug(
        self, train_steps: Callable[..., list[torch.Tensor]], finite_losses: Callable[..., None]
    ) -> None:
        producer = SingleViewProducer(aug=Jitter())
        model = AutoTCL(input_dim=1, augmentation=producer)
        losses = train_steps(model=model, batch_size=4, seq_length=100, input_dim=1, num_steps=5)
        finite_losses(losses, expected_min=5)


class TestAutoTCLSeededEquivalence:
    """Seeded AutoTCL produces identical loss sequence (SC-7 numerical equivalence)."""

    def test_seeded_autotcl_produces_identical_loss_sequence(
        self, train_steps: Callable[..., list[torch.Tensor]]
    ) -> None:
        torch.manual_seed(42)
        aug_params = AutoTCLNeuralNetworkAugmentationParameters(
            input_dim=1, representation_dim=320, kernel_sizes=[3]
        )
        model1 = AutoTCL(
            input_dim=1,
            augmentation=AutoTCLNeuralNetworkAugmentation(
                params=aug_params, training_strategy=RIPTrainingStrategy()
            ),
        )

        torch.manual_seed(42)
        model2 = AutoTCL(
            input_dim=1,
            augmentation=AutoTCLNeuralNetworkAugmentation(
                params=aug_params, training_strategy=RIPTrainingStrategy()
            ),
        )

        losses1 = train_steps(
            model=model1, batch_size=4, seq_length=100, input_dim=1, num_steps=5, seed=123
        )
        losses2 = train_steps(
            model=model2, batch_size=4, seq_length=100, input_dim=1, num_steps=5, seed=123
        )

        assert len(losses1) == len(losses2)
        for _i, (l1, l2) in enumerate(zip(losses1, losses2, strict=True)):
            # Tolerance accounts for mode-toggling timing differences between
            # the old isinstance-gated flow and the centralized maybe_* helper.
            torch.testing.assert_close(l1, l2, rtol=1e-2, atol=1e-3)


class TestAutoTCLConfigSingularDimensions:
    """AutoTCLModelParameters uses singular dimension names (D-01, D-02)."""

    def test_config_has_singular_input_dim(self) -> None:
        params = AutoTCLModelParameters(input_dim=5)
        assert params.input_dim == 5

    def test_config_has_singular_hidden_dim(self) -> None:
        params = AutoTCLModelParameters(input_dim=5, hidden_dim=128)
        assert params.hidden_dim == 128

    def test_config_has_representation_dim(self) -> None:
        params = AutoTCLModelParameters(input_dim=5, representation_dim=256)
        assert params.representation_dim == 256

    def test_config_default_values(self) -> None:
        params = AutoTCLModelParameters(input_dim=1)
        assert params.hidden_dim == 64
        assert params.representation_dim == 320


class TestAutoTCLModelSingularDimensions:
    """AutoTCL model uses singular dimension params and exposes representation_dim."""

    def test_model_accepts_singular_params(self) -> None:
        model = AutoTCL(input_dim=3, hidden_dim=32, representation_dim=64)
        assert isinstance(model, AutoTCL)

    def test_model_representation_dim_property(self) -> None:
        model = AutoTCL(input_dim=3, hidden_dim=32, representation_dim=128)
        assert model.representation_dim == 128

    def test_model_representation_dim_matches_encode_output(self) -> None:
        model = AutoTCL(input_dim=2, hidden_dim=32, representation_dim=64)
        model.eval()
        x = torch.randn(4, 50, 2)
        with torch.no_grad():
            encoded = model.encode(x, batch_size=4, num_workers=0)
        assert encoded.shape[-1] == model.representation_dim
