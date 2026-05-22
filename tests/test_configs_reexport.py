"""Tests for the models/configs/ re-export package (Phase 4, Plan 4).

Verifies that central import paths provide backward-compatible access
to all config classes, augmentation parameter dataclasses, and training
strategies via the configs/ barrel package.
"""

import pytest


class TestConfigsPackageInit:
    """configs/__init__.py is an empty barrel."""

    def test_empty_all_list(self) -> None:
        import tscollection.models.configs as configs_module

        assert configs_module.__all__ == []


class TestConfigsModelsReexport:
    """configs/models.py re-exports all five model config classes."""

    def test_all_list_contains_all_configs(self) -> None:
        from tscollection.models.configs import models as models_config

        expected = [
            'AutoTCLModelParameters',
            'CoSTModelParameters',
            'DilatedCNNModelParameters',
            'ModelParameters',
            'TS2VecModelParameters',
        ]
        assert models_config.__all__ == expected

    def test_ts2vec_model_parameters_importable(self) -> None:
        from tscollection.models.configs.models import TS2VecModelParameters

        p = TS2VecModelParameters(input_dims=1)
        assert p.input_dims == 1

    def test_ts2vec_is_subclass_of_dilated_cnn(self) -> None:
        from tscollection.models.configs.models import (
            DilatedCNNModelParameters,
            TS2VecModelParameters,
        )

        assert issubclass(TS2VecModelParameters, DilatedCNNModelParameters)

    def test_cost_model_parameters_importable(self) -> None:
        from tscollection.models.configs.models import CoSTModelParameters

        p = CoSTModelParameters(input_dims=1, sequence_length=100)
        assert p.input_dims == 1
        assert p.sequence_length == 100

    def test_autotcl_model_parameters_importable(self) -> None:
        from tscollection.models.configs.models import AutoTCLModelParameters

        p = AutoTCLModelParameters(input_dims=1)
        assert p.input_dims == 1

    def test_dilated_cnn_model_parameters_importable(self) -> None:
        from tscollection.models.configs.models import DilatedCNNModelParameters

        p = DilatedCNNModelParameters(input_dims=1)
        assert p.input_dims == 1

    def test_model_parameters_is_abstract(self) -> None:
        from tscollection.models.configs.models import ModelParameters

        with pytest.raises(TypeError, match='abstract'):
            ModelParameters()

    def test_model_parameters_importable(self) -> None:
        from tscollection.models.configs.models import ModelParameters

        assert ModelParameters is not None


class TestConfigsAugmentationMethodsReexport:
    """configs/augmentation/methods.py re-exports aug param dataclasses."""

    def test_all_list(self) -> None:
        from tscollection.models.configs.augmentation import methods as aug_methods

        expected = [
            'AutoTCLNeuralNetworkAugmentationParameters',
            'CosTRandomFunctionAugmentationParameters',
            'CropShiftAugmentationParameters',
        ]
        assert aug_methods.__all__ == expected

    def test_crop_shift_params_importable(self) -> None:
        from tscollection.models.configs.augmentation.methods import (
            CropShiftAugmentationParameters,
        )

        p = CropShiftAugmentationParameters()
        assert p.temporal_unit == 0

    def test_cost_aug_params_importable(self) -> None:
        from tscollection.models.configs.augmentation.methods import (
            CosTRandomFunctionAugmentationParameters,
        )

        p = CosTRandomFunctionAugmentationParameters(sigma=0.1)
        assert p.sigma == 0.1
        assert p.p == 0.5

    def test_autotcl_aug_params_importable(self) -> None:
        from tscollection.models.configs.augmentation.methods import (
            AutoTCLNeuralNetworkAugmentationParameters,
        )

        p = AutoTCLNeuralNetworkAugmentationParameters(input_dims=1, output_dims=32)
        assert p.input_dims == 1
        assert p.output_dims == 32


class TestConfigsAugmentationTrainingReexport:
    """configs/augmentation/training.py re-exports training strategies + ABC."""

    def test_all_list(self) -> None:
        from tscollection.models.configs.augmentation import training as aug_training

        expected = [
            'AdversarialTrainingStrategy',
            'AugmentationTrainingStrategy',
            'RIPTrainingStrategy',
        ]
        assert aug_training.__all__ == expected

    def test_rip_strategy_importable(self) -> None:
        from tscollection.models.configs.augmentation.training import RIPTrainingStrategy

        s = RIPTrainingStrategy()
        assert s.should_train(0, 0) is True

    def test_rip_strategy_training_ratio_step(self) -> None:
        from tscollection.models.configs.augmentation.training import RIPTrainingStrategy

        s = RIPTrainingStrategy(training_ratio_step=3)
        assert s.should_train(0, 0) is True
        assert s.should_train(1, 0) is False
        assert s.should_train(3, 0) is True

    def test_adversarial_strategy_importable(self) -> None:
        from tscollection.models.configs.augmentation.training import (
            AdversarialTrainingStrategy,
        )

        s = AdversarialTrainingStrategy()
        assert s.should_train(0, 0) is True

    def test_augmentation_training_strategy_importable(self) -> None:
        from tscollection.models.configs.augmentation.training import (
            AugmentationTrainingStrategy,
        )

        # It's an ABC — should not be instantiable directly
        assert AugmentationTrainingStrategy is not None


class TestAugmentationPackageInit:
    """configs/augmentation/__init__.py is an empty barrel."""

    def test_empty_all_list(self) -> None:
        import tscollection.models.configs.augmentation as aug_module

        assert aug_module.__all__ == []
