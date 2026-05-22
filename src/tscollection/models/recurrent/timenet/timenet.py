__all__ = ['extract_representations_timenet']

from functools import partial
import logging
from pathlib import Path
from typing import Protocol

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from experiments.utils import get_early_stopping_params
from tscollection.datasets.enums import TimeSeriesDatasetMode
from src.autotsrc.models.representations import (
    construct_and_make_representations_folders,
    do_extract_representations,
)

from tscollection.models.recurrent.timenet.model import get_timenet_model


class _TimeNetDatasetModule(Protocol):
    name: str

    def train_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        shuffle: bool | None = None,
        strict_batch_size: bool = True,
        extra_args: dict | None = None,
    ) -> DataLoader:
        """Build the dataloader used for training."""

    def val_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        strict_batch_size: bool = True,
        extra_args: dict | None = None,
    ) -> DataLoader | None:
        """Build the dataloader used for validation."""

    def test_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader:
        """Build the dataloader used for testing."""


def extract_representations_timenet(
    dataset_module: _TimeNetDatasetModule,
    model_params: dict,
    max_epochs: int,
    outputs_folder_path: Path,
) -> None:
    """Train TimeNet and extract representations from the best and final checkpoints."""
    model_name = 'timenet'
    encoder_name = 'timenet'

    # train the model
    model = get_timenet_model(model_params=model_params)
    model.switch_to_training_mode()

    dataset_name = dataset_module.name

    early_stopping_params = get_early_stopping_params(max_epochs=max_epochs)

    representations_folders_dict = construct_and_make_representations_folders(
        outputs_folder_path=outputs_folder_path
    )
    best_models_folder_path = representations_folders_dict['best_models_folder_path']
    final_models_folder_path = representations_folders_dict['final_models_folder_path']
    best_model_representations_folder_path = representations_folders_dict[
        'best_model_representations_folder_path'
    ]
    final_model_representations_folder_path = representations_folders_dict[
        'final_model_representations_folder_path'
    ]

    early_stopping_callback = pl.callbacks.EarlyStopping(**early_stopping_params)

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss_epoch',
        dirpath=best_models_folder_path,
        filename=f'{dataset_name}_{model_name}_best',
        save_top_k=1,
        mode='min',
        save_weights_only=True,
        verbose=True,
    )

    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    lightning_logs_folder_path = outputs_folder_path / 'lightning_logs'
    lightning_logs_folder_path.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(
        save_dir=lightning_logs_folder_path, name=f'{dataset_name}_{model_name}', version=None
    )

    trainer = Trainer(
        devices='auto',
        accelerator='auto',
        callbacks=[early_stopping_callback, model_checkpoint_callback, lr_monitor_callback],
        max_epochs=max_epochs,
        logger=logger,
        deterministic=True,
        log_every_n_steps=5,
    )

    trainer.fit(
        model=model,
        train_dataloaders=dataset_module.train_dataloader(),
        val_dataloaders=dataset_module.val_dataloader(),
    )

    final_model_path = final_models_folder_path / f'{dataset_name}_{model_name}_final.ckpt'
    trainer.save_checkpoint(filepath=final_model_path, weights_only=True)
    model_class = type(model)
    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path is not None and Path(best_model_path).exists():
        best_model = model_class.load_from_checkpoint(checkpoint_path=best_model_path)
    else:
        logging.error(f'No best model found for {model_name} on {dataset_name}')
        return

    final_model = model_class.load_from_checkpoint(checkpoint_path=final_model_path)

    data_loader_map = {
        'train': partial(
            dataset_module.train_dataloader, mode=TimeSeriesDatasetMode.WITH_LABELS, shuffle=False
        ),
        'valid': partial(dataset_module.val_dataloader, mode=TimeSeriesDatasetMode.WITH_LABELS),
        'test': partial(dataset_module.test_dataloader, mode=TimeSeriesDatasetMode.WITH_LABELS),
    }

    best_model.switch_to_representation_mode()
    do_extract_representations(
        encoder=best_model,
        model_name=model_name,
        encoder_name=encoder_name,
        dataset_name=dataset_name,
        data_loader_map=data_loader_map,
        output_folder_path=best_model_representations_folder_path,
    )

    final_model.switch_to_representation_mode()
    do_extract_representations(
        encoder=final_model,
        model_name=model_name,
        encoder_name=encoder_name,
        dataset_name=dataset_name,
        data_loader_map=data_loader_map,
        output_folder_path=final_model_representations_folder_path,
    )

    logging.info(f'Finished extracting representations for {model_name} on {dataset_name}')
    logging.info(f'Finished running {model_name} on {dataset_name}')
