__all__ = ['encode_data_mcl', 'extract_representations_mcl', 'train_mcl_model']

from functools import partial
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from experiments.utils import get_early_stopping_params
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from src.autotsrc.models.representations import (
    construct_and_make_representations_folders,
    do_extract_representations,
)
import torch

from src.autotsrc.enums import TimeSeriesDatasetMode

from .model import get_mcl_model

if TYPE_CHECKING:
    from lightning.pytorch.callbacks.callback import Callback


def train_mcl_model(
    model: Any,
    train_data: Any,
    max_epochs: int,
    outputs_folder_path: Path | None = None,
    best_models_folder_path: Path | None = None,
    final_models_folder_path: Path | None = None,
    device: str = 'cpu',
) -> Any | tuple[Any, ModelCheckpoint, Path]:
    """Train an MCL model and optionally return checkpoint metadata."""
    dataset_name = train_data.name
    model_name = 'mcl'

    early_stopping_params = get_early_stopping_params(max_epochs=max_epochs)

    early_stopping_callback = pl.callbacks.EarlyStopping(**early_stopping_params)

    callbacks: list[Callback] = [early_stopping_callback]

    if best_models_folder_path:
        model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss_epoch',
            dirpath=best_models_folder_path,
            filename=f'{dataset_name}_{model_name}_best',
            save_top_k=1,
            mode='min',
            save_weights_only=True,
            verbose=True,
        )
        callbacks.append(model_checkpoint_callback)

    if outputs_folder_path:
        lightning_logs_folder_path = outputs_folder_path / 'lightning_logs'
        lightning_logs_folder_path.mkdir(parents=True, exist_ok=True)
        logger = CSVLogger(
            save_dir=lightning_logs_folder_path, name=f'{dataset_name}_{model_name}', version=None
        )
        lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor_callback)

    else:
        logger = False

    trainer = Trainer(
        devices='auto',
        accelerator=device,
        callbacks=callbacks,
        max_epochs=max_epochs,
        logger=logger,
        enable_checkpointing=False,
        deterministic=True,
        log_every_n_steps=5,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_data.train_dataloader(strict_batch_size=True),
        val_dataloaders=train_data.val_dataloader(strict_batch_size=True),
    )

    # save the final model
    if final_models_folder_path:
        final_model_path = final_models_folder_path / f'{dataset_name}_{model_name}_final.ckpt'
        trainer.save_checkpoint(filepath=final_model_path, weights_only=True)

    if final_models_folder_path and best_models_folder_path:
        return model, model_checkpoint_callback, final_model_path
    return model


def encode_data_mcl(data: Any, model: Any) -> torch.Tensor:
    """Encode data with an MCL model in representation mode."""
    model.switch_to_representation_mode()
    model.eval()
    with torch.inference_mode():
        encoded_data = model(data)
    return encoded_data


def extract_representations_mcl(
    dataset_module: Any,
    model_params: dict,
    max_epochs: int,
    outputs_folder_path: Path,
    device: str = 'cpu',
) -> None:
    """Train MCL and extract representations from the best and final checkpoints."""
    model_name = 'mcl'
    encoder_name = 'mcl'

    # train the model
    model = get_mcl_model(model_params=model_params)
    model.switch_to_training_mode()

    dataset_name = dataset_module.name

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

    model, model_checkpoint_callback, final_model_path = train_mcl_model(
        model=model,
        train_data=dataset_module,
        max_epochs=max_epochs,
        outputs_folder_path=outputs_folder_path,
        best_models_folder_path=best_models_folder_path,
        final_models_folder_path=final_models_folder_path,
        device=device,
    )

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
