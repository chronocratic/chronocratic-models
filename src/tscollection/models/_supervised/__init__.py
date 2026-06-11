"""Unified fine-tuning package for downstream classification and regression.

Provides a single :class:`FineTuningModule` wrapper with injected collaborators
(batch adapters, representation functions, loss) plus :class:`FlattenLinearHead`,
a :class:`BackboneUnfreeze` callback, and factory constructors for TST,
Series2Vec, and TS-TCC backbones.

Usage::

    from tscollection.models._supervised import make_tst_finetuner

    backbone = TST(feat_dim=1, max_seq_len=100, d_model=32, n_heads=4, num_layers=2)
    finetuner = make_tst_finetuner(backbone, num_outputs=5, task='classification')

See the design spec at
``.planning/todos/heads-design_structure.md`` for architecture details.
"""

from __future__ import annotations

from tscollection.models._supervised.adapters import (
    series2vec_representations,
    supervised_batch_adapter,
    tst_batch_adapter,
    tst_representations,
    tstcc_representations,
)
from tscollection.models._supervised.callbacks import BackboneUnfreeze
from tscollection.models._supervised.factory import (
    make_series2vec_finetuner,
    make_tst_finetuner,
    make_tstcc_finetuner,
)
from tscollection.models._supervised.supervised import (
    BatchAdapter,
    FineTuningModule,
    FlattenLinearHead,
    RepresentationBackbone,
)
from tscollection.models._supervised.utils import (
    classification_loss,
    regression_loss,
)

__all__ = [
    'BackboneUnfreeze',
    'BatchAdapter',
    'FineTuningModule',
    'FlattenLinearHead',
    'RepresentationBackbone',
    'classification_loss',
    'make_series2vec_finetuner',
    'make_tst_finetuner',
    'make_tstcc_finetuner',
    'regression_loss',
    'series2vec_representations',
    'supervised_batch_adapter',
    'tst_batch_adapter',
    'tst_representations',
    'tstcc_representations',
]
