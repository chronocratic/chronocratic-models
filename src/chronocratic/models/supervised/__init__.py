"""Unified supervised-training package for downstream classification and regression.

Provides a single :class:`SupervisedModule` wrapper with injected collaborators
(batch adapters, representation functions, loss) plus :class:`FlattenLinearHead`,
a :class:`BackboneUnfreeze` callback, and factory constructors for TST,
Series2Vec, and TS-TCC backbones.

Covers all supervised modes via configuration — linear probe, full fine-tune,
gradual unfreeze, and supervised-from-scratch (a fresh backbone with
``freeze_backbone=False``); see :class:`SupervisedModule` for the mode table.

Usage::

    from chronocratic.models.supervised import make_tst_supervised

    backbone = TST(feat_dim=1, max_seq_len=100, d_model=32, n_heads=4, num_layers=2)
    module = make_tst_supervised(backbone, num_outputs=5, task='classification')

See the design spec at
``.planning/todos/heads-design_structure.md`` for architecture details.
"""

from __future__ import annotations

from chronocratic.models.supervised._adapters import (
    series2vec_representations,
    supervised_batch_adapter,
    tst_batch_adapter,
    tst_representations,
    tstcc_representations,
)
from chronocratic.models.supervised._callbacks import BackboneUnfreeze
from chronocratic.models.supervised._utils import classification_loss, regression_loss
from chronocratic.models.supervised.factory import (
    make_series2vec_supervised,
    make_tst_supervised,
    make_tstcc_supervised,
)
from chronocratic.models.supervised.supervised import (
    BatchAdapter,
    FlattenLinearHead,
    RepresentationBackbone,
    SupervisedModule,
)

__all__ = [
    'BackboneUnfreeze',
    'BatchAdapter',
    'FlattenLinearHead',
    'RepresentationBackbone',
    'SupervisedModule',
    'classification_loss',
    'make_series2vec_supervised',
    'make_tst_supervised',
    'make_tstcc_supervised',
    'regression_loss',
    'series2vec_representations',
    'supervised_batch_adapter',
    'tst_batch_adapter',
    'tst_representations',
    'tstcc_representations',
]
