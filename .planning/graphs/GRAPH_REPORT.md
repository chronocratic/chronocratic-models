# Graph Report - tsmodels  (2026-06-12)

## Corpus Check
- 83 files · ~27,495 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 955 nodes · 1763 edges · 53 communities (48 shown, 5 thin omitted)
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 360 edges (avg confidence: 0.56)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `decc4cd3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 52|Community 52]]

## God Nodes (most connected - your core abstractions)
1. `AugmentationMethod` - 49 edges
2. `MaskMode` - 46 edges
3. `TrainingViews` - 38 edges
4. `DualAugmentation` - 24 edges
5. `TrainableAugmentation` - 23 edges
6. `BaseVariationalAutoencoder` - 22 edges
7. `Tensor` - 21 edges
8. `TST` - 20 edges
9. `AugmentationTrainingStrategy` - 19 edges
10. `PoolingEncodingMixin` - 18 edges

## Surprising Connections (you probably didn't know these)
- `AugmentationMethod` --uses--> `AugmentationMethod`  [INFERRED]
  src/tscollection/models/convolutional/dilated/autotcl/utils.py → src/tscollection/models/augmentation/base.py
- `Parameter` --uses--> `MaskMode`  [INFERRED]
  src/tscollection/models/convolutional/dilated/encoders/encoders.py → src/tscollection/models/convolutional/dilated/encoders/masking.py
- `Optimizer` --uses--> `FCNEncoder`  [INFERRED]
  src/tscollection/models/convolutional/standard/mcl/model.py → src/tscollection/models/convolutional/standard/mcl/encoder.py
- `DualAugmentation` --uses--> `TrainingViews`  [INFERRED]
  src/tscollection/models/augmentation/dual.py → src/tscollection/models/augmentation/base.py
- `AutoTCLNeuralNetworkAugmentation` --uses--> `TrainingViews`  [INFERRED]
  src/tscollection/models/convolutional/dilated/autotcl/augmentation/methods.py → src/tscollection/models/augmentation/base.py

## Import Cycles
- None detected.

## Communities (53 total, 5 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (17): AutoTCLTimeSeriesEncoder, BaseTimeSeriesEncoder, CoSTTimeSeriesEncoder, A class to encode time-series data using a Dilated Convolutional Neural Network, Encode a batch of time series into trend representations.          Args:, Parameters      input_dims: Number of input dimensions.     output_dims: Number, Obtain a sample graph while maintaining the possibility to backprop.          Pa, Encode the input and produce an augmented view via a learned channel mask. (+9 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (45): CoST, Return SGD optimizer over trainable query encoder and projection head parameters, Momentum update for key encoder., CoST Model.      Code source: https://github.com/salesforce/CoST, Augment the batch twice, compute the contrastive loss, perform a manual update s, Compute and log the contrastive validation loss without updating model parameter, compute_amplitude_and_phase(), Compute the amplitude and phase of a complex-valued tensor.      Parameters (+37 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (43): BaseFinetuning, Protocol, LightningModule, Optimizer, Series2Vec, TST, TSTCC, Module (+35 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (21): Tensor, device, Tensor, Tensor, Encode a batch and return logits plus convolutional features.          Args:, Three-block Conv1D encoder backbone for TS-TCC.      Returns ``(logits, features, TCCEncoder, NTXentLoss (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (43): apply_fft(), filter_frequencies(), highpass_filter(), lowpass_filter(), Randomly apply low-pass or high-pass filtering to FFT-transformed samples., Return the complex FFT of one time-series sample., Apply a Butterworth low-pass filter to ``data``., Apply a Butterworth high-pass filter to ``data``. (+35 more)

### Community 5 - "Community 5"
Cohesion: 0.10
Nodes (22): Compute the adversarial augmentation loss.          Args:             x_embeddin, Compute the RIP augmentation loss.          Args:             x_embeddings: Enco, _compute_gaussian_kernel(), info_nce_loss(), l1_out_loss(), local_info_nce_loss(), maximum_mean_discrepancy_with_gaussian_kernel_loss(), Compute L1out loss.      Parameters     ----------     z1 : torch.Tensor (+14 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (17): Barrel for recurrent models (TimeNet)., Sequential, Module, Tensor, Configuration for the TimeNet model.  Provides TimeNetModelParameters with setti, Configuration for the TimeNet model.      Args:         hidden_dims: Number of h, TimeNetModelParameters, GRUWrapper (+9 more)

### Community 7 - "Community 7"
Cohesion: 0.26
Nodes (7): FCN, FCN encoder for Mixup Contrastive Learning (MCL).      This model was implemente, Return projected MCL representations for ``x``., Add a trailing singleton dim so the shape matches the flag-pattern convention., Compute and log the training loss for one batch., Compute and log the validation loss for one batch., Tensor

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (20): Function, range, compute_softdtw(), compute_softdtw_backward(), compute_softdtw_cuda(), prange(), profile(), CUDA implementation is inspired by the diagonal one proposed in https://ieeexplo (+12 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (11): Module, Tensor, _Attention, _FeedForward, _PreNorm, Temporal Contrastive module from TS-TCC (CPC-style).      Computes a temporal co, Return temporal contrastive loss and projection for two views.          Args:, _Residual (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (19): Return the Adam optimizer used to train MCL., OptimizerLRScheduler, Optimizer, Module, ndarray, Optimizer, Tensor, Return the Adam optimizer used to train TimeNet. (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (10): extract_features_from_batch(), Extracts the features (inputs) from a batch.      Parameters     ----------, Tensor, Run one TS2Vec training step with manual optimization., Compute and log the TS2Vec validation loss., TS2Vec Model.      Code source: https://github.com/zhihanyue/ts2vec, Return the primary (non-averaged) encoder for inspection and checkpointing., Return the AdamW optimizer for the TS2Vec encoder. (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (21): AugmentationMethod, Return an augmented view produced by the encoder model.          Args:, Container for augmentation output views and metadata.      The number and shape, Abstract base class for all time-series augmentation strategies.      Subclass t, Return augmented views of ``data``.          Args:             data: Input tenso, TrainingViews, Apply ``first`` and ``second`` to ``data`` and return both views., Any (+13 more)

### Community 13 - "Community 13"
Cohesion: 0.17
Nodes (8): Conv1dDilatedEncoder, A sequence of Conv1dMultiBlock layers with exponentially increasing dilation., Conv1dSamePad, Conv1dSamePadMultiBlock, A 1D convolutional layer that ensures the output has the same length as the inpu, A block consisting of n 1D convolutional layers with user specified activation f, Tensor, Tensor

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (9): ABC, Abstract base classes for augmentation strategies.  This module defines the shar, DualAugmentation, Abstract two-view augmentation contract.  Defines :class:`DualAugmentation` — th, Abstract augmentation that produces two views from a single input.      Subclass, Augmentation producing the first view., Augmentation producing the second view., Augmentation package — abstract types and concrete re-exports.  This package own (+1 more)

### Community 15 - "Community 15"
Cohesion: 0.18
Nodes (11): BasicEncodingMixin, Lightweight shared encoding mixin for fixed-length sequence models.  Provides a, Extract representations for ``data`` in mini-batches.          Iterates ``data``, Uniform ``encode()`` API for fixed-length sequence models.      Designed to be m, Return the callable used to produce per-batch representations.          Typicall, Return the ``nn.Module`` whose train/eval state should be toggled.          Defa, Return the positional args to pass to the encoder.          Default: ``(batch_x,, Return the final representation tensor from the encoder output.          Default (+3 more)

### Community 16 - "Community 16"
Cohesion: 0.22
Nodes (6): AdversarialTrainingStrategy, AutoTCL augmentation training strategies.  Contains ``RIPTrainingStrategy`` and, Initialize the adversarial training strategy.          Args:             trainin, Initialize the RIP training strategy.          Args:             consistency_wei, Adversarial training strategy.      Maximizes the InfoNCE loss between original, AugmentationTrainingStrategy

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (16): generate_all_false_mask(), generate_all_true_mask(), generate_binomial_mask(), generate_continuous_mask(), generate_mask(), generate_mask_last_mask(), get_mask_function(), Generate a mask where all elements are True except for the last element.      Ar (+8 more)

### Community 18 - "Community 18"
Cohesion: 0.07
Nodes (32): Barrel for generative models (TimeVAE)., BandedFourierLayer, LevelModel, Return polynomial trend values for each latent vector., Return additive seasonal values for each latent vector., Return the output shape for Keras-compatible callers., Return the level component for each latent vector., Banded Fourier Layer for applying banded Fourier transform to the input tensor. (+24 more)

### Community 19 - "Community 19"
Cohesion: 0.17
Nodes (14): _compute_contrastive_loss_logits(), instance_contrastive_loss(), Compute contrastive loss logits between two sets of embeddings.      Parameters, Compute instance contrastive loss.      Parameters     ----------     instance_1, ndarray, Tensor, ndarray, Tensor (+6 more)

### Community 20 - "Community 20"
Cohesion: 0.16
Nodes (15): Series2Vec, Tensor, TST, TSTCC, Batch adapters, representation functions, and loss helpers.  Each model has its, Decode TST batch ``(X, targets, padding_masks, IDs)``.      Returns:         ``(, Decode standard ``(X, targets)`` batch.      Used by Series2Vec and TS-TCC downs, Run the TST trunk and zero padded positions.      Args:         backbone: A :cla (+7 more)

### Community 21 - "Community 21"
Cohesion: 0.16
Nodes (11): ActivationFn, PositionalEncoder, FixedPositionalEncoding, _get_activation_fn(), get_pos_encoder(), LearnablePositionalEncoding, Return the positional encoding class for ``pos_encoding``., r"""This transformer encoder layer block is made up of self-attn and feedforward (+3 more)

### Community 22 - "Community 22"
Cohesion: 0.29
Nodes (11): A class to encode time-series data using a Convolutional Sparse Transformer base, TS2VecTimeSeriesEncoder, MaskMode, Masking strategies applied to time-series encoder inputs.      Attributes:, PoolingEncodingMixin, Mixin for pooling-based encoding (TS2Vec, AutoTCL).      Extends :class:`BaseEnc, AdamW, AdamW (+3 more)

### Community 23 - "Community 23"
Cohesion: 0.20
Nodes (7): Tensor, Return transformer representations of shape ``(batch, seq_len, d_model)``., Run the transformer trunk, skipping the reconstruction output layer., Run the full backbone, including the reconstruction output layer.          Used, Compute and log the masked-reconstruction training loss for one batch., Compute and log the masked-reconstruction validation loss for one batch., Expose representation extraction to ``BasicEncodingMixin.encode``.

### Community 24 - "Community 24"
Cohesion: 0.16
Nodes (20): AugmentationTrainingStrategy, Defines how a trainable augmentation network is optimized.      Subclass to crea, AutoTCLNeuralNetworkAugmentation, Instantiate the underlying encoder model., Run the encoder forward pass.          Args:             data: Input time-series, Return an augmented view produced by the encoder model.          Args:, Return the underlying ``AutoTCLAugmentationTimeSeriesEncoder``., Run one aug-network training step.          Forward pass through aug network to (+12 more)

### Community 25 - "Community 25"
Cohesion: 0.13
Nodes (11): BoolTensor, OptimizerLRSchedulerConfig, Tensor, Module, Optimizer, MaskedMSELoss, Compute the loss between a target value and a prediction.          Args:, Clip gradients by global norm to stabilise training. (+3 more)

### Community 26 - "Community 26"
Cohesion: 0.15
Nodes (13): CosTRandomFunctionAugmentation, CosTRandomFunctionAugmentationParameters, CoST augmentation: random jitter/scale/shift.  Contains the ``CosTRandomFunction, Parameters for :class:`CosTRandomFunctionAugmentation`.      Controls the stocha, Stochastic jitter/scale/shift augmentation used by CoST., Initialize the random-function augmentation.          Args:             params:, Add Gaussian noise with std ``sigma`` with probability ``p``., Multiply each channel by a Gaussian factor around 1 with probability ``p``. (+5 more)

### Community 27 - "Community 27"
Cohesion: 0.27
Nodes (9): Any, Tensor, TrainingViews, _normalize_dim(), Return one scaled view of ``data``., Return one view with per-sample time segments permuted., Apply each configured augmentation sequentially., Return one jittered view of ``data``. (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.40
Nodes (4): Jitter, JitterParameters, Parameters for :class:`Jitter`.      Args:         sigma: Std of the additive Ga, Add elementwise Gaussian noise with std ``sigma``.

### Community 29 - "Community 29"
Cohesion: 0.20
Nodes (6): Tensor, r"""Pass the input through the encoder layer.          Args:             src: th, Encode and reconstruct a padded batch.          Args:             x: ``(batch_si, Return transformer representations before output_layer.          Args:, r"""Inputs of forward function         Args:             x: the sequence fed to, r"""Inputs of forward function         Args:             x: the sequence fed to

### Community 30 - "Community 30"
Cohesion: 0.19
Nodes (8): Barrel for transformer models (TST)., Configuration for the TST (Time Series Transformer) model.  Provides TSTModelPar, Configuration for the TST model.      Args:         feat_dim: Number of input fe, TSTModelParameters, Synthesize all-true padding masks; ``encode()`` carries no mask info., Flattened representation size handed to a downstream head.          Returns:, PyTorch Lightning module for the Time Series Transformer (TST).      Representat, TST

### Community 31 - "Community 31"
Cohesion: 0.14
Nodes (16): AugmentationMethod, ComposeAugmentation, Permutation, PermutationParameters, Augmentation primitives and the paired weak/strong augmentation for TS-TCC.  Con, Multiply data by a per-channel Gaussian scale factor., Parameters for :class:`Permutation`.      Args:         max_segments: Upper boun, Split each sample's time axis into segments and permute them. (+8 more)

### Community 33 - "Community 33"
Cohesion: 0.33
Nodes (4): Compute the augmentation network loss.          Args:             x_embeddings:, Run one augmentation-network training step.          Subclasses define their own, Module, Tensor

### Community 34 - "Community 34"
Cohesion: 0.16
Nodes (10): Augmentation with learnable parameters.      Composes an ``AugmentationTrainingS, Initialize a trainable augmentation.          Args:             training_strateg, Return optimizer over this module's parameters.          Args:             lr: L, TrainableAugmentation, AutoTCLNeuralNetworkAugmentationParameters, AutoTCL augmentation methods.  Contains ``AutoTCLNeuralNetworkAugmentation`` and, Parameters for :class:`AutoTCLNeuralNetworkAugmentation`.      Mirrors the const, AdamW (+2 more)

### Community 35 - "Community 35"
Cohesion: 0.21
Nodes (9): Barrel for the convolutional model family., Convenient imports for the main model classes and their configs., Configuration for the Series2Vec model.  Provides Series2VecModelParameters with, Configuration for the Series2Vec model.      Args:         input_dims: Number of, Series2VecModelParameters, Barrel for standard-convolution models (MCL, Series2Vec, TS-TCC)., Configuration for the TS-TCC model.  Provides TSTCCModelParameters with all sett, Configuration for the TS-TCC model.      Args:         input_channels: Number of (+1 more)

### Community 36 - "Community 36"
Cohesion: 0.25
Nodes (4): DisjoinEncoder, Encode input shaped ``(batch, channels, time)``., Initialize convolution weights with Xavier uniform initialization., Tensor

### Community 37 - "Community 37"
Cohesion: 0.23
Nodes (7): AutoTCLModelParameters, Configuration for the AutoTCL model.  Provides AutoTCLModelParameters with AutoT, Configuration for the AutoTCL model.      Args:         input_dims: Number of in, AutoTCL, AutoTCL Model.      Code source: https://github.com/AslanDing/AutoTCL, Return encoder optimizer(s); two optimizers when using TrainableAugmentation., Barrel for dilated-convolution models (AutoTCL, CoST, TS2Vec).

### Community 38 - "Community 38"
Cohesion: 0.20
Nodes (6): FCNEncoder, Encode a batch of time series into flat FCN representations., Three-block dilated Conv1D encoder for MCL.      Produces a flat representation, Expose the FCN encoder (before the MixUp projection head)., Tensor, Module

### Community 39 - "Community 39"
Cohesion: 0.28
Nodes (4): MixUpLoss, Compute soft-label cross-entropy as the mean of per-sample log-softmax dot produ, MixUp contrastive loss used by MCL., Tensor

### Community 40 - "Community 40"
Cohesion: 0.40
Nodes (3): MCLModelParameters, Configuration for the MCL (MixUp Contrastive Learning) model.  Provides MCLModel, Configuration for the MCL model.      Args:         n_in: Number of input featur

### Community 41 - "Community 41"
Cohesion: 0.40
Nodes (3): Run one AutoTCL training step with manual optimization.          Two-phase train, Compute validation contrastive loss using the averaged encoder., Tensor

### Community 42 - "Community 42"
Cohesion: 0.33
Nodes (6): Tensor, classification_loss(), Loss helpers for downstream fine-tuning.  Uses ``nn.functional`` exclusively (no, Cross-entropy loss with safe target flattening.      ``cross_entropy`` expects 1, MSE loss for regression tasks.      Args:         predictions: Model outputs of, regression_loss()

### Community 43 - "Community 43"
Cohesion: 0.50
Nodes (3): CoSTModelParameters, Configuration for the CoST model.  Provides CoSTModelParameters with CoST-specif, Configuration for the CoST model.      Args:         input_dims: Number of input

### Community 44 - "Community 44"
Cohesion: 0.40
Nodes (4): ndarray, Tensor, extract_subsequences_per_row(), Extract subsequences from each row of a 2D tensor     based on provided starting

### Community 46 - "Community 46"
Cohesion: 0.50
Nodes (3): Configuration for the TS2Vec model.  Provides TS2VecModelParameters with all TS2, Configuration for the TS2Vec model.      Args:         input_dims: Number of inp, TS2VecModelParameters

## Knowledge Gaps
- **26 isolated node(s):** `Module`, `AdamW`, `Module`, `Tensor`, `ndarray` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BasicEncodingMixin` connect `Community 15` to `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 14`, `Community 18`, `Community 30`?**
  _High betweenness centrality (0.439) - this node is a cross-community bridge._
- **Why does `TimeNet` connect `Community 6` to `Community 2`, `Community 10`, `Community 15`?**
  _High betweenness centrality (0.194) - this node is a cross-community bridge._
- **Why does `Series2Vec` connect `Community 4` to `Community 35`, `Community 15`?**
  _High betweenness centrality (0.183) - this node is a cross-community bridge._
- **Are the 43 inferred relationships involving `AugmentationMethod` (e.g. with `DualAugmentation` and `AutoTCL`) actually correct?**
  _`AugmentationMethod` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `MaskMode` (e.g. with `AutoTCLNeuralNetworkAugmentation` and `AutoTCLNeuralNetworkAugmentationParameters`) actually correct?**
  _`MaskMode` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 33 inferred relationships involving `TrainingViews` (e.g. with `DualAugmentation` and `AutoTCLNeuralNetworkAugmentation`) actually correct?**
  _`TrainingViews` has 33 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `DualAugmentation` (e.g. with `AugmentationMethod` and `TrainingViews`) actually correct?**
  _`DualAugmentation` has 16 INFERRED edges - model-reasoned connections that need verification._