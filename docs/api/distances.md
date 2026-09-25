# Distance Metrics

Distance functions used by model loss computations.

## SoftDTW

Differentiable dynamic time warping distance (CUDA-accelerated). Used by Series2Vec for temporal distance targets.

```{eval-rst}
.. automodule:: chronocratic.models.utils.distances.soft_dtw.soft_dtw_cuda
   :members:
   :show-inheritance:
```

## Soft-DTW values (no-gradient, O(T) memory)

Memory-lean soft-DTW values for no-gradient use, such as Series2Vec's temporal
supervision targets. Avoids the full dynamic-programming table `SoftDTW` needs
for backpropagation.

```{eval-rst}
.. automodule:: chronocratic.models.utils.distances.soft_dtw.values
   :members:
   :show-inheritance:
```
