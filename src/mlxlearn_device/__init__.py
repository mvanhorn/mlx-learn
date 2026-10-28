"""Device-native, NumPy-free GPU algorithms for mlx-learn.

Unlike the sklearn-compatible ``mlxlearn`` interface, this package accepts and
returns MLX arrays and never imports NumPy or scikit-learn. The binary logistic
model stores its intercept in centered coordinates, retaining numerical
precision for shifted features. See :func:`fit_logistic` for supported inputs.
"""

from ._logistic import DeviceConvergenceWarning, DeviceLogisticModel, fit_logistic

__all__ = ["DeviceConvergenceWarning", "DeviceLogisticModel", "fit_logistic"]
