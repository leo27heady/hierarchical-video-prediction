"""Versioned synthetic datasets with scientific metadata and manifests."""

from .moving_mnist import MovingMNISTBundle, generate_moving_mnist, save_moving_mnist

__all__ = ["MovingMNISTBundle", "generate_moving_mnist", "save_moving_mnist"]
