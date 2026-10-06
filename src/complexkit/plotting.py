"""Plotting helpers (require numpy + matplotlib: pip install complexkit[plot])."""

from __future__ import annotations

from typing import Callable

ComplexMap = Callable[[complex], complex]


def domain_coloring(
    f: ComplexMap,
    xlim: tuple[float, float] = (-2, 2),
    ylim: tuple[float, float] = (-2, 2),
    resolution: int = 400,
    *,
    show: bool = True,
    title: str | None = None,
):
    """Domain-colouring plot of f: hue = arg f(z), brightness bands = log|f(z)|.

    Zeros and poles show up as points where all colours meet (opposite
    colour orientation distinguishes zeros from poles).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import hsv_to_rgb

    xs = np.linspace(*xlim, resolution)
    ys = np.linspace(*ylim, resolution)
    Z = xs[None, :] + 1j * ys[:, None]

    with np.errstate(all="ignore"):
        try:
            W = np.asarray(f(Z), dtype=complex)
            if W.shape != Z.shape:
                raise TypeError
        except TypeError:
            def safe(z):
                try:
                    return f(z)
                except (ZeroDivisionError, OverflowError, ValueError):
                    return complex("nan")

            W = np.vectorize(safe, otypes=[complex])(Z)

        hue = (np.angle(W) / (2 * np.pi)) % 1.0
        mag = np.log2(np.abs(W))
        value = 0.55 + 0.45 * (mag - np.floor(mag))
        hsv = np.stack([hue, np.full_like(hue, 0.85), value], axis=-1)
        hsv = np.nan_to_num(hsv, nan=0.0, posinf=1.0, neginf=0.0)
        rgb = hsv_to_rgb(np.clip(hsv, 0, 1))

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(rgb, extent=(*xlim, *ylim), origin="lower")
    axis.set_xlabel("Re")
    axis.set_ylabel("Im")
    axis.set_title(title or "domain colouring of f(z)")
    if show:
        plt.show()
    return figure, axis
