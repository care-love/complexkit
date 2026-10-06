"""Complex mapping utilities: standard maps, grids, local geometry, plotting."""

from __future__ import annotations

import cmath
import math
from cmath import exp
from typing import Callable, Iterable

ComplexMap = Callable[[complex], complex]
Grid = list[list[complex]]

NAN = complex(math.nan, math.nan)


# ----------------------------------------------------------------- helpers
def safe(mapping: ComplexMap) -> ComplexMap:
    """Wrap a mapping so poles / branch points give nan instead of raising."""

    def wrapped(z: complex) -> complex:
        try:
            return mapping(z)
        except (ZeroDivisionError, OverflowError, ValueError):
            return NAN

    return wrapped


def apply_map(mapping: ComplexMap, points: Iterable[complex], *, on_error: str = "nan") -> list[complex]:
    """Apply a complex mapping to points. ``on_error`` is 'nan' or 'raise'."""
    fn = safe(mapping) if on_error == "nan" else mapping
    return [fn(z) for z in points]


def compose(*mappings: ComplexMap) -> ComplexMap:
    """Compose left to right: compose(f, g)(z) == g(f(z))."""
    if not mappings:
        return lambda z: z

    def composed(z: complex) -> complex:
        value = z
        for mapping in mappings:
            value = mapping(value)
        return value

    return composed


# ---------------------------------------------------------- standard maps
def mobius(a: complex, b: complex, c: complex, d: complex) -> ComplexMap:
    """f(z) = (az + b) / (cz + d).  (See also complexkit.Mobius for the full class.)"""
    if a * d - b * c == 0:
        raise ValueError("Mobius transform requires ad - bc != 0")

    def transform(z: complex) -> complex:
        denominator = c * z + d
        if denominator == 0:
            raise ZeroDivisionError("mapping has a pole at this point")
        return (a * z + b) / denominator

    return transform


def power(exponent: complex) -> ComplexMap:
    """f(z) = z ** exponent (principal branch)."""
    return lambda z: z**exponent


def inverse(z: complex) -> complex:
    """f(z) = 1 / z."""
    return 1 / z


def exp_map(z: complex) -> complex:
    """f(z) = exp(z)."""
    return exp(z)


def log_map(z: complex) -> complex:
    """f(z) = Log z (principal branch, cut along the negative real axis)."""
    return cmath.log(z)


def sqrt_map(z: complex) -> complex:
    """f(z) = sqrt(z) (principal branch)."""
    return cmath.sqrt(z)


def sin_map(z: complex) -> complex:
    return cmath.sin(z)


def cos_map(z: complex) -> complex:
    return cmath.cos(z)


def joukowski(c: float = 1.0) -> ComplexMap:
    """Joukowski map f(z) = z + c^2 / z  (airfoil / flat-plate mappings)."""
    return lambda z: z + (c * c) / z


def cayley(z: complex) -> complex:
    """Cayley transform (z - i)/(z + i): upper half-plane -> unit disk."""
    return (z - 1j) / (z + 1j)


# ----------------------------------------------------- local geometry
def derivative(f: ComplexMap, z: complex, h: float = 1e-6) -> complex:
    """Central-difference estimate of f'(z) for an analytic f."""
    step = h * max(1.0, abs(z))
    return (f(z + step) - f(z - step)) / (2 * step)


def conformal_factor(f: ComplexMap, z: complex) -> tuple[float, float]:
    """Return (scale, rotation) of f at z: |f'(z)| and arg f'(z) in radians.

    Near z, f stretches lengths by `scale` and rotates directions by `rotation`.
    If scale ~ 0, z is a critical point and the map is not conformal there.
    """
    d = derivative(f, z)
    return abs(d), cmath.phase(d)


def is_conformal_at(f: ComplexMap, z: complex, tol: float = 1e-8) -> bool:
    return conformal_factor(f, z)[0] > tol


# ------------------------------------------------------------ sampling
def sample_line(start: complex, end: complex, samples: int = 200) -> list[complex]:
    """Points along a straight segment."""
    return [start + (end - start) * i / (samples - 1) for i in range(samples)]


def sample_circle(center: complex = 0j, radius: float = 1.0, samples: int = 400) -> list[complex]:
    """Points around a full circle (closed)."""
    return [center + radius * exp(2j * math.pi * i / (samples - 1)) for i in range(samples)]


def rectangular_grid(
    xmin: float, xmax: float, ymin: float, ymax: float, lines: int = 11, samples: int = 200
) -> Grid:
    """Horizontal and vertical grid lines in the complex plane."""
    if lines < 2:
        raise ValueError("lines must be at least 2")
    if samples < 2:
        raise ValueError("samples must be at least 2")

    xs = _linspace(xmin, xmax, samples)
    ys = _linspace(ymin, ymax, samples)
    grid_values = _linspace(0, 1, lines)

    horizontal = [[complex(x, ymin + (ymax - ymin) * u) for x in xs] for u in grid_values]
    vertical = [[complex(xmin + (xmax - xmin) * u, y) for y in ys] for u in grid_values]
    return horizontal + vertical


def polar_grid(
    rmax: float = 1.0, rings: int = 6, rays: int = 12,
    theta_min: float = 0.0, theta_max: float = 2 * math.pi, samples: int = 200,
) -> Grid:
    """Concentric arcs and radial rays (ideal for z**n, sqrt, log, exp...)."""
    if rings < 1 or rays < 1:
        raise ValueError("rings and rays must be at least 1")
    thetas = _linspace(theta_min, theta_max, samples)
    radii = _linspace(0, rmax, samples)
    ring_lines = [[r * exp(1j * th) for th in thetas] for r in _linspace(rmax / rings, rmax, rings)]
    full = abs(theta_max - theta_min) >= 2 * math.pi - 1e-12
    # a full circle needs `rays` distinct angles; a sector needs `rays + 1` (both edges)
    count = rays if full else rays + 1
    ray_angles = [theta_min + (theta_max - theta_min) * i / rays for i in range(count)]
    ray_lines = [[r * exp(1j * th) for r in radii] for th in ray_angles]
    return ring_lines + ray_lines


def map_grid(mapping: ComplexMap, grid: Grid, *, on_error: str = "nan") -> Grid:
    """Apply a mapping to every point of every line in a grid."""
    return [apply_map(mapping, line, on_error=on_error) for line in grid]


# ------------------------------------------------------------- plotting
def plot_mapping(mapping: ComplexMap, grid: Grid, *, show: bool = True, window: float | None = None):
    """Plot original and mapped grid lines (matplotlib is imported lazily).

    ``window`` sets the half-width of the mapped plot (useful when the image
    contains points near a pole that fly off to infinity).
    """
    import matplotlib.pyplot as plt

    mapped = map_grid(mapping, grid)
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    _plot_grid(axes[0], grid, "original: z")
    _plot_grid(axes[1], mapped, "mapped: w = f(z)")
    if window is not None:
        axes[1].set_xlim(-window, window)
        axes[1].set_ylim(-window, window)
        axes[1].set_aspect("equal", adjustable="box")
    if show:
        plt.show()
    return figure, axes


def _plot_grid(axis, grid: Grid, title: str) -> None:
    axis.axhline(0, color="0.8", linewidth=0.8)
    axis.axvline(0, color="0.8", linewidth=0.8)
    for line in grid:
        axis.plot([z.real for z in line], [z.imag for z in line], color="#2563eb", linewidth=0.9)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(title)
    axis.set_xlabel("Re")
    axis.set_ylabel("Im")


def _linspace(start: float, end: float, count: int) -> list[float]:
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]
