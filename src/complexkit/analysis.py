"""Complex analysis tools built on contour integration."""

from __future__ import annotations

import math
from math import factorial, pi
from typing import Callable, Sequence

from .integration import Contour, circle, integrate

ComplexFn = Callable[[complex], complex]

TWO_PI_I = 2j * pi


def winding_number(contour: Contour, point: complex, n: int = 400) -> int:
    """Winding number of a closed contour around `point` (rounded to an int)."""
    value = integrate(lambda z: 1 / (z - point), contour, n=n) / TWO_PI_I
    return round(value.real)


def residue(f: ComplexFn, pole: complex, radius: float = 1e-2, n: int = 400) -> complex:
    """Residue of f at an isolated singularity, via a small circle.

    `radius` must be smaller than the distance to any other singularity.
    """
    return integrate(f, circle(pole, radius), n=n) / TWO_PI_I


def cauchy_derivative(
    f: ComplexFn, z0: complex, order: int = 0, radius: float = 0.5, n: int = 400
) -> complex:
    """f^(order)(z0) from Cauchy's integral formula (f analytic inside the circle)."""
    if order < 0:
        raise ValueError("order must be >= 0")
    g = lambda z: f(z) / (z - z0) ** (order + 1)
    return factorial(order) * integrate(g, circle(z0, radius), n=n) / TWO_PI_I


def residue_theorem(
    f: ComplexFn, poles: Sequence[complex], contour: Contour, n: int = 400
) -> complex:
    """Evaluate the closed-contour integral of f as 2*pi*i * sum n(C, p) Res(f, p).

    `poles` lists every singularity of f near/inside the contour.
    """
    poles = list(poles)
    if not poles:
        return 0j
    if len(poles) > 1:
        gap = min(abs(p - q) for i, p in enumerate(poles) for q in poles[i + 1:])
        radius = min(0.1, 0.4 * gap)
    else:
        radius = 0.1
    total = 0j
    for p in poles:
        w = winding_number(contour, p, n=n)
        if w:
            total += w * residue(f, p, radius=radius, n=n)
    return TWO_PI_I * total


def argument_principle(
    f: ComplexFn, contour: Contour, fprime: ComplexFn | None = None, n: int = 2000
) -> int:
    """(#zeros - #poles) of f inside a closed contour, counted with multiplicity."""
    if fprime is None:
        def fprime(z: complex, f=f) -> complex:
            h = 1e-6 * max(1.0, abs(z))
            return (f(z + h) - f(z - h)) / (2 * h)

    value = integrate(lambda z: fprime(z) / f(z), contour, n=n) / TWO_PI_I
    return round(value.real)


def is_close(a: complex, b: complex, tol: float = 1e-8) -> bool:
    """Convenience for comparing integral results."""
    return math.isclose(a.real, b.real, abs_tol=tol) and math.isclose(a.imag, b.imag, abs_tol=tol)
