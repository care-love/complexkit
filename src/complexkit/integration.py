"""Numerical contour integration for complex-valued functions."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from math import pi
from typing import Callable, Iterable, Sequence

ComplexFn = Callable[[complex], complex]
ParametricPath = Callable[[float], complex]

# 5-point Gauss-Legendre rule on [-1, 1].
_GL_NODES = (-0.9061798459386640, -0.5384693101056831, 0.0,
             0.5384693101056831, 0.9061798459386640)
_GL_WEIGHTS = (0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
               0.4786286704993665, 0.2369268850561891)


@dataclass(frozen=True)
class Contour:
    """A parametrized contour z(t), where t runs from start to end.

    Contours can be joined with ``+`` and reversed with ``-`` (or ``.reverse()``).
    """

    z: ParametricPath
    dz: ParametricPath
    start: float = 0.0
    end: float = 1.0
    pieces: tuple["Contour", ...] = ()

    def _parts(self) -> tuple["Contour", ...]:
        return self.pieces if self.pieces else (self,)

    def __add__(self, other: "Contour") -> "Contour":
        if not isinstance(other, Contour):
            return NotImplemented
        return _join(self._parts() + other._parts())

    def reverse(self) -> "Contour":
        """Return the same path traversed in the opposite direction."""
        if self.pieces:
            return _join(tuple(p.reverse() for p in reversed(self.pieces)))
        s, e = self.start, self.end
        return Contour(
            z=lambda t: self.z(s + e - t),
            dz=lambda t: -self.dz(s + e - t),
            start=s,
            end=e,
        )

    __neg__ = reverse


def _join(parts: tuple[Contour, ...]) -> Contour:
    """Glue contours end-to-end into one contour parametrized on [0, 1]."""
    k = len(parts)

    def locate(t: float) -> tuple[Contour, float]:
        if t >= 1:
            index, s = k - 1, 1.0
        else:
            scaled = max(0.0, t) * k
            index = int(scaled)
            s = scaled - index
        p = parts[index]
        return p, p.start + (p.end - p.start) * s

    def z(t: float) -> complex:
        p, u = locate(t)
        return p.z(u)

    def dz(t: float) -> complex:
        p, u = locate(t)
        return p.dz(u) * (p.end - p.start) * k

    return Contour(z=z, dz=dz, pieces=parts)


def integrate(f: ComplexFn, contour: Contour, n: int = 200, method: str = "gauss") -> complex:
    """Approximate the contour integral of f(z) dz.

    Parameters
    ----------
    f:
        Complex-valued function to integrate.
    contour:
        Parametrized path (see :func:`line`, :func:`circle`, :func:`polygon`...).
    n:
        Number of subintervals (per piece for joined contours).
    method:
        ``"gauss"`` (composite 5-point Gauss-Legendre, default, very accurate)
        or ``"trapezoid"``.
    """

    if n < 1:
        raise ValueError("n must be at least 1")
    if method not in ("gauss", "trapezoid"):
        raise ValueError("method must be 'gauss' or 'trapezoid'")

    if contour.pieces:
        per_piece = max(1, -(-n // len(contour.pieces)))  # ceil division
        return sum((integrate(f, p, per_piece, method) for p in contour.pieces), 0j)

    if method == "trapezoid":
        return _trapezoid(f, contour, contour.start, contour.end, n)
    return _gauss(f, contour, contour.start, contour.end, n)


def _trapezoid(f: ComplexFn, contour: Contour, a: float, b: float, n: int) -> complex:
    h = (b - a) / n
    total = 0.5 * _integrand(f, contour, a)
    for i in range(1, n):
        total += _integrand(f, contour, a + i * h)
    total += 0.5 * _integrand(f, contour, b)
    return total * h


def _gauss(f: ComplexFn, contour: Contour, a: float, b: float, n: int) -> complex:
    h = (b - a) / n
    total = 0j
    for i in range(n):
        mid = a + (i + 0.5) * h
        for x, w in zip(_GL_NODES, _GL_WEIGHTS):
            total += w * _integrand(f, contour, mid + 0.5 * h * x)
    return total * h / 2


def line(start: complex, end: complex) -> Contour:
    """Straight-line contour from start to end."""
    delta = end - start
    return Contour(z=lambda t: start + delta * t, dz=lambda _t: delta)


def circle(center: complex = 0j, radius: float = 1.0) -> Contour:
    """Positively (counter-clockwise) oriented full circle."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    return arc(center, radius, 0.0, 2 * pi)


def arc(center: complex, radius: float, theta0: float, theta1: float) -> Contour:
    """Circular arc from angle theta0 to theta1 (radians)."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    span = theta1 - theta0

    def z(t: float) -> complex:
        return center + radius * cmath.exp(1j * (theta0 + span * t))

    def dz(t: float) -> complex:
        return radius * 1j * span * cmath.exp(1j * (theta0 + span * t))

    return Contour(z=z, dz=dz)


def polygon(points: Sequence[complex]) -> Contour:
    """Closed piecewise-linear contour through the given points."""
    if len(points) < 2:
        raise ValueError("polygon needs at least two points")
    vertices = list(points)
    if vertices[0] != vertices[-1]:
        vertices.append(vertices[0])
    return _join(tuple(line(a, b) for a, b in zip(vertices, vertices[1:])))


def rectangle(x0: float, y0: float, x1: float, y1: float) -> Contour:
    """Counter-clockwise rectangle with opposite corners (x0, y0), (x1, y1)."""
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return polygon([complex(xa, ya), complex(xb, ya), complex(xb, yb), complex(xa, yb)])


def integrate_segments(f: ComplexFn, points: Iterable[complex], n_per_segment: int = 200) -> complex:
    """Integrate f over an open polyline, summing straight segment integrals."""
    vertices = list(points)
    if len(vertices) < 2:
        raise ValueError("at least two points are required")
    return sum(
        (integrate(f, line(a, b), n=n_per_segment) for a, b in zip(vertices, vertices[1:])),
        0j,
    )


def _integrand(f: ComplexFn, contour: Contour, t: float) -> complex:
    return f(contour.z(t)) * contour.dz(t)
