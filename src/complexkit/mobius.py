"""Mobius (linear fractional) transformations on the extended complex plane."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

INF = complex(math.inf, 0.0)  # the point at infinity


def is_infinite(z: complex) -> bool:
    return math.isinf(z.real) or math.isinf(z.imag)


@dataclass(frozen=True)
class GeneralizedCircle:
    """A circle or a straight line (a 'circle' through infinity)."""

    kind: str  # "circle" or "line"
    center: complex = 0j  # circle only
    radius: float = 0.0  # circle only
    point: complex = 0j  # line only: a point on the line
    direction: complex = 1 + 0j  # line only: unit direction

    def __str__(self) -> str:
        if self.kind == "circle":
            return f"circle(center={self.center:.6g}, radius={self.radius:.6g})"
        return f"line(through={self.point:.6g}, direction={self.direction:.6g})"


@dataclass(frozen=True)
class Mobius:
    """f(z) = (a z + b) / (c z + d),  with  ad - bc != 0.

    Supports calling, composition (``f @ g`` means f(g(z))), ``inverse()``,
    fixed points, classification, building from three point pairs, and
    mapping circles/lines.
    """

    a: complex
    b: complex
    c: complex
    d: complex

    def __post_init__(self) -> None:
        if self.det == 0:
            raise ValueError("Mobius transform requires ad - bc != 0")

    # ---- basics
    @property
    def det(self) -> complex:
        return self.a * self.d - self.b * self.c

    def __call__(self, z: complex) -> complex:
        if is_infinite(z):
            return INF if self.c == 0 else self.a / self.c
        denominator = self.c * z + self.d
        if denominator == 0:
            return INF
        return (self.a * z + self.b) / denominator

    def __matmul__(self, other: "Mobius") -> "Mobius":
        """Composition: (f @ g)(z) == f(g(z)) -- matrix multiplication."""
        if not isinstance(other, Mobius):
            return NotImplemented
        return Mobius(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )

    def inverse(self) -> "Mobius":
        return Mobius(self.d, -self.b, -self.c, self.a)

    def normalized(self) -> "Mobius":
        """Scale coefficients so that ad - bc = 1."""
        s = cmath.sqrt(self.det)
        return Mobius(self.a / s, self.b / s, self.c / s, self.d / s)

    @property
    def pole(self) -> complex:
        """The point sent to infinity (INF if c == 0)."""
        return INF if self.c == 0 else -self.d / self.c

    @property
    def image_of_infinity(self) -> complex:
        return INF if self.c == 0 else self.a / self.c

    # ---- structure
    def fixed_points(self) -> list[complex]:
        """Fixed points in the extended plane (INF is included when c == 0)."""
        if self.c == 0:
            if self.a == self.d:  # translation (or identity): only infinity
                return [INF]
            return [self.b / (self.d - self.a), INF]
        disc = cmath.sqrt((self.d - self.a) ** 2 + 4 * self.b * self.c)
        z1 = (self.a - self.d + disc) / (2 * self.c)
        z2 = (self.a - self.d - disc) / (2 * self.c)
        if abs(z1 - z2) < 1e-12 * max(1.0, abs(z1)):
            return [z1]
        return [z1, z2]

    def classify(self, tol: float = 1e-9) -> str:
        """'identity', 'parabolic', 'elliptic', 'hyperbolic' or 'loxodromic'."""
        n = self.normalized()
        if abs(n.c) < tol and abs(n.b) < tol and abs(n.a - n.d) < tol:
            return "identity"
        t2 = (n.a + n.d) ** 2  # trace^2 with det = 1
        if abs(t2 - 4) < tol:
            return "parabolic"
        if abs(t2.imag) < tol:
            if 0 <= t2.real < 4:
                return "elliptic"
            if t2.real > 4:
                return "hyperbolic"
        return "loxodromic"

    # ---- geometry
    def map_circle(self, center: complex, radius: float) -> GeneralizedCircle:
        """Image of the circle |z - center| = radius (a circle or a line)."""
        pts = [center + radius * cmath.exp(2j * math.pi * k / 3) for k in range(3)]
        return circle_through(*[self(p) for p in pts])

    def map_line(self, point: complex, direction: complex) -> GeneralizedCircle:
        """Image of the line through `point` with the given direction."""
        d = direction / abs(direction)
        pts = [point, point + d, point - d]
        return circle_through(*[self(p) for p in pts])

    # ---- constructors
    @classmethod
    def from_points(cls, z: tuple, w: tuple) -> "Mobius":
        """The unique Mobius map with z[i] -> w[i] for three distinct pairs.

        Each of the six points may be INF.
        """
        if len(z) != 3 or len(w) != 3:
            raise ValueError("exactly three points required on each side")
        return _to_standard(w).inverse() @ _to_standard(z)

    @classmethod
    def identity(cls) -> "Mobius":
        return cls(1, 0, 0, 1)

    @classmethod
    def cayley(cls) -> "Mobius":
        """Upper half-plane -> unit disk: (z - i)/(z + i)."""
        return cls(1, -1j, 1, 1j)

    @classmethod
    def disk_automorphism(cls, a: complex, theta: float = 0.0) -> "Mobius":
        """e^{i theta} (z - a)/(1 - conj(a) z): unit disk -> unit disk, a -> 0."""
        if abs(a) >= 1:
            raise ValueError("|a| must be < 1")
        e = cmath.exp(1j * theta)
        return cls(e, -e * a, -complex(a).conjugate(), 1)


def _to_standard(pts: tuple) -> Mobius:
    """Mobius map sending the three points to 0, 1, infinity."""
    z1, z2, z3 = pts
    if len({z1, z2, z3}) < 3:
        raise ValueError("points must be distinct")
    if is_infinite(z1):
        return Mobius(0, z2 - z3, 1, -z3)
    if is_infinite(z2):
        return Mobius(1, -z1, 1, -z3)
    if is_infinite(z3):
        return Mobius(1, -z1, 0, z2 - z1)
    return Mobius(z2 - z3, -z1 * (z2 - z3), z2 - z1, -z3 * (z2 - z1))


def cross_ratio(z1: complex, z2: complex, z3: complex, z4: complex) -> complex:
    """(z1, z2; z3, z4) = (z1-z3)(z2-z4) / ((z2-z3)(z1-z4)). Invariant under Mobius maps."""
    return (z1 - z3) * (z2 - z4) / ((z2 - z3) * (z1 - z4))


def circle_through(z1: complex, z2: complex, z3: complex, tol: float = 1e-9) -> GeneralizedCircle:
    """Circle (or line) through three points; INF among them forces a line."""
    pts = [z1, z2, z3]
    finite = [p for p in pts if not is_infinite(p)]
    if len(finite) < 3:
        p, q = finite[0], finite[1]
        return GeneralizedCircle("line", point=p, direction=(q - p) / abs(q - p))
    # collinear?  area of the triangle ~ 0
    cross = ((z2 - z1).conjugate() * (z3 - z1)).imag
    scale = max(abs(z2 - z1), abs(z3 - z1), 1.0) ** 2
    if abs(cross) < tol * scale:
        far = max(finite, key=lambda p: abs(p - z1))
        d = (far - z1) / abs(far - z1)
        return GeneralizedCircle("line", point=z1, direction=d)
    # circumcenter
    ax, ay, bx, by, cx, cy = z1.real, z1.imag, z2.real, z2.imag, z3.real, z3.imag
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    center = complex(ux, uy)
    return GeneralizedCircle("circle", center=center, radius=abs(z1 - center))
