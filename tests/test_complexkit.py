import cmath
from math import pi

import pytest

import complexkit as ck
from complexkit import INF, Mobius

TWO_PI_I = 2j * pi


def close(a, b, tol=1e-8):
    return abs(a - b) < tol


# ---------------------------------------------------------------- integration
def test_cauchy_circle():
    assert close(ck.integrate(lambda z: 1 / z, ck.circle()), TWO_PI_I)


def test_entire_function_closed_is_zero():
    assert close(ck.integrate(lambda z: z**2 + cmath.exp(z), ck.circle(1 + 1j, 2)), 0)


def test_polygon_square_around_origin():
    sq = ck.rectangle(-1, -1, 1, 1)
    assert close(ck.integrate(lambda z: 1 / z, sq), TWO_PI_I)


def test_line_integral_matches_antiderivative():
    # int_0^{1+i} z^2 dz = (1+i)^3 / 3
    assert close(ck.integrate(lambda z: z**2, ck.line(0, 1 + 1j)), (1 + 1j) ** 3 / 3)


def test_arc_half_circle():
    # int over upper half circle of 1/z = i*pi
    assert close(ck.integrate(lambda z: 1 / z, ck.arc(0, 1, 0, pi)), 1j * pi)


def test_contour_add_and_reverse():
    c = ck.line(0, 1) + ck.line(1, 1j) + ck.line(1j, 0)
    assert close(ck.integrate(lambda z: z, c), 0)
    # reversing flips the sign
    f = lambda z: 1 / z
    c = ck.circle()
    assert close(ck.integrate(f, -c), -TWO_PI_I)


def test_trapezoid_option():
    assert close(ck.integrate(lambda z: 1 / z, ck.circle(), n=400, method="trapezoid"), TWO_PI_I)


def test_bad_arguments():
    with pytest.raises(ValueError):
        ck.integrate(lambda z: z, ck.circle(), n=0)
    with pytest.raises(ValueError):
        ck.circle(radius=-1)


# ------------------------------------------------------------------- analysis
def test_residue_simple_pole():
    f = lambda z: 1 / (z**2 + 1)
    assert close(ck.residue(f, 1j), -0.5j)


def test_residue_theorem_matches_direct_integral():
    f = lambda z: 1 / (z**2 + 1)
    c = ck.circle(0, 2)
    direct = ck.integrate(f, c)
    assert close(ck.residue_theorem(f, [1j, -1j], c), direct, 1e-6)
    assert close(direct, 0, 1e-6)  # residues cancel


def test_residue_theorem_only_enclosed_poles():
    f = lambda z: 1 / (z**2 + 1)
    c = ck.circle(1j, 0.5)  # encloses only +i
    assert close(ck.residue_theorem(f, [1j, -1j], c), TWO_PI_I * (-0.5j), 1e-6)  # = pi


def test_winding_number():
    assert ck.winding_number(ck.circle(), 0) == 1
    assert ck.winding_number(-ck.circle(), 0) == -1
    assert ck.winding_number(ck.circle(), 5) == 0
    assert ck.winding_number(ck.circle() + ck.circle(), 0) == 2


def test_cauchy_derivative():
    assert close(ck.cauchy_derivative(cmath.exp, 0, order=3), 1, 1e-8)
    assert close(ck.cauchy_derivative(lambda z: z**5, 1, order=2), 20, 1e-6)


def test_argument_principle_counts_zeros():
    assert ck.argument_principle(lambda z: z**3 - 1, ck.circle(0, 2)) == 3
    assert ck.argument_principle(lambda z: z**3 - 1, ck.circle(0, 0.5)) == 0
    # pole of order 1 and 2 zeros: Z - P = 1
    assert ck.argument_principle(lambda z: (z - 1) * (z + 1) / z, ck.circle(0, 2)) == 1


# --------------------------------------------------------------------- mobius
def test_mobius_call_and_poles():
    f = Mobius(1, 0, 0, 1)  # identity
    assert f(3 + 1j) == 3 + 1j
    g = Mobius(0, 1, 1, 0)  # 1/z
    assert g(0) == INF
    assert g(INF) == 0


def test_compose_and_inverse():
    f = Mobius(1, 2, 3, 5)
    g = Mobius(2, 1, 1, 1)
    z = 0.3 + 0.7j
    assert close((f @ g)(z), f(g(z)))
    assert close(f.inverse()(f(z)), z)


def test_from_points_hits_targets():
    z = (0, 1, 2)
    w = (1j, 5, -3 + 1j)
    T = Mobius.from_points(z, w)
    for zi, wi in zip(z, w):
        assert close(T(zi), wi)


def test_from_points_with_infinity():
    T = Mobius.from_points((0, 1, INF), (1, INF, 0))
    assert close(T(0), 1) and T(1) == INF and close(T(INF), 0)


def test_cross_ratio_invariant():
    f = Mobius(2, 1j, 1, 3)
    pts = [0.5, 1 + 1j, -2, 3j]
    assert close(ck.cross_ratio(*pts), ck.cross_ratio(*[f(p) for p in pts]))


def test_fixed_points():
    f = Mobius(2, 1, 1, 3)
    for p in f.fixed_points():
        assert close(f(p), p)
    assert Mobius(1, 5, 0, 1).fixed_points() == [INF]  # translation
    assert INF in Mobius(2, 0, 0, 1).fixed_points()  # dilation: 0 and INF


def test_classify():
    assert Mobius(1, 1, 0, 1).classify() == "parabolic"
    assert Mobius(2, 0, 0, 1).classify() == "hyperbolic"
    assert Mobius(cmath.exp(0.5j), 0, 0, cmath.exp(-0.5j)).classify() == "elliptic"
    assert Mobius(1 + 1j, 0, 0, 1).classify() == "loxodromic"
    assert Mobius.identity().classify() == "identity"


def test_cayley_maps_real_axis_to_unit_circle_and_uhp_to_disk():
    C = Mobius.cayley()
    for x in (-5, -1, 0, 0.3, 2, 40):
        assert abs(abs(C(x)) - 1) < 1e-12
    assert abs(C(1j * 3 + 0.2)) < 1
    img = C.map_line(0, 1)
    assert img.kind == "circle" and close(img.center, 0) and abs(img.radius - 1) < 1e-9


def test_circle_through_pole_becomes_line():
    inv = Mobius(0, 1, 1, 0)  # 1/z
    img = inv.map_circle(1, 1)  # circle |z-1|=1 passes through 0 -> line Re w = 1/2
    assert img.kind == "line"
    assert abs(img.point.real - 0.5) < 1e-9 and abs(abs(img.direction.imag) - 1) < 1e-9


def test_disk_automorphism():
    T = Mobius.disk_automorphism(0.3 + 0.2j)
    assert close(T(0.3 + 0.2j), 0)
    assert abs(abs(T(cmath.exp(1.1j))) - 1) < 1e-12


def test_singular_mobius_rejected():
    with pytest.raises(ValueError):
        Mobius(1, 2, 2, 4)


# -------------------------------------------------------------------- mapping
def test_standard_maps():
    assert close(ck.joukowski(1)(cmath.exp(0.7j)), 2 * cmath.cos(0.7))  # unit circle -> segment
    assert close(ck.cayley(1j * 5 - 1), ck.Mobius.cayley()(1j * 5 - 1))
    assert close(ck.compose(ck.log_map, ck.exp_map)(2 + 1j), 2 + 1j)


def test_conformal_factor():
    scale, rot = ck.conformal_factor(lambda z: z**2, 1j)
    assert abs(scale - 2) < 1e-6 and abs(rot - pi / 2) < 1e-6
    assert not ck.is_conformal_at(lambda z: z**2, 0)


def test_grid_with_pole_does_not_crash():
    grid = ck.rectangular_grid(-1, 1, -1, 1, lines=5, samples=21)  # passes through 0
    mapped = ck.map_grid(ck.inverse, grid)
    assert any(cmath.isnan(w) for line in mapped for w in line)
    with pytest.raises(ZeroDivisionError):
        ck.map_grid(ck.inverse, grid, on_error="raise")


def test_polar_grid_shapes():
    g = ck.polar_grid(1, rings=3, rays=8, samples=10)
    assert len(g) == 3 + 8
    s = ck.polar_grid(1, rings=3, rays=4, theta_min=0, theta_max=pi / 2, samples=10)
    assert len(s) == 3 + 5
