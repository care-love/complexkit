"""Quick tour of complexkit. Run:  python examples/demo.py   (plots need matplotlib)"""
from math import pi

import complexkit as ck
from complexkit import Mobius

# 1. Contour integrals -------------------------------------------------------
print("∮ 1/z over unit circle  =", ck.integrate(lambda z: 1 / z, ck.circle()), " (2πi =", 2j * pi, ")")
print("∮ 1/(z²+1) over |z|=2   =", ck.integrate(lambda z: 1 / (z**2 + 1), ck.circle(0, 2)))
print("Residue of 1/(z²+1) at i =", ck.residue(lambda z: 1 / (z**2 + 1), 1j))
print("Zeros of z³-1 in |z|<2   =", ck.argument_principle(lambda z: z**3 - 1, ck.circle(0, 2)))

# 2. Möbius problems ---------------------------------------------------------
T = Mobius.from_points((0, 1, 2), (1j, 5, -3 + 1j))
print("\nMöbius sending 0,1,2 -> i,5,-3+i :", T)
print("Fixed points :", T.fixed_points(), "| type:", T.classify())
print("Image of |z-1|=1 under 1/z :", Mobius(0, 1, 1, 0).map_circle(1, 1))
print("Image of real axis under Cayley :", Mobius.cayley().map_line(0, 1))

# 3. Local geometry of a map -------------------------------------------------
scale, rot = ck.conformal_factor(lambda z: z**2, 1 + 1j)
print(f"\nz² at 1+i stretches by {scale:.4f} and rotates by {rot:.4f} rad")

# 4. Pictures ----------------------------------------------------------------
try:
    ck.plot_mapping(ck.exp_map, ck.rectangular_grid(-2, 2, -3, 3, lines=13))
    ck.plot_mapping(ck.joukowski(1), [ck.sample_circle(0, r) for r in (1, 1.2, 1.5, 2)], window=4)
    from complexkit.plotting import domain_coloring
    domain_coloring(lambda z: (z**2 - 1) / (z**2 + 1j), (-2, 2), (-2, 2))
except ImportError:
    print("(install matplotlib + numpy to see the plots)")
