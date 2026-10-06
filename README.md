# complexkit

Solve complex-analysis problems in a few lines of Python: contour integrals,
residues, Möbius transformations, conformal maps and pretty pictures.
The core has **zero dependencies**; plotting is optional.

```bash
pip install complexkit            # core
pip install "complexkit[plot]"    # + matplotlib / numpy for plots
```

## Contour integration & residues
```python
import cmath
import complexkit as ck
from math import pi

ck.integrate(lambda z: 1/z, ck.circle())                   # 2πi
ck.integrate(f, ck.rectangle(-1, -1, 1, 1))                # polygons / rectangles
ck.integrate(f, ck.line(0, 1) + ck.arc(0, 1, 0, pi))       # join contours with +
ck.residue(lambda z: 1/(z**2+1), 1j)                       # -i/2
ck.residue_theorem(f, poles=[1j, -1j], contour=ck.circle(0, 2))
ck.winding_number(ck.circle(), 0)                          # 1
ck.argument_principle(lambda z: z**3 - 1, ck.circle(0, 2)) # 3 zeros inside
ck.cauchy_derivative(cmath.exp, 0, order=3)                # f'''(0) = 1
```

## Möbius transformations
```python
from complexkit import Mobius, INF

T = Mobius.from_points((0, 1, 2), (1j, 5, -3+1j))   # 3 point pairs -> unique map
T(0.5), T.inverse(), T @ S                           # call, inverse, compose
T.fixed_points(); T.classify()                       # 'loxodromic', 'elliptic', ...
Mobius(0, 1, 1, 0).map_circle(1, 1)                  # image of |z-1|=1 -> a LINE
Mobius.cayley().map_line(0, 1)                       # real axis -> unit circle
Mobius.disk_automorphism(0.3+0.2j)
ck.cross_ratio(z1, z2, z3, z4)
```

## Mappings, grids, local geometry
```python
ck.plot_mapping(ck.exp_map, ck.rectangular_grid(-2, 2, -3, 3))
ck.plot_mapping(ck.power(2), ck.polar_grid(2, rings=6, rays=16, theta_max=pi))
ck.plot_mapping(ck.joukowski(1), [ck.sample_circle(0, r) for r in (1, 1.2, 1.5)], window=4)

ck.conformal_factor(lambda z: z**2, 1+1j)   # (|f'|, arg f') = (2.83, π/4)
ck.is_conformal_at(lambda z: z**2, 0)       # False: critical point

from complexkit.plotting import domain_coloring
domain_coloring(lambda z: (z**2-1)/(z**2+1j))
```
Available maps: `mobius, power, inverse, exp_map, log_map, sqrt_map, sin_map,
cos_map, joukowski, cayley`, plus `compose(f, g)` (applies f first).

## Development
```bash
pip install -e ".[dev]" && pytest
python -m build && twine upload dist/*
```
