"""complexkit: numerical complex integration, mappings and Mobius tools."""

from .analysis import (
    argument_principle,
    cauchy_derivative,
    residue,
    residue_theorem,
    winding_number,
)
from .integration import (
    Contour,
    arc,
    circle,
    integrate,
    integrate_segments,
    line,
    polygon,
    rectangle,
)
from .mapping import (
    apply_map,
    cayley,
    compose,
    conformal_factor,
    cos_map,
    derivative,
    exp_map,
    inverse,
    is_conformal_at,
    joukowski,
    log_map,
    map_grid,
    mobius,
    plot_mapping,
    polar_grid,
    power,
    rectangular_grid,
    safe,
    sample_circle,
    sample_line,
    sin_map,
    sqrt_map,
)
from .mobius import INF, GeneralizedCircle, Mobius, circle_through, cross_ratio

__version__ = "0.2.1"

__all__ = [
    "Contour", "GeneralizedCircle", "INF", "Mobius",
    "apply_map", "arc", "argument_principle", "cauchy_derivative", "cayley",
    "circle", "circle_through", "compose", "conformal_factor", "cos_map",
    "cross_ratio", "derivative", "exp_map", "integrate", "integrate_segments",
    "inverse", "is_conformal_at", "joukowski", "line", "log_map", "map_grid",
    "mobius", "plot_mapping", "polar_grid", "polygon", "power",
    "rectangle", "rectangular_grid", "residue", "residue_theorem", "safe",
    "sample_circle", "sample_line", "sin_map", "sqrt_map", "winding_number",
]
