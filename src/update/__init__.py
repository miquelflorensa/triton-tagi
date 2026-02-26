"""TAGI update rules: observation innovation and parameter updates."""

from .observation import compute_innovation
from .parameters import update_parameters, get_cap_factor
