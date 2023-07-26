r"""
 _
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
"""

from ._linalg_array import det as _det
from ._linalg_array import inv as _inv
from ._linalg_array import pinv as _pinv
from ._linalg_tensor import det, eigh, eigvalsh, expm, inv, pinv

__all__ = ["_det", "_inv", "_pinv", "det", "eigh", "eigvalsh", "expm", "inv", "pinv"]
