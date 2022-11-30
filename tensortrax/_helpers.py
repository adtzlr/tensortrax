r"""
 _                            
| |                          ████████╗██████╗  █████╗ ██╗  ██╗
| |_ ___ _ __  ___  ___  _ __╚══██╔══╝██╔══██╗██╔══██╗╚██╗██╔╝
| __/ _ \ '_ \/ __|/ _ \| '__|  ██║   ██████╔╝███████║ ╚███╔╝ 
| ||  __/ | | \__ \ (_) | |     ██║   ██╔══██╗██╔══██║ ██╔██╗ 
 \__\___|_| |_|___/\___/|_|     ██║   ██║  ██║██║  ██║██╔╝ ██╗
                                ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  
"""


def f(A):
    "Return the value of the tensor."
    return A.x


def δ(A):
    "Return the (dual) δ-variation of the tensor."
    return A.δx


def Δ(A):
    "Return the (dual) Δ-variation of the tensor."
    return A.Δx


def Δδ(A):
    "Return the (dual) Δδ-variation of the tensor."
    return A.Δδx
