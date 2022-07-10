"""
Module concerning the numerical approximate operations between good functions.
Written in Python 3.9.
"""

from __future__ import annotations
import numpy as np
from typing import *
import matplotlib.pyplot as plt
import bisect


class Good(Callable):
    """
    Class of good functions.
    """
    func: Callable

    def __init__(self, _func: Callable):
        self.func = _func

    def __add__(self, other: Good) -> Good:
        return Good(lambda x: self(x) + other(x))

    def __sub__(self, other: Good) -> Good:
        return Good(lambda x: self(x) - other(x))

    def __mul__(self, other: float) -> Good:
        return Good(lambda x: self(x) * other)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __matmul__(self, other: Good) -> Good:
        return self.oplus(other)

    def inverse(self, max_x: float = 5, step: float = 1e-5) -> Good:
        """
        Computes the pseudo-inverse.

        :param max_x: The maximum number to compute.
        :param step: The interval to compute.
        :return: The pseudo-inverse function.
        """
        val: List[float] = self.compute(max_x, step)

        def out(x: float) -> float:
            """
            Inverse function to return.

            :param x: The argument.
            :return: The evaluation.
            """
            if x <= val[0]:
                return 0
            if x >= val[-1]:
                return np.inf
            k = bisect.bisect_right(val, x) - 1
            if val[k + 1] is np.inf:
                return step * k
            return step * (k + (x - val[k]) / (val[k + 1] - val[k]))
        return Good(out)

    def compute(self, max_x: float, step: float) -> List[float]:
        """
        Computes numerical values up until the given point.

        :param max_x: The maximum number to compute.
        :param step: The interval to compute.
        :return: The evaluation.
        """
        return [self(x) for x in np.arange(0, max_x + step, step)]

    def oplus(self, other) -> Good:
        """
        Computes (f*+g*)*.

        :param other: The function g.
        :return: The resulting function.
        """
        return (self.inverse() + other.inverse()).inverse()

    def draw(self, max_x: float = 5, step: float = 1e-5, **kwargs) -> None:
        """
        Draws function using plt. Use plt.show() after calling.

        :param max_x: The maximum number to compute.
        :param step: The interval to compute.
        """
        x = np.arange(0, max_x + step, step)
        y = self.compute(max_x, step)
        plt.plot(x, y, **kwargs)
        plt.gca().set_xlim(left=0)
        plt.gca().set_ylim(bottom=0)


def delta_to_wye(a: List[Good]) -> List[Good]:
    """
    Returns the functions given by a transformed into a Y-shape. Order corresponds to facing sides.

    :param a: The Delta-form latency functions.
    :return: The Y-form latency functions.
    """
    return [
        ((a[0] + a[1]) @ a[2] + (a[0] + a[2]) @ a[1] - (a[1] + a[2]) @ a[0]) * 0.5,
        ((a[1] + a[2]) @ a[0] + (a[0] + a[1]) @ a[2] - (a[0] + a[2]) @ a[1]) * 0.5,
        ((a[0] + a[2]) @ a[1] + (a[0] + a[1]) @ a[2] - (a[0] + a[1]) @ a[2]) * 0.5
    ]
