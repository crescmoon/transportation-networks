"""
Main file of project.
Contains functions used to visualize the latency functions from the 4-node, 5-edge example.
Written in Python 3.9.
"""

from goodfuncs import *  # np, typing, plt


LATENCY_STEP = 1e-5


def lin(a: float = 1, b: float = 0) -> Good:
    """
    Returns y = ax + b.

    :param a: The coefficient of x.
    :param b: The constant.
    :return: The function ax + b.
    """
    return Good(lambda x: a * x + b)


def const(y: float = 1) -> Good:
    """
    Returns the constant function object.

    :param y: The constant for the function to return.
    :return: The constant function.
    """
    return Good(lambda x: y)


def get_latencies(ls: List[Good], f: List[float]) -> List[float]:
    """
    Evaluates the latencies each flow experiences.

    :param ls: The 5 latency functions.
    :param f: Flow for each path.
    :return: Latency for each flow.
    """
    return [
        ls[0](f[0] + f[1]) + ls[1](f[0] + f[3]),
        ls[0](f[0] + f[1]) + ls[4](f[1] + f[3]) + ls[3](f[1] + f[2]),
        ls[2](f[2] + f[3]) + ls[3](f[1] + f[2]),
        ls[2](f[2] + f[3]) + ls[4](f[1] + f[3]) + ls[1](f[0] + f[3])
    ]


def evaluate_at(ls: List[Good], x: float) -> float:
    """
    Evaluates the user equilibrium given the input flow of the graph.
    Implemented with naive simulation.

    :param ls: The 5 latency functions.
    :param x: Input flow.
    :return: Total latency.
    """
    f: List[float] = [x / 4] * 4
    # f[0]: Flow through ls[0]-ls[1]
    # f[1]: Flow through ls[0]-ls[4]-ls[3]
    # f[2]: Flow through ls[2]-ls[3]
    # f[3]: Flow through ls[2]-ls[4]-ls[1]

    lats = get_latencies(ls, f)

    step: float = x / 40
    while step > x / (4 * 10 ** 7):
        for i in range(20):
            m = [k for k in range(4) if lats[k] <= min(lats) + LATENCY_STEP]
            for j in range(4):
                if lats[j] > min(lats) + LATENCY_STEP and f[j] > 0:
                    f[j] -= step
                    for k in m:
                        f[k] += step / len(m)
            lats = get_latencies(ls, f)
        step /= 10
    return min(lats)


def evaluate_graph(ls: List[Good], max_x: float = 5, x_step: float = 0.01) -> None:
    """
    Evaluates the total latency function of the graph.

    :param ls: The 5 latency functions.
    :param max_x: The maximum x value to graph.
    :param x_step: Step of the x value to graph.
    """
    xs: List[float] = []
    ys: List[float] = []
    for x in np.arange(0, max_x, x_step):
        xs.append(x)
        ys.append(evaluate_at(ls, x))
    plt.plot(xs, ys)
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    plt.show()


# Drawing Fig 5.
# plt.figure()
# plt.xlabel("X")
# plt.ylabel("T")
# plt.title("Functions evaluated by the Delta to Wye formula")
# wye = delta_to_wye([
#     Good(lambda x: 1 + 0.05 * x ** 4),
#     Good(lambda x: 3 + 0.01 * x ** 3.5),
#     Good(lambda x: 2.5 + 0.025 * x ** 4)
# ])
# wye[0].draw(color="red")
# wye[1].draw(color="blue")
# wye[2].draw(color="green")
# plt.show()

# Drawing Fig #.
# plt.figure()
# plt.xlabel("X")
# plt.ylabel("T")
# plt.title("Total cost by input flow")
# evaluate_graph([lin(), const(), const(), lin(), const(0.5)])
# plt.show()
