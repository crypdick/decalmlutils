"""
Calculates the bounds for the distance to a decision plane that correspond to confidence values between 0.45 and 0.55
using a sigmoid function.

distances to the decision plane are unbounded (-inf, inf)
confidences are bounded to [0, 1]

the distances to the plane are converted to confidences using the sigmoid function

sigmoid(0) = 0.5
... which means that when the distance to the decision plane is 0, the confidence is 0.5

what are the inputs to the sigmoid function that result in confidence values between 0.45 and 0.55?
"""

import numpy as np

LOWER = 0.35
UPPER = 0.65
TEST_BOUND_LOWER = -1
TEST_BOUND_UPPER = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    # we know that the solution is bounded by -1 and 1, so we can zoom in on that range
    x = np.linspace(TEST_BOUND_LOWER, TEST_BOUND_UPPER, 100)
    print(f"x: {x}")
    y = sigmoid(x)
    print(f"y: {y}")

    # find the indices of the values that are between 0.45 and 0.55
    indices = np.where((y >= LOWER) & (y <= UPPER))
    indices = indices[0]
    print(f"indices: {indices}")
    # find the bounds
    bottom_ix, top_ix = indices[0], indices[-1]
    # convert the ix to the actual values
    bottom, top = x[bottom_ix], x[top_ix]
    print(f"bottom: {bottom}, top: {top}")

    import matplotlib.pyplot as plt

    plt.plot(x, y)
    plt.vlines(
        bottom, 0, 1, color="r", linestyle="--", label=f"lower bound= {bottom:0.3f}"
    )
    plt.vlines(top, 0, 1, color="r", linestyle="--", label=f"upper bound= {top:0.3f}")
    plt.hlines(0.35, TEST_BOUND_LOWER, TEST_BOUND_UPPER, color="b", linestyle=":")
    plt.hlines(0.65, TEST_BOUND_LOWER, TEST_BOUND_UPPER, color="b", linestyle=":")
    plt.legend()
    plt.title(
        f"Upper and lower bounds for decision plane distance for confidences between {LOWER} and {UPPER}"
    )
    plt.xlabel("Distance to decision plane")
    plt.ylabel("Confidence")
    plt.show()


if __name__ == "__main__":
    main()
