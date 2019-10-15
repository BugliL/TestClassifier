# Create a single sphere of points to train a dummy classifier and see it graphically
# equation
# fx = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r
import random
import pandas as pd

import matplotlib.pyplot as plt
from sympy import symbols, solvers
from mpl_toolkits.mplot3d import Axes3D


def get_solutions_4d(P, r, n=100):
    """
    Creates a list of coordinates that are inside the sphere of Center P and radius R
    :param P: Tuple, center of sphere
    :param r: radius of the sphere
    :param n: number of values to return
    :return: List[Tuple]
    """
    result = []
    x0, y0, z0, w0 = P
    x, y, z, w = symbols('x y z w', real=True)  # ONLY REAL values
    circle_equation = lambda P, r: (x - P[0]) ** 2 + (y - P[1]) ** 2 + (z - P[2]) ** 2 + (w - P[3]) ** 2 - r
    equation = circle_equation((x0, y0, z0, w0), r)
    while len(result) < n:
        x_, y_, z_ = random.uniform(x0 - r, x0 + r), random.uniform(y0 - r, y0 + r), random.uniform(z0 - r, z0 + r),
        evaluated_circle = equation.subs({x: x_, y: y_, z: z_}).evalf()
        result += [(x_, y_, z_, w_) for w_ in (solvers.solve(evaluated_circle))]

        if len(result) % 10 == 0:
            print("Trovati {}".format(len(result)))
    return result


def get_solutions_3d(P, r, n=100):
    """
    Creates a list of coordinates that are inside the sphere of Center P and radius R
    :param P: Tuple, center of sphere
    :param r: radius of the sphere
    :param n: number of values to return
    :return: List[Tuple]
    """
    result = []
    x0, y0, z0 = P
    x, y, z = symbols('x y z', real=True)  # ONLY REAL values
    circle_equation = lambda P, r: (x - P[0]) ** 2 + (y - P[1]) ** 2 + (z - P[2]) ** 2 - r
    equation = circle_equation((x0, y0, z0), r)
    while len(result) < n:
        x_, y_ = random.uniform(x0 - r, x0 + r), random.uniform(y0 - r, y0 + r)
        evaluated_circle = equation.subs({x: x_, y: y_}).evalf()
        z_values = solvers.solve(evaluated_circle)

        while z_values:
            result.append((x_, y_, z_values.pop()))
    return result


def plot_solutions_3d(solutions_list: list):
    def plotting_points(solutions, color, marker='o'):
        ax.scatter(
            [round(x[0], 2) for x in solutions],
            [round(x[1], 2) for x in solutions],
            [round(x[2], 2) for x in solutions],
            c=color, marker=marker)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plotting_points(solutions_list[0], 'r')
    plotting_points(solutions_list[1], 'b')
    plt.show()


if __name__ == '__main__':
    solutions1 = get_solutions_4d((0., 0., 0., 0), 5)
    solutions2 = get_solutions_4d((10., 10., 10., 10.), 10)

    solutions1_pd = pd.DataFrame([x + (0,) for x in solutions1])
    solutions2_pd = pd.DataFrame([x + (1,) for x in solutions2])

    print(solutions1_pd.to_csv(index=False))
    print(solutions2_pd.to_csv(index=False))
    # plot_solutions_3d([solutions1, solutions2])
