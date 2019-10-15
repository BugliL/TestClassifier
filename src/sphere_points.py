# Create a single sphere of points to train a dummy classifier and see it graphically
# equation
# fx = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r
import random

import matplotlib.pyplot as plt
from sympy import symbols, solvers
from mpl_toolkits.mplot3d import Axes3D

r = 5
x, y, z = symbols('x y z', real=True)  # ONLY REAL values

circle_equation = lambda P, r: (x - P[0]) ** 2 + (y - P[1]) ** 2 + (z - P[2]) ** 2 - r


def get_solutions(P, r, n=100):
    """
    Creates a list of coordinates that are inside the sphere of Center P and radius R
    :param P: Tuple, center of sphere
    :param r: radius of the sphere
    :param n: number of values to return
    :return: List[Tuple]
    """
    result = []
    x0, y0, z0 = P
    equation = circle_equation((x0, y0, z0), r)
    while len(result) < n:
        x_, y_ = random.uniform(x0 - r, x0 + r), random.uniform(y0 - r, y0 + r)
        evaluated_circle = equation.subs({x: x_, y: y_}).evalf()
        z_values = solvers.solve(evaluated_circle)

        while z_values:
            result.append((x_, y_, z_values.pop()))
    return result


solutions1 = get_solutions((0., 0., 0.), 5)
solutions2 = get_solutions((10., 10., 10.), 10)


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

plotting_points(solutions1, 'r')
plotting_points(solutions2, 'b')
plt.show()

if __name__ == '__main__':
    pass
