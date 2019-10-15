# Create a single sphere of points to train a dummy classifier and see it graphically
# equation
# fx = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r
import random

import matplotlib.pyplot as plt
from sympy import symbols, solvers
from mpl_toolkits.mplot3d import Axes3D

r = 5
x0, y0, z0 = (0., 0., 0.)
x, y, z = symbols('x y z', real=True)

circle_equation = lambda P, r: (x - P[0]) ** 2 + (y - P[1]) ** 2 + (z - P[2]) ** 2 - r
print(circle_equation((x0, y0, z0), r))

solutions = []
while len(solutions) < 100:
    for (x_, y_) in [(random.uniform(x0 - r, x0 + r), random.uniform(y0 - r, y0 + r)) for k in range(100)]:
        evaluated_circle = circle_equation((x0, y0, z0), r).subs({x: x_, y: y_}).evalf()
        z_values = solvers.solve(evaluated_circle, real=True)

        while z_values:
            solutions.append((x_, y_, z_values.pop()))

print("\n".join([str(t) for t in solutions]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def plotting_points(solutions, color, marker):
    ax.scatter(
        [round(x[0], 2) for x in solutions],
        [round(x[1], 2) for x in solutions],
        [round(x[2], 2) for x in solutions],
        c=color, marker=marker)


plotting_points(solutions, 'r', 'o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

if __name__ == '__main__':
    pass
