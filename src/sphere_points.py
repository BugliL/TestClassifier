# Create a single sphere of points to train a dummy classifier and see it graphically
# equation
# fx = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r
import random

from sympy import symbols, lambdify
from sympy.abc import x, y, z, w

r = 10
x0, y0, z0 = (0., 0., 0.)
# x, y, z = symbols('x y z')

circle_equation = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r
print(circle_equation)

for (x_, y_) in [(random.randint(0, 100), random.randint(0, 100)) for k in range(10)]:
    print()

if __name__ == '__main__':
    pass
# 
# >>> from sympy import Symbol
# >>> x, y = Symbol('x y')
# >>> f = x + y
# >>> f.subs({x:10, y: 20})
# >>> f
# 30
