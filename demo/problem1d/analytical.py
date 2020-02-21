import sympy as sym

x, L, C, D, c_0, c_1, = sym.symbols('x L C D c_0 c_1')


def model1(f, L):
    """Solve -u’’ = f(x), u(0)=0, u(L)=0."""
    # Integrate twice
    u_x = - sym.integrate(f, (x, 0, x)) + c_0
    u = sym.integrate(u_x, (x, 0, x)) + c_1
    # Set up 2 equations from the 2 boundary conditions and solve
    # with respect to the integration constants c_0, c_1
    r = sym.solve([u.subs(x, 0)-0,   # x=0 condition
                   u.subs(x, L)-0],  # x=L condition
                  [c_0, c_1])        # unknowns
    # Substitute the integration constants in the solution
    u = u.subs(c_0, r[c_0]).subs(c_1, r[c_1])
    u = sym.simplify(sym.expand(u))
    return u


f = sym.sin(x)
L = sym.pi
u = model1(f, L)
print('model1:', u)
