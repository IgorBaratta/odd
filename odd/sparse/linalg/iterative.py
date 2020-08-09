import numpy
from .utils import make_system

norm = numpy.linalg.norm


def cg(
    A,
    b,
    x0=None,
    tol=1e-5,
    maxiter=None,
    xtype=None,
    M=None,
    callback=None,
    residuals=None,
):
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # determine maxiter
    if maxiter is None:
        maxiter = int(1.3 * len(b)) + 2
    elif maxiter < 1:
        raise ValueError("Number of iterations must be positive")

    r = b - A @ x
    z = r.copy()
    p = z.copy()

    rz = numpy.dot(r, z)
    normr = numpy.sqrt(rz)

    if residuals is not None:
        residuals[:] = [normr]  # initial residual

    # Check initial guess ( scaling by b, if b != 0,
    #   must account for case when norm(b) is very small)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0
    if normr < tol * normb:
        return (postprocess(x), 0)

    # Scale tol by ||r_0||_M
    if normr != 0.0:
        tol = tol * normr

    # How often should r be recomputed
    recompute_r = 8
    iter = 0

    while True:
        p.update()
        Ap = A @ p

        rz_old = rz
        # Step number in Saad's pseudocode
        pAp = numpy.dot(Ap, p)  # check curvature of A

        if pAp < 0.0:
            # warn("\nIndefinite matrix detected in CG, aborting\n")
            return (postprocess(x), -1)

        alpha = rz / pAp  # 3
        x = x + alpha * p  # 4

        if numpy.mod(iter, recompute_r) and iter > 0:  # 5
            r = r - alpha * Ap
        else:
            r = b - A * x

        z = r.copy()  # 6
        rz = numpy.dot(r, z)

        if rz < 0.0:  # check curvature of M
            # warn("\nIndefinite preconditioner detected in CG, aborting\n")
            return (postprocess(x), -1)

        beta = rz / rz_old  # 7
        p = p * beta + z  # 8

        iter += 1

        normr = numpy.sqrt(rz)  # use preconditioner norm

        if residuals is not None:
            residuals.append(normr)

        if callback is not None:
            callback(x)

        if normr < tol:
            return (postprocess(x), 0)
        elif rz == 0.0:
            return (postprocess(x), -1)

        if iter == maxiter:
            return (postprocess(x), iter)
