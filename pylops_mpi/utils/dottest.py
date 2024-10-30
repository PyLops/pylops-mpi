__all__ = ["dottest"]

from typing import Optional

import numpy as np

from pylops_mpi.DistributedArray import DistributedArray
from pylops.utils.backend import to_numpy


def dottest(
    Op,
    u: DistributedArray,
    v: DistributedArray,
    nr: Optional[int] = None,
    nc: Optional[int] = None,
    rtol: float = 1e-6,
    atol: float = 1e-21,
    raiseerror: bool = True,
    verb: bool = False,
) -> bool:
    r"""Dot test.

    Perform dot-test to verify the validity of forward and adjoint
    operators using user-provided random vectors :math:`\mathbf{u}`
    and :math:`\mathbf{v}` (whose Partition must be consistent with
    the operator being tested). This test can help to detect errors
    in the operator ximplementation.

    Parameters
    ----------
    Op : :obj:`pylops_mpi.LinearOperator`
        Linear operator to test.
    u : :obj:`pylops_mpi.DistributedArray`
        Distributed array of size equal to the number of columns of operator
    v : :obj:`pylops_mpi.DistributedArray`
        Distributed array of size equal to the number of rows of operator
    nr : :obj:`int`
        Number of rows of operator (i.e., elements in data)
    nc : :obj:`int`
        Number of columns of operator (i.e., elements in model)
    rtol : :obj:`float`, optional
        Relative dottest tolerance
    atol : :obj:`float`, optional
        Absolute dottest tolerance
        .. versionadded:: 2.0.0
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when dottest fails
    verb : :obj:`bool`, optional
        Verbosity

    Returns
    -------
    passed : :obj:`bool`
        Passed flag.

    Raises
    ------
    AssertionError
        If dot-test is not verified within chosen tolerances.

    Notes
    -----
    A dot-test is mathematical tool used in the development of numerical
    linear operators.

    More specifically, a correct implementation of forward and adjoint for
    a linear operator should verify the following *equality*
    within a numerical tolerance:

    .. math::
        (\mathbf{Op}\,\mathbf{u})^H\mathbf{v} =
        \mathbf{u}^H(\mathbf{Op}^H\mathbf{v})

    """
    if nr is None:
        nr = Op.shape[0]
    if nc is None:
        nc = Op.shape[1]

    if (nr, nc) != Op.shape:
        raise AssertionError("Provided nr and nc do not match operator shape")

    y = Op.matvec(u)  # Op * u
    x = Op.rmatvec(v)  # Op'* v

    yy = np.vdot(y.asarray(), v.asarray())  # (Op  * u)' * v
    xx = np.vdot(u.asarray(), x.asarray())  # u' * (Op' * v)

    # convert back to numpy (in case cupy arrays were used), make into a numpy
    # array and extract the first element. This is ugly but allows to handle
    # complex numbers in subsequent prints also when using cupy arrays.
    xx, yy = np.array([to_numpy(xx)])[0], np.array([to_numpy(yy)])[0]

    # evaluate if dot test passed
    passed = np.isclose(xx, yy, rtol, atol)

    # verbosity or error raising
    if (not passed and raiseerror) or verb:
        passed_status = "passed" if passed else "failed"
        msg = f"Dot test {passed_status}, v^H(Opu)={yy} - u^H(Op^Hv)={xx}"
        if not passed and raiseerror:
            raise AssertionError(msg)
        else:
            print(msg)

    return passed
