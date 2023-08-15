.. _addingoperator:

Implementing new operators
==========================
Users are welcome to create new operators and add them to the PyLops-MPI library.

In this tutorial, we will go through the key steps in the definition of an operator, using the
:py:class:`pylops_mpi.basicoperators.MPIBlockDiag` as an example. This operator creates a block-diagonal matrix of
N pylops linear operators.

Creating the operator
---------------------
The first thing we need to do is to create a new file with the name of the operator we would like to implement.
Note that as the operator will be a class, we need to follow the UpperCaseCamelCase convention both for the class itself
and for the filename. It's recommended to prefix the class name with ``MPI`` to distinguish it as an MPI Operator.

Once we have created the file, we will start by importing the modules that will be needed by the operator.
While this varies from operator to operator, you will always need to import the :py:class:`pylops_mpi.DistributedArray` class
which we use in this library as an alternative to the NumPy arrays. This class allows you to work with distributed arrays that
are partitioned/broadcasted across different ranks or processes in a parallel computing environment. Additionally, you will
need to import the :py:class:`pylops_mpi.MPILinearOperator` class, which serves as the **parent** class for any of our operators:

.. code-block:: python

   from pylops_mpi import DistributedArray, MPILinearOperator

After that we define our new operator:

.. code-block:: python

   class MPIBlockDiag(MPILinearOperator):

followed by a `numpydoc docstring <https://numpydoc.readthedocs.io/en/latest/format.html>`__
(starting with ``r"""`` and ending with ``"""``) containing the documentation of the operator. Such docstring should
contain at least a short description of the operator, a ``Parameters`` section with a detailed description of the
input parameters and a ``Notes`` section providing a mathematical explanation of the operator. Take a look at
some of the core operators of PyLops-MPI to get a feeling of the level of details of the mathematical explanation.

Initialization (``__init__``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We then need to create the ``__init__`` where the input parameters are passed and saved as members of our class.
While the input parameters change from operator to operator, it is always required to create three members, the first
called ``shape`` with a tuple containing the dimensions of the operator in the data and model space, the second
called ``dtype`` with the data type object (:obj:`np.dtype`) of the model and data, and the third called ``base_comm``
with an MPI base communicator (:obj:`mpi4py.MPI.Comm`, default set to :obj:`mpi4py.MPI.COMM_WORLD`) responsible for
communicating between different processes. In the context of MPIBlockDiag, we calculate the ``shape`` by performing
a sum-reduction on the shapes of each operator. If the ``dtype`` is not provided, it is determined from the operators,
while the ``base_comm`` is set to ``MPI.COMM_WORLD`` if not provided.

.. code-block:: python

    def __init__(self, ops: Sequence[LinearOperator],
                 base_comm: MPI.Comm = MPI.COMM_WORLD,
                 dtype: Optional[DTypeLike] = None):
        self.ops = ops
        mops = np.zeros(len(self.ops), dtype=np.int64)
        nops = np.zeros(len(self.ops), dtype=np.int64)
        for iop, oper in enumerate(self.ops):
            nops[iop] = oper.shape[0]
            mops[iop] = oper.shape[1]
        self.mops = mops.sum()
        self.nops = nops.sum()
        shape = (base_comm.allreduce(self.nops), base_comm.allreduce(self.mops))
        dtype = _get_dtype(ops) if dtype is None else np.dtype(dtype)
        super().__init__(shape=shape, dtype=dtype, base_comm=base_comm)

Forward mode (``_matvec``)
^^^^^^^^^^^^^^^^^^^^^^^^^^
We can then move onto writing the *forward mode* in the method ``_matvec``. In other words, we will need to write
the piece of code that will implement the following operation :math:`\mathbf{y} = \mathbf{A}\mathbf{x}`.
Such method is always composed of two inputs (the object itself ``self`` and the input model  ``x``).
Here, both the input model ``x`` and input data ``y`` are instances of :py:class:`pylops_mpi.DistributedArray`.
In the case of MPIBlockDiag, each set of operators performs a matrix-vector product in forward mode,
and the final result is collected in a DistributedArray.

.. code-block:: python

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        y = DistributedArray(global_shape=self.shape[0], local_shapes=self.local_shapes_n, dtype=x.dtype)
        y1 = []
        for iop, oper in enumerate(self.ops):
            y1.append(oper.matvec(x.local_array[self.mmops[iop]:
                                                self.mmops[iop + 1]]))
        y[:] = np.concatenate(y1)
        return y

Adjoint mode (``_rmatvec``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Finally we need to implement the *adjoint mode* in the method ``_rmatvec``. In other words, we will need to write
the piece of code that will implement the following operation :math:`\mathbf{x} = \mathbf{A}^H\mathbf{y}`.
Such method is also composed of two inputs (the object itself ``self`` and the input data ``y``).
Similar to ``_matvec``, both the input model ``x`` and input data ``y`` are instances of :py:class:`pylops_mpi.DistributedArray`.
In the case of MPIBlockDiag, each set of operators performs a matrix-vector product in adjoint mode,
and the final result is collected in a DistributedArray.

.. code-block:: python

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        y = DistributedArray(global_shape=self.shape[1], local_shapes=self.local_shapes_m, dtype=x.dtype)
        y1 = []
        for iop, oper in enumerate(self.ops):
            y1.append(oper.rmatvec(x.local_array[self.nnops[iop]:
                                                 self.nnops[iop + 1]]))
        y[:] = np.concatenate(y1)
        return y

And that's it, we have implemented our first MPI Linear Operator!

Testing the operator
--------------------
Being able to write an operator is not yet a guarantee of the fact that the operator is correct, or in other words
that the adjoint code is actually the *adjoint* of the forward code.
We add tests for the operator by creating a new test within an existing/new ``test_*.py`` file in the ``tests`` folder.

Generally a test file will start with a number of dictionaries containing different parameters we would like to
use in the testing of one or more operators. The test itself starts with two **decorators**. The first **decorator** indicates
that the tests need to be run with MPI processes, with a ``min_size`` of 2. The second **decorator** contains a list of all
(or some) of the dictionaries that will be used for our specific operator, which is followed by the definition of the
test.

.. code-block:: python

    @pytest.mark.mpi(min_size=2)
    @pytest.mark.parametrize("par", [(par1), (par2)])
    def test_blockdiag(par):

After this, you can write your test for the operator inside this method. We recommend using the :py:func:`numpy.testing.assert_allclose`
function with an ``rtol=1e-14`` to check the functionality of the operator. For assistance, you can refer to other test files
in the ``tests`` folder.

Documenting the operator
------------------------
Once the operator has been created, we can add it to the documentation of PyLops-MPI. To do so, simply add the name of
the operator within the ``index.rst`` file in ``docs/source/api`` directory.

Moreover, in order to facilitate the user of your operator by other users, a simple example should be provided as part of the
Sphinx-gallery of the documentation of the PyLops-MPI library. The directory ``examples`` contains several scripts that
can be used as template.

Final checklist
---------------

Before submitting your new operator for review, use the following **checklist** to ensure that your code
adheres to the guidelines of PyLops-MPI:

- you have created a new file containing a single class (or a function when the new operator is a simple combination of
  existing operators and added to a new or existing directory within the ``pylops_mpi`` package.

- the new class contains at least ``__init__``, ``_matvec`` and ``_rmatvec`` methods.

- the new class (or function) has a `numpydoc docstring <https://numpydoc.readthedocs.io/>`__ documenting
  at least the input ``Parameters`` and with a ``Notes`` section providing a mathematical explanation of the operator.

- a new test has been added to an existing ``test_*.py`` file within the ``tests`` folder. Moreover it is advisable to
  create a small toy example where the operator is applied in forward mode.

- the new operator is used within at least one *example* (in ``examples`` directory) or one *tutorial*
  (in ``tutorials`` directory).
