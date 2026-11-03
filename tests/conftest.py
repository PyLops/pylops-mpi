from _pytest.nodeid import NodeId
from mpi4py import MPI


def pytest_itemcollected(item):
    """Append MPI rank to the test ID as it is collected."""
    rank = MPI.COMM_WORLD.Get_rank()
    nodeid = f"{item.nodeid}[Rank {rank}]"
    item._id = NodeId.parse(nodeid)
