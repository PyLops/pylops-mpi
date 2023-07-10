Search.setIndex({"docnames": ["api/generated/pylops_mpi.DistributedArray", "api/generated/pylops_mpi.MPILinearOperator", "api/generated/pylops_mpi.Partition", "api/generated/pylops_mpi.basicoperators.MPIBlockDiag", "api/generated/pylops_mpi.optimization.basic.cgls", "api/generated/pylops_mpi.optimization.cls_basic.CGLS", "api/index", "gallery/index", "gallery/plot_distributed_array", "gallery/plot_post_stack_inversion", "gallery/sg_execution_times", "index"], "filenames": ["api/generated/pylops_mpi.DistributedArray.rst", "api/generated/pylops_mpi.MPILinearOperator.rst", "api/generated/pylops_mpi.Partition.rst", "api/generated/pylops_mpi.basicoperators.MPIBlockDiag.rst", "api/generated/pylops_mpi.optimization.basic.cgls.rst", "api/generated/pylops_mpi.optimization.cls_basic.CGLS.rst", "api/index.rst", "gallery/index.rst", "gallery/plot_distributed_array.rst", "gallery/plot_post_stack_inversion.rst", "gallery/sg_execution_times.rst", "index.rst"], "titles": ["pylops_mpi.DistributedArray", "pylops_mpi.MPILinearOperator", "pylops_mpi.Partition", "pylops_mpi.basicoperators.MPIBlockDiag", "pylops_mpi.optimization.basic.cgls", "pylops_mpi.optimization.cls_basic.CGLS", "PyLops MPI API", "Gallery", "Distributed Array", "Post Stack Inversion - 3D", "Computation times", "Overview"], "terms": {"class": [0, 1, 2, 3, 5, 8], "global_shap": [0, 8, 9], "base_comm": [0, 1, 3], "mpi4pi": [0, 1, 3, 9], "mpi": [0, 1, 3, 4, 9, 11], "intracomm": [0, 1, 3], "object": [0, 1, 3], "partit": [0, 3, 8], "scatter": [0, 2, 8], "axi": [0, 8, 9], "0": [0, 3, 4, 8, 9, 10], "dtype": [0, 1, 3], "numpi": [0, 4, 8, 9], "float64": 0, "sourc": [0, 1, 2, 3, 4, 5, 8, 9], "distribut": [0, 1, 2, 4, 5, 6, 7, 9, 10, 11], "arrai": [0, 1, 2, 3, 7, 10], "multidimension": 0, "like": 0, "It": [0, 9], "bring": 0, "high": 0, "perform": [0, 1, 3, 9], "comput": [0, 3, 6, 8, 9, 11], "paramet": [0, 1, 3, 4, 5, 8, 9], "tupl": [0, 1, 3, 4], "int": [0, 1, 4], "shape": [0, 1, 3, 8, 9], "global": [0, 8], "comm": [0, 1, 3], "option": [0, 1, 3, 4], "commun": [0, 1, 3], "over": 0, "which": [0, 3, 9], "i": [0, 1, 3, 4, 5, 8, 9, 11], "default": [0, 1, 3], "comm_world": [0, 1, 3, 9], "broadcast": [0, 2], "str": [0, 1, 3], "type": [0, 1, 2, 3], "element": [0, 1, 3, 4, 8], "input": [0, 1, 3, 8], "along": [0, 8, 9], "occur": 0, "method": [0, 1, 2, 3, 5, 8], "post": [0, 1, 7, 10], "stack": [0, 1, 3, 7, 10], "invers": [0, 1, 6, 7, 10], "3d": [0, 1, 7, 10], "2023": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "op": [1, 3, 4, 5, 9], "none": [1, 2, 3, 4, 5], "common": 1, "interfac": [1, 6], "matrix": 1, "vector": [1, 3], "product": 1, "fashion": 1, "thi": [1, 3, 8, 9], "provid": [1, 3, 8], "adjoint": [1, 3], "end": [1, 3, 8, 9], "user": 1, "pylop": [1, 3, 9, 11], "should": 1, "directli": 1, "simpli": 1, "oper": [1, 3, 4, 5, 8, 9, 11], "ar": [1, 4, 9], "alreadi": 1, "implement": [1, 8, 9], "meant": 1, "develop": 1, "onli": [1, 9], "ha": [1, 3], "parent": 1, "ani": 1, "new": 1, "within": [1, 3], "linearoper": [1, 3, 9], "linear": [1, 3, 4, 9, 11], "base": [1, 3], "valu": 2, "name": 2, "modul": 2, "qualnam": 2, "start": [2, 8, 9], "1": [2, 4, 8, 9], "boundari": 2, "enum": 2, "data": [2, 3, 4, 5, 9], "among": 2, "differ": [2, 3, 9], "process": [2, 6, 8, 11], "all": [2, 3, 8, 9], "uniqu": 2, "portion": [2, 8], "each": [2, 3, 4, 8, 9], "block": 3, "diagon": 3, "creat": [3, 9], "from": [3, 8, 9], "set": [3, 4], "us": [3, 4, 5, 7, 8, 9], "rank": [3, 8, 9], "must": 3, "initi": [3, 4], "one": [3, 9], "more": 3, "both": 3, "model": [3, 4, 9], "distributedarrai": [3, 4, 8, 9], "between": [3, 4], "accord": 3, "list": [3, 4], "One": 3, "note": [3, 4, 5, 9], "an": [3, 4, 5, 8], "compos": 3, "n": [3, 4, 5, 8], "repres": 3, "l": 3, "we": [3, 9], "here": [3, 8], "compactli": 3, "mathbf": [3, 4, 5], "_i": 3, "forward": [3, 9], "mode": 3, "its": [3, 8], "correspond": [3, 8], "denot": 3, "m": [3, 4, 5, 9], "effect": 3, "wai": [3, 8], "local": [3, 8, 9], "agre": 3, "those": 3, "The": [3, 6], "collect": [3, 9], "refer": 3, "d": [3, 9], "begin": 3, "bmatrix": 3, "_1": 3, "_2": [3, 4, 5], "vdot": 3, "_n": 3, "ldot": 3, "ddot": 3, "likewis": 3, "execut": [3, 10], "h": [3, 9], "attribut": 3, "y": [4, 5, 9], "x0": 4, "niter": 4, "10": [4, 8], "damp": [4, 5], "tol": 4, "0001": 4, "show": [4, 8], "fals": 4, "itershow": 4, "callback": [4, 5], "conjug": [4, 5], "gradient": [4, 5], "least": [4, 5], "squar": [4, 5], "solv": [4, 5], "overdetermin": [4, 5], "system": [4, 5], "equat": [4, 5], "given": [4, 5], "mpilinearoper": [4, 5], "iter": [4, 5], "invert": [4, 5], "size": [4, 5, 9], "time": [4, 5, 8, 9], "guess": 4, "number": 4, "float": [4, 8, 9], "coeffici": [4, 5], "toler": 4, "residu": 4, "norm": 4, "bool": 4, "displai": 4, "log": [4, 9], "first": [4, 9], "n1": 4, "step": 4, "last": 4, "n2": 4, "everi": 4, "n3": 4, "where": [4, 5], "three": 4, "callabl": 4, "function": [4, 5], "signatur": 4, "x": [4, 5, 8, 9], "call": 4, "after": 4, "return": 4, "estim": 4, "istop": 4, "give": [4, 9], "reason": 4, "termin": 4, "mean": 4, "approxim": 4, "solut": 4, "2": [4, 5, 8, 9], "problem": 4, "iit": 4, "upon": 4, "r1norm": 4, "r": 4, "r2norm": 4, "sqrt": 4, "t": [4, 9], "epsilon": [4, 5], "equal": 4, "cost": 4, "ndarrai": 4, "histori": 4, "through": 4, "see": 4, "cls_basic": 4, "minim": 5, "follow": 5, "j": 5, "applic": 6, "program": 6, "enabl": [6, 11], "parallel": [6, 8, 11], "larg": [6, 11], "scale": [6, 11], "algebra": [6, 11], "exampl": [7, 8, 9], "pylops_mpi": [7, 8, 9], "gener": [7, 8, 9], "sphinx": [7, 8, 9], "go": [8, 9], "download": [8, 9], "full": [8, 9], "code": [8, 9], "how": 8, "across": 8, "multipl": 8, "environ": 8, "matplotlib": [8, 9], "import": [8, 9], "pyplot": [8, 9], "plt": [8, 9], "np": [8, 9], "close": [8, 9], "random": 8, "seed": 8, "42": 8, "defin": [8, 9], "5": [8, 9], "let": [8, 9], "": [8, 9], "arr": 8, "fill": 8, "arang": [8, 9], "local_shap": 8, "reshap": [8, 9], "plot_distributed_arrai": [8, 10], "below": 8, "second": [8, 9], "To": 8, "convert": 8, "you": 8, "can": 8, "to_dist": 8, "classmethod": 8, "allow": 8, "depict": 8, "same": [8, 9], "arr1": 8, "arr2": 8, "plot": 8, "plot_local_arrai": 8, "vmin": [8, 9], "vmax": [8, 9], "wise": 8, "addit": 8, "add": 8, "togeth": 8, "sum_arr": 8, "subtract": 8, "diff_arr": 8, "multipli": 8, "mult_arr": 8, "total": [8, 9, 10], "run": [8, 9], "script": [8, 9], "minut": [8, 9], "243": [8, 10], "python": [8, 9, 11], "py": [8, 9, 10], "jupyt": [8, 9], "notebook": [8, 9], "ipynb": [8, 9], "galleri": [8, 9, 10], "illustr": 9, "demonstr": 9, "involv": 9, "synthet": 9, "seismic": 9, "subsurfac": 9, "acoust": 9, "imped": 9, "scipi": 9, "signal": 9, "filtfilt": 9, "util": 9, "wavelet": 9, "ricker": 9, "basicoper": 9, "transpos": 9, "avo": 9, "poststack": 9, "poststacklinearmodel": 9, "mpiblockdiag": 9, "get_rank": 9, "get_siz": 9, "requir": 9, "load": 9, "testdata": 9, "poststack_model": 9, "npz": 9, "z": 9, "make": 9, "ny_i": 9, "20": 9, "direct": 9, "m3d_i": 9, "tile": 9, "newaxi": 9, "nx": 9, "nz": 9, "ny": 9, "allreduc": 9, "smooth": 9, "nsmoothi": 9, "nsmoothx": 9, "nsmoothz": 9, "30": 9, "mback3d_i": 9, "ones": 9, "dt": 9, "004": 9, "t0": 9, "ntwav": 9, "41": 9, "wav": 9, "15": 9, "m3d": 9, "mback3d": 9, "concaten": 9, "allgath": 9, "now": 9, "version": 9, "subset": 9, "Such": 9, "pass": 9, "individu": 9, "overal": 9, "simplifi": 9, "handl": 9, "split": 9, "rearrang": 9, "form": 9, "flatten": 9, "m3d_dist": 9, "ppop": 9, "nt0": 9, "spatdim": 9, "top": [9, 11], "bdiag": 9, "d_dist": 9, "d_local": 9, "local_arrai": 9, "asarrai": 9, "check": 9, "result": 9, "rank0": 9, "ppop0": 9, "d0": 9, "two": 9, "print": 9, "distr": 9, "allclos": 9, "visual": 9, "fig": 9, "ax": 9, "subplot": 9, "nrow": 9, "3": 9, "ncol": 9, "figsiz": 9, "9": 9, "12": 9, "constrained_layout": 9, "true": 9, "imshow": 9, "cmap": 9, "gist_rainbow": 9, "min": 9, "max": 9, "set_titl": 9, "tight": 9, "400": 9, "220": 9, "grai": 9, "7": 9, "699": [9, 10], "plot_post_stack_invers": [9, 10], "00": 10, "08": 10, "942": 10, "file": 10, "07": 10, "mb": 10, "01": 10, "librari": 11, "built": 11, "design": 11}, "objects": {"pylops_mpi": [[0, 0, 1, "", "DistributedArray"], [1, 0, 1, "", "MPILinearOperator"], [2, 0, 1, "", "Partition"]], "pylops_mpi.basicoperators": [[3, 0, 1, "", "MPIBlockDiag"]], "pylops_mpi.optimization.basic": [[4, 1, 1, "", "cgls"]], "pylops_mpi.optimization.cls_basic": [[5, 0, 1, "", "CGLS"]]}, "objtypes": {"0": "py:class", "1": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "function", "Python function"]}, "titleterms": {"pylops_mpi": [0, 1, 2, 3, 4, 5], "distributedarrai": [0, 6], "exampl": [0, 1, 2], "us": [0, 1, 2], "mpilinearoper": 1, "partit": 2, "basicoper": 3, "mpiblockdiag": 3, "optim": [4, 5], "basic": [4, 6], "cgl": [4, 5], "cls_basic": 5, "pylop": 6, "mpi": 6, "api": 6, "linear": 6, "oper": 6, "templat": 6, "solver": 6, "galleri": 7, "distribut": 8, "arrai": 8, "post": 9, "stack": 9, "invers": 9, "3d": 9, "comput": 10, "time": 10, "overview": 11}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx.ext.intersphinx": 1, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"pylops_mpi.DistributedArray": [[0, "pylops-mpi-distributedarray"]], "Examples using pylops_mpi.DistributedArray": [[0, "examples-using-pylops-mpi-distributedarray"]], "pylops_mpi.MPILinearOperator": [[1, "pylops-mpi-mpilinearoperator"]], "Examples using pylops_mpi.MPILinearOperator": [[1, "examples-using-pylops-mpi-mpilinearoperator"]], "pylops_mpi.Partition": [[2, "pylops-mpi-partition"]], "Examples using pylops_mpi.Partition": [[2, "examples-using-pylops-mpi-partition"]], "pylops_mpi.basicoperators.MPIBlockDiag": [[3, "pylops-mpi-basicoperators-mpiblockdiag"]], "pylops_mpi.optimization.basic.cgls": [[4, "pylops-mpi-optimization-basic-cgls"]], "pylops_mpi.optimization.cls_basic.CGLS": [[5, "pylops-mpi-optimization-cls-basic-cgls"]], "PyLops MPI API": [[6, "pylops-mpi-api"]], "DistributedArray": [[6, "distributedarray"]], "Linear operators": [[6, "linear-operators"]], "Templates": [[6, "templates"]], "Basic Operators": [[6, "basic-operators"]], "Solvers": [[6, "solvers"]], "Basic": [[6, "basic"]], "Gallery": [[7, "gallery"]], "Distributed Array": [[8, "distributed-array"]], "Post Stack Inversion - 3D": [[9, "post-stack-inversion-3d"]], "Computation times": [[10, "computation-times"]], "Overview": [[11, "overview"]]}, "indexentries": {"distributedarray (class in pylops_mpi)": [[0, "pylops_mpi.DistributedArray"]], "mpilinearoperator (class in pylops_mpi)": [[1, "pylops_mpi.MPILinearOperator"]], "partition (class in pylops_mpi)": [[2, "pylops_mpi.Partition"]], "mpiblockdiag (class in pylops_mpi.basicoperators)": [[3, "pylops_mpi.basicoperators.MPIBlockDiag"]], "cgls() (in module pylops_mpi.optimization.basic)": [[4, "pylops_mpi.optimization.basic.cgls"]], "cgls (class in pylops_mpi.optimization.cls_basic)": [[5, "pylops_mpi.optimization.cls_basic.CGLS"]]}})