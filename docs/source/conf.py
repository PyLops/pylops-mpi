# -*- coding: utf-8 -*-
import sys
import os
import datetime

# Sphinx needs to be able to import the package to use autodoc and get the version number
sys.path.insert(0, os.path.abspath("../../pylops_mpi"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "nbsphinx",
    "sphinx_gallery.gen_gallery",
]

# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable/", None),
    "pylops": ("https://pylops.readthedocs.io/en/stable/", None)
}

# Generate autodoc stubs with summaries from code
autosummary_generate = True

# Include Python objects as they appear in source files
autodoc_member_order = "bysource"

# Default flags used by autodoc directives
autodoc_default_flags = ["members"]

# Avoid showing typing annotations in doc
autodoc_typehints = "none"

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [
        "../../examples",
        "../../tutorials",
    ],
    # path where to save gallery generated examples
    "gallery_dirs": ["gallery", "tutorials"],
    "filename_pattern": r"\.py",
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": "ExampleTitleSortKey",
    # directory where function granular galleries are stored
    "backreferences_dir": "api/generated/backreferences",
    # Modules for which function level galleries are created.
    "doc_module": "pylops_mpi",
    # Insert links to documentation of objects in the examples
    "reference_url": {"pylops-mpi": None},
}

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ["png"]

# Sphinx project configuration
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "**.ipynb", "**.md5"]
source_suffix = ".rst"


# The encoding of source files.
source_encoding = "utf-8-sig"
master_doc = "index"

# General information about the project
year = datetime.date.today().year
project = "Pylops"
copyright = "{}, PyLops Development Team".format(year)

# # Version
# version = __version__
# if len(version.split("+")) > 1 or version == "unknown":
#     version = "main"

# These enable substitutions using |variable| in the rst files
rst_epilog = """
.. |year| replace:: {year}
""".format(
    year=year
)
html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"
html_title = "PyLops-MPI"
html_short_title = "PyLops-MPI"
html_logo = "_static/pylopsmpi.png"
html_favicon = "_static/favicon.ico"
html_extra_path = []
pygments_style = "default"
add_function_parentheses = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# Theme config
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "image_light": "pylopsmpi_b.png",
        "image_dark": "pylopsmpi.png",
    }
}
html_css_files = [
    'css/custom.css',
]

html_context = {
    "menu_links_name": "Repository",
    "menu_links": [
        (
            '<i class="fa fa-github fa-fw"></i> Source Code',
            "https://github.com/PyLops/pylops-mpi",
        ),
    ],
    # Custom variables to enable "Improve this page"" and "Download notebook"
    # links
    "doc_path": "docs/source",
    "galleries": sphinx_gallery_conf["gallery_dirs"],
    "gallery_dir": dict(
        zip(sphinx_gallery_conf["gallery_dirs"], sphinx_gallery_conf["examples_dirs"])
    ),
    "github_project": "PyLops",
    "github_repo": "pylops-mpi",
    "github_version": "main",
}


# Load the custom CSS files (needs sphinx >= 1.6 for this to work)
def setup(app):
    app.add_css_file("style.css")
