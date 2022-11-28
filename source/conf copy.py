import pytorch_sphinx_theme

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Galileo Dataquality"
copyright = "2022, Galileo"
author = "Galileo"
release = "0.8"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.doctest",
    # "sphinx.ext.intersphinx",
    # "sphinx.ext.todo",
    # "sphinx.ext.coverage",
    # "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # "sphinxcontrib.katex",
    # "sphinx.ext.autosectionlabel",
    # "sphinx_copybutton",
    # "sphinx_panels",
    # "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


# copied from pytorch
# build the templated autosummary files
autosummary_generate = True
numpydoc_show_class_members = False

# Theme has bootstrap already
panels_add_bootstrap_css = False

# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = True

# katex options
#
#

# katex_prerender = True

napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]


html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "pytorch_project": "docs",
    "canonical_url": "https://pytorch.org/docs/stable/",
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "analytics_id": "UA-117752657-2",
}
