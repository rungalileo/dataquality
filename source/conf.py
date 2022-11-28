import os
import sys

sys.path.insert(0, os.path.abspath("../"))  # Source code dir relative to this file


import pytorch_sphinx_theme

autosummary_generate = True


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
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    # "sphinx.ext.napoleon"
    # "sphinx.ext.doctest",
    # "sphinx.ext.todo",
    # "sphinx.ext.coverage",
    # "sphinx.ext.napoleon",
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

html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
# html_static_path = ["_static"]
# html_copy_source = True
