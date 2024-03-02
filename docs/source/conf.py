# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime

from dataquality import __version__

project = "dataquality"
copyright = f"{datetime.now().year}, Galileo Technologies Inc."

author = "Galileo Technologies Inc."
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "myst_parser",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_markdown_builder",
    "sphinxcontrib.autodoc_pydantic",
]

templates_path = ["_templates"]

add_module_names = False
autoclass_content = "both"
autodoc_default_flags = ["show-inheritance", "members", "undoc-members"]
autodoc_member_order = "bysource"

# autosummary
autosummary_generate = True

# autosectionlabel
autosectionlabel_prefix_document = True

# autodoc_pydantic
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# Theme options
html_logo = "_static/logo.png"
