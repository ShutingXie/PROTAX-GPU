# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PROTAX-GPU'
copyright = '2025, Roy Li, Sujeevan Ratnasingham, Iuliia Zarubiieva, Panu Somervuo and Graham W. Taylor'
author = 'Roy Li, Sujeevan Ratnasingham, Iuliia Zarubiieva, Panu Somervuo and Graham W. Taylor'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- Extensions --------------------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',      # Automatically document Python modules
    'sphinx.ext.viewcode',     # Link to highlighted source code
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.githubpages',  # GitHub Pages support
    'sphinx.ext.intersphinx',  # Link to other Sphinx documentation
    'myst_parser',             # Support for Markdown files
    'sphinx_rtd_theme',        # Read the Docs theme
]

# Support .md 和 .rst
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_extra_path = ['../source/_static']

# Enable figure numbering
numfig = True

# -- MyST parser configuration -----------------------------------------------
myst_enable_extensions = [
    "deflist",      # Definition lists
    "tasklist",     # Task lists
    "colon_fence",  # Colon fences for code blocks
    "html_image",   # Enable HTML image tags
]

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True   # Support Google style docstrings
napoleon_numpy_docstring = True    # Support NumPy style docstrings
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# -- AutoDoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,           # Include all members of the module
    'undoc-members': True,     # Include undocumented members
    'show-inheritance': True,  # Show class inheritance
}