# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys

from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "tensortrax"
year = datetime.date.today().year
copyright = f"2022-{year}, Andreas Dutzler"
author = "Andreas Dutzler"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
]

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["examples"],
    "image_scrapers": ("matplotlib",),
    "download_all_examples": False,
    "remove_config_comments": True,
    "reset_modules_order": "both",
    "filename_pattern": "ex.*\\.py",
    "backreferences_dir": None,
    "pypandoc": True,
    "capture_repr": ("_repr_html_",),
    "within_subsection_order": FileNameSortKey,
}

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "python": ("https://docs.python.org/3/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# matplotlib plot directive configuration options
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ["png"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_title = "tensortrax"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "icon_links": [
        {
            "name": "Discussions",
            "url": "https://github.com/adtzlr/tensortrax/discussions",
            "icon": "fa-solid fa-comment",
            "type": "fontawesome",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/adtzlr/tensortrax",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Read the Docs",
            "url": "https://readthedocs.org/projects/tensortrax",
            "icon": "fa-solid fa-book",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/tensortrax/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "logo": {
        "text": "tensortrax",
        "image_light": "logo_without_text.png",
        "image_dark": "logo_without_text.png",
    },
}
html_context = {
    "github_user": "adtzlr",
    "github_repo": "tensortrax",
    "github_version": "main",
    "doc_path": "docs/",
}
