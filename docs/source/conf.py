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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle
from pybtex.plugin import register_plugin
from collections import Counter


# -- Project information -----------------------------------------------------

project = 'pytorchltr'
copyright = '2020, Rolf Jagerman'
author = 'Rolf Jagerman'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinxcontrib.bibtex',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# -- Extension configuration -------------------------------------------------

class AuthorYearLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        # First generate processed entries
        processed_entries = []
        for entry in sorted_entries:
            year = ""
            if "year" in entry.fields:
                year = entry.fields["year"]
                if len(year) == 4 and year.startswith("20"):
                    year = year[-2:]
            authors = ""
            author_limit = 3
            for author in entry.persons["author"][:author_limit]:
                authors += author.last_names[0][0]
            if len(entry.persons["author"]) > author_limit:
                authors += "+"
            processed_entries.append("%s%s" % (authors, year))

        # Mark duplicates with incremental alphabet
        counts = Counter(processed_entries)
        marked = Counter()
        out_entries = []
        for entry in processed_entries:
            if counts[entry] > 1:
                marked[entry] += 1
                entry += _number_to_alphabet(marked[entry] - 1)
            out_entries.append(entry)

        return out_entries


def _number_to_alphabet(number):
    out = chr((number % 26) + 97)
    if number >= 26:
        out = _number_to_alphabet(number // 26 - 1) + out
    return out


class AuthorYearStyle(UnsrtStyle):
    default_label_style = AuthorYearLabelStyle


register_plugin('pybtex.style.formatting', 'authoryearstyle', AuthorYearStyle)
