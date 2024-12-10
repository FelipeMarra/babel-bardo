from pathlib import Path
from setuptools import find_packages, setup

with open("README.md", 'r') as file:
    long_description = file.read()

HERE = Path(__file__).parent

REQUIRED = [i.strip() for i in open(HERE / 'requirements.txt') if not i.startswith('#')]

setup(
    name="babel_bardo",
    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/guides/single-sourcing-package-version/
    version="0.0.1",
    description="Babel-Bardo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FelipeMarra/babel-bardo",
    author="Felipe Ferreira Marra et al",
    author_email="felipeferreiramarra@gmail.com",
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="generative, ai, model, music",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8.0",
    install_requires=REQUIRED,
    extras_require={
        "dev": ['pytest'],
    },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/FelipeMarra/babel-bardo/issues",
    #    "Funding": "https://donate.pypi.org",
    #    "Say Thanks!": "http://saythanks.io/to/example",
        "Source": "https://github.com/FelipeMarra/babel-bardo",
    },
)