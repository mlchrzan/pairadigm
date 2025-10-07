"""
Setup script for Pairadigm package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pairadigm",
    version="0.1.0",
    author="Michael Leon Chrzan",
    author_email="mlchrzan1@gmail.com",
    description="Concept-Guided Chain-of-Thought prompting with Alternative Annotator Test",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlchrzan/pairadigm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "choix>=0.3.5",
        "plotly>=5.0.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "python-dotenv>=0.19.0",
        "google-genai>=0.1.0",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "pairadigm=pairadigm.cli:main",
        ],
    },
)