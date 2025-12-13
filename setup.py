"""
Setup configuration for pairadigm package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="pairadigm",
    version="0.5.1",
    author="Michael Leon Chrzan",
    author_email="mlchrzan1@gmail.com",  
    description="Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation using Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlchrzan/pairadigm",
    project_urls={
        "Bug Reports": "https://github.com/mlchrzan/pairadigm/issues",
        "Source": "https://github.com/mlchrzan/pairadigm",
    },
    packages=find_packages(exclude=["tests", "experiments", "data"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "plotly>=5.0.0",
        "networkx>=2.6.0",
        "choix>=0.3.5",
        "python-dotenv>=0.19.0",
        "google-genai>=0.1.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.5.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    keywords="nlp annotation pairwise-comparison llm machine-learning text-analysis",
    include_package_data=True,
    zip_safe=False,
)
