from setuptools import setup, find_packages

setup(
    name="adapts",
    version="0.1.0",
    author="Vidith Iyer",
    author_email="your.email@example.com",
    description="Lightweight Adaptation for Time Series Foundation Models",
    url="https://github.com/YOUR_USERNAME/adapts",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
        "yfinance>=0.2.0",
        "matplotlib>=3.4.0",
    ],
)