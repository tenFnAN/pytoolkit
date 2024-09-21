# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="Paul",
    description="A package with data related helper functions.",
    name="pytoolkit",
    packages=find_packages(include=["pytoolkit", "pytoolkit.*"]),
    install_requires=['numpy>=1.22', 'pandas>=1.5', 'seaborn>=0.12.0', 'scikit-learn>=1.1.0', 'matplotlib>=3.4.0'],
    python_requires='>=3.9',
    version="0.1.0",
)
