# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="Paul",
    description="A package with data related helper functions.",
    name="pytoolkit",
    packages=find_packages(include=["pytoolkit", "pytoolkit.*"]),
    install_requires=['numpy>=1.22', 'pandas>=1.5'],
    python_requires='>=3.8.10',
    version="0.1.0",
)
