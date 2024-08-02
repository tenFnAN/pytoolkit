# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="Paul",
    description="A package with data related helper functions.",
    name="pytoolkit",
    packages=find_packages(include=["pytoolkit", "pytoolkit.*"]),
    version="0.1.0",
)
