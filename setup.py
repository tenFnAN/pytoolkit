# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="Paul",
    description="A package with data related helper functions.",
    name="pytoolkit",
    packages=find_packages(include=["pytoolkit", "pytoolkit.*"]),
    install_requires=['numpy>=1.22', 'pandas>=1.5', 'seaborn>=0.12.0', 'scikit-learn>=1.1.0',
                      'matplotlib>=3.4.0', 'plotnine>=0.12.1', 'plotly>=5.15.0', 'feature-engine>=1.5.2'],
    python_requires='>=3.8.10',
    version="0.1.0"
)
