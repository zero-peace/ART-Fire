"""
Art-Fire: Advanced Temporal Anomaly Detection for Wildfire Monitoring

A comprehensive Python package for detecting temporal anomalies in environmental data,
specifically designed for wildfire monitoring and early warning systems.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def parse_requirements(filename):
    """Parse requirements from requirements.txt file."""
    with open(filename, 'r') as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and lines starting with -
            if line and not line.startswith('#') and not line.startswith('-'):
                requirements.append(line)
        return requirements

# Parse main requirements
requirements = parse_requirements('requirements.txt')

setup(
    name='art-fire',
    version='0.1.0',
    author='Art-Fire Team',
    author_email='contact@art-fire.org',
    description='Advanced Temporal Anomaly Detection for Wildfire Monitoring',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/art-fire/art-fire',
    packages=find_packages(exclude=['tests*', 'examples*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'flake8>=4.0',
            'mypy>=0.900',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
            'sphinxcontrib-napoleon>=0.7',
        ],
        'optional': [
            'plotly>=5.0',
            'dash>=2.0',
            'jupyter>=1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'art-fire=core.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords='anomaly detection, time series, wildfire monitoring, temporal analysis, environmental data',
    project_urls={
        'Bug Reports': 'https://github.com/art-fire/art-fire/issues',
        'Source': 'https://github.com/art-fire/art-fire',
        'Documentation': 'https://art-fire.readthedocs.io/',
    },
)