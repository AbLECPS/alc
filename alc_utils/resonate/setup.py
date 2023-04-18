from setuptools import setup

setup(
    name='resonate',
    version='0.1.0',
    author='Charles Hartsell',
    author_email='hartsellcharles@gmail.com',
    packages=['resonate'],
    scripts=[],
    # url='http://pypi.python.org/pypi/PackageName/',
    license='LICENSE.md',
    description='Python implementation of the ReSonAte framework.',
    long_description=open('README.md').read(),
    install_requires=[
        "tqdm >= 4.56.0",
        "numpy >= 1.16",
        "pandas >= 0.24"
    ],
    python_requires='>=2.6, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
)
