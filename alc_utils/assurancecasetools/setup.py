from setuptools import setup, find_packages

setup(
    name='ac_construction',
    version='0.1.0',
    author='Charles Hartsell',
    author_email='hartsellcharles@gmail.com',
    packages=['ac_construction', 'ac_construction.patterns'],
    #packages=find_packages('ac_construction'),
    package_dir={
            'ac_construction': 'ac_construction',
            'ac_construction.patterns': 'ac_construction/patterns'
        },
    include_package_data=True,
    scripts=[],
    # url='http://pypi.python.org/pypi/PackageName/',
    license='LICENSE.md',
    description='Python assurance case construction tools.',
    long_description=open('README.md').read(),
    install_requires=[
        "networkx >= 2.2",
        "numpy >= 1.16.6",
        "matplotlib >= 2.2.5",
        "six >= 1.16.0"
    ],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
)
