from setuptools import setup

setup(
    name='pse',
    version='0.0.0',
    description='Pairwise sequence embedding utilities',
    author='Colin Raffel',
    author_email='craffel@gmail.com',
    url='https://github.com/craffel/pse',
    packages=['pse'],
    long_description="""\
    Pairwise sequence embedding utilities.
    """,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords='machine learning',
    license='MIT',
    install_requires=[
        'numpy >= 1.7.0',
    ],
)
