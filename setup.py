import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'PySubsurface - A bag of useful object to load, manipulate, and visualize subsurface data'

# Setup
setup(
    name='pysubsurface',
    description=descr,
    long_description=open(src('README.md')).read(),
    long_description_content_type='text/markdown',
    keywords=['geoscience', 'subsurface', 'visualization'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    author='mrava',
    author_email='matteo.ravasi@kaust.edu.sa',
    install_requires=['numpy', 'scipy', 'matplotlib','pandas', 'scikit-learn'],


	packages=find_packages(exclude=["pytests"]),
    use_scm_version=dict(
        root=".", relative_to=__file__, write_to=src("pysubsurface/version.py")
    ),
    setup_requires=["pytest-runner", "setuptools_scm"],
    test_suite="pytests",
    tests_require=["pytest"],
    zip_safe=True,
)
