import os
import sys
import shutil
import re

path_list = [
    'build',
    'dist',
]

for path in path_list:
    if os.path.exists(path):
        shutil.rmtree(path)


def get_version():
    filename = "localaplace/__init__.py"
    with open(filename) as f:
        match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""", f.read(),
                          re.M)
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0] 
    return version


def main():
    version = get_version()
    from distutils.core import setup
    from setuptools import find_packages

    setup(
        name = 'localaplace',
        version = version,
        description = 'implement of local laplace filter algorithm',
        long_description='I found there is almost no package can be used to solve my homework, so I make one :D',
        author = 'lstm-kirigaya',
        author_email = '1193466151@qq.com',
        url = 'https://github.com/LSTM-Kirigaya',
        license = 'Apache 2.0',
        install_requires = [
            'numpy',
            'opencv-python',
            'tqdm'
        ],
        classifiers = [
            'Intended Audience :: Developers',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Utilities'
        ],
        keywords = [
            'local laplace filter',
            'local-laplace-filter',
            'computer vision',
            'tone mapping',
            'image process'
        ],
        packages = find_packages(),
        package_data={
            "localaplace": [
                "main.py",
                "images/*",
                "results/*"
            ]
        },
        include_package_data = False,
    )

if __name__ == "__main__":
    main()