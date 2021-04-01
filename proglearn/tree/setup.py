import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("proglearn", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    config.add_extension("split",
                         sources=["split.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"],
                        #  extra_compile_args=["-Xpreprocessor", "-fopenmp",],
                        #  extra_link_args=["-Xpreprocessor", "-fopenmp"],
                         language="c++"
            )

    # config.add_subpackage("tests")
    # config.add_data_files("split.pxd")
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
