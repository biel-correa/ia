import pip
import numpy
import jupyter
import matplotlib
import sklearn
import scipy
import pandas
import PIL
import seaborn
import h5py
import tensorflow
import keras


def check_version(pkg, version):
    actual = pkg.__version__.split('.')
    if len(actual) == 3:
        actual_major = '.'.join(actual[:2])
    elif len(actual) == 2:
        actual_major = '.'.join(actual)
    else:
        raise NotImplementedError(pkg.__name__ +
                                  "actual version :"+
                                  pkg.__version__)
    try:
        assert(actual_major == version)
    except Exception as ex:
        print("{} {}\t=> {}".format(pkg.__name__,
                                    version,
                                    pkg.__version__))
        raise ex

check_version(pip, '19.1')
check_version(numpy, '1.16')
check_version(matplotlib, '3.1')
check_version(sklearn, '0.21')
check_version(scipy, '1.2')
check_version(pandas, '0.24')
check_version(PIL, '6.1')
check_version(seaborn, '0.9')
check_version(h5py, '2.9')
check_version(tensorflow, '1.13')
check_version(keras, '2.2')

print("Houston we are go!")