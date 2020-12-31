import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.system('make -j%d' % os.cpu_count())

# Python interface
setup(
    name='image-transformation',
    version='0.0.0',
    install_requires=['torch'],
    packages=['MYTH'],
    package_dir={'MYTH': './'},
    ext_modules=[
        CUDAExtension(
            name='Warping',
            include_dirs=['./'],
            sources=[
                'pybind/bind.cpp',
            ],
            libraries=['make_pytorch'],
            library_dirs=['objs'],
            # extra_compile_args=['-g']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Khang Truong',
    author_email='khangtg@kaist.ac.kr',
    description='Warping function in multiview stereo',
    keywords='Pytorch C++ Extension',
    url='https://github.com/TruongKhang/depth-completion-seq-net',
    zip_safe=False,
)