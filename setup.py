from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Ensure the compiler can be found by the build system
os.environ["CXX"] = "cl.exe"
os.environ["CC"] = "cl.exe"

setup(
    name='fused_op',
    ext_modules=[
        CUDAExtension(
            name='fused_op',
            sources=['fused_act.cu'],
            extra_compile_args={
                'cxx': ['/O2'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_120,code=sm_120'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)