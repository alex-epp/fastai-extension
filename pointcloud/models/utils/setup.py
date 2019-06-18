from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension('pointnet2_cuda', [
            'cuda_src/pointnet2_api.cpp',
            'cuda_src/ball_query.cpp',
            'cuda_src/ball_query_gpu.cu',
            'cuda_src/group_points.cpp',
            'cuda_src/group_points_gpu.cu',
            'cuda_src/interpolate.cpp',
            'cuda_src/interpolate_gpu.cu',
            'cuda_src/sampling.cpp',
            'cuda_src/sampling_gpu.cu',
        ],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
