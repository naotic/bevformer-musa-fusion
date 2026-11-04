import importlib.metadata
from pathlib import Path

from setuptools import setup
import torch_musa
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

for package in ('torch', 'torch_musa'):
    if importlib.metadata.version(package) != '2.7.1.post1':
        raise RuntimeError('This extension requires ' + package + ' 2.7.1.post1')

header_root = Path(torch_musa.__file__).resolve().parent.parent
if not (header_root / 'torch_musa/csrc/aten/utils/Context.h').is_file():
    raise RuntimeError('The installed torch_musa package is missing C++ headers')

extension = MUSAExtension(
    'bevformer_musa_fusion._native',
    ['fusion.cpp'],
    include_dirs=[str(header_root)],
    libraries=['mudnn'],
    extra_compile_args={'cxx': ['-O2', '-std=c++17']},
)
extension.extra_compile_args['cxx'] = [
    flag for flag in extension.extra_compile_args['cxx']
    if flag != '-march=native'
] + ['-march=x86-64']

setup(
    name='bevformer_musa_fusion',
    version='0.1.0',
    packages=['bevformer_musa_fusion'],
    ext_modules=[extension],
    cmdclass={'build_ext': BuildExtension},
    license_files=['LICENSE.torch_musa'],
)
