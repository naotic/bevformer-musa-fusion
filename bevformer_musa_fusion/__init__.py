import importlib.metadata
from pathlib import Path
import torch
import torch_musa
for name in ('torch', 'torch_musa'):
    if importlib.metadata.version(name) != '2.7.1.post1':
        raise RuntimeError('fusion requires pinned post1 ' + name)
if torch_musa.__version__ != '2.7.1.post1+e5c8cd4':
    raise RuntimeError('fusion requires pinned torch_musa build')
files = list(Path(__file__).parent.glob('_native*.so'))
if len(files) != 1:
    raise RuntimeError('expected one newly compiled fusion library')
torch.ops.load_library(str(files[0]))
if not torch._C._dispatch_has_kernel_for_dispatch_key('musa_fused::conv2d_bias_act_add', 'PrivateUse1'):
    raise RuntimeError('selected convolution operator is not registered')
