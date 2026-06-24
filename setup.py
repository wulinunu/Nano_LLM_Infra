import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")


setup(
    name="nano-llm-infra",
    version="0.1.0",
    package_dir={"": "src"},
    packages=["nano_llm_infra", "nano_llm_infra.ops"],
    ext_modules=[
        CUDAExtension(
            name="nano_llm_infra._C",
            sources=[str(ROOT / "csrc" / "rmsnorm" / "rms_norm.cu")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
