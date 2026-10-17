from setuptools import setup

setup(
    name="radeon-sparse",
    version="2.0.0",
    description="High-performance OpenCL GPU compute engine for sparse LLM inference on AMD Radeon GPUs.",
    packages=["radeon_sparse"],
    package_dir={"radeon_sparse": "python"},
    package_data={"radeon_sparse": ["*.dll", "*.so"]},
    install_requires=["numpy"],
    python_requires=">=3.6",
)
