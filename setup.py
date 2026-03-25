from setuptools import setup, find_packages

setup(
    name='gerberformer',
    version='1.0.0',
    description='Design-Conditioned PCB Defect Detection via Synthetic Gerber Priors',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'ultralytics>=8.0.0',
        'timm>=0.9.12',
        'albumentations==1.3.1',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'pyyaml>=6.0',
        'matplotlib>=3.7.0',
    ],
)
