import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="niqe", # Replace with your own username
    version="0.0.1",
    author="Dino Vougioukas",
    author_email="tech@synthy.ai",
    description="Package which implements the Natural Image Quality Evaluator (NIQE)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SynthyAI/niqe",
    packages=setuptools.find_packages(),
    package_data={'niqe': ['resources/*.mat']}, 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-video',
        'opencv-python',
        ],
    python_requires='>=3.6',
)
