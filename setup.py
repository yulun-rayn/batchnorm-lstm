import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="batchnorm-lstm",
    version="0.0.1",
    author="Yulun Rayn Wu",
    author_email="yulun_wu@berkeley.edu",
    description="A package that implements Many-to-One Long Short-Term Memory with batch normalization, dropout and layer stacking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yulun-rayn/batchnorm-lstm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


