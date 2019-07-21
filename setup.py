import setuptools

setuptools.setup(
    name="tensorjo",
    version="0.0.1",
    author="Jonas Valfridsson",
    author_email="jonas@valfridsson.net",
    description="A tiny tensor library made for differentiable things",
    url="",
    packages=setuptools.find_packages(exclude=[]),
    install_requires=["numpy==1.16.4"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
