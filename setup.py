import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wind_power_prediction",
    version="0.1.0",
    description="kaggle competition",
    authors=["Farzad Roozitalab <farzadroozitalab@cmail.carleton.ca>"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Farzad-R/wind_power_kaggle",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: linux",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8.5",
)
