from setuptools import find_packages, setup

setup(
    name="tcco2-accuracy",
    version="0.0.0",
    description="Python port of TcCO2 accuracy meta-analysis and simulation",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
