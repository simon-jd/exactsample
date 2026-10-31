"""Mark wheels containing the compiled Torch operator as platform dependent."""
from pathlib import Path
from setuptools import Distribution, setup


class ExactSampleDistribution(Distribution):
    def has_ext_modules(self):
        return Path(__file__).parent.joinpath("python/exactsample/exactsample_cuda.so").is_file()


setup(distclass=ExactSampleDistribution)
