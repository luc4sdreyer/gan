from setuptools import setup, find_packages


tests_require = [
    'pytest',
]

setup(
    name='gan',
    tests_require=tests_require,
    packages=find_packages()
)
