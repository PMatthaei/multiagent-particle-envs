from setuptools import setup, find_packages

setup(name='multiagent-rts',
      version='0.0.1',
      description='Multi-Agent RTS Environment',
      url='https://github.com/PMatthaei/ma-env',
      author='Patrick Matthaei',
      author_email='pmd.matthaei@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl', 'pygame']
)
