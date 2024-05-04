from setuptools import setup

setup(name='heig',
      version='1.0.0',
      description='Highly Efficient Imaging Genetics (HEIG)',
      url='https://github.com/Zhiwen-Owen-Jiang/HEIG',
      author='Zhiwen Jiang and Hongtu Zhu',
      author_email='zhiwenowenjiang@gmail.com',
      license='GPLv3',
      packages=['heig'],
      scripts=['heig.py'],
      install_requires=[
          'numpy==1.26.4',
          'pandas==2.2.2',
          'nibabel==5.2.1',
          'scipy==1.11.4',
          'sklearn==1.4.2',
          'bitarray==2.9.2'  
      ])

