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
          'numpy==1.21.5',
          'pandas==1.1.5',
          'scikit-sparse==0.4.12',
          'nibabel==3.2.1',
          'scipy==1.5.2',
          'sklearn==0.23.2',
          'bitarray==2.6.0'  
      ])

