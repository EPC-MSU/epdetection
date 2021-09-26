from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='epdetection',
      version='1.0.5',
      description='PCB components detection module for EyePoint P10.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/EPC-MSU/epdetection',
      author='EPC MSU',
      author_email='a.p.marakulin@gmail.com',
      license='CC0-1.0',
      packages=['detection'],
      install_requires=[
            'numpy==1.18.1',
            'opencv-python==3.3.0.10',
            'scikit-image==0.16.2',
            'scikit-learn==0.20.1',
            'scipy==1.5.4',
            'hg+https://anonymous:anonymous@hg.ximc.ru/eyepoint/epcore#egg=epcore'
      ],
      package_data={"detection": ["dumps/*"]},
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: CC0 License",
            "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      zip_safe=False)
