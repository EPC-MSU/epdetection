from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='epdetection',
      version='1.5.0',  # 1.5.x reserved for neural networks
      description='PCB components detection module for EyePoint P10.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/EPC-MSU/epdetection',
      author='EPC MSU',
      author_email='a.p.marakulin@gmail.com',
      license='CC0-1.0',
      packages=['detection'],
      install_requires=[],
      package_data={"detection": ["dumps/*"]},
      include_package_data=True,
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: CC0 License",
            "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      zip_safe=False)
