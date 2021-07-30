from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='MyHelloWorldProject',
      version='0.0.0',
      description='YOUR DESCRIPTION',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/EPC-MSU/URL_TO_YOUR_PROJECT',
      author='EPC MSU',
      author_email='YOUR_EMAIL@physlab.ru',
      license='CC0-1.0',
      packages=['hello_world'],
      install_requires=[
            '',  # YOUR DEPENDENCIES ARE HERE
      ],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: CC0 License",
            "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      zip_safe=False)
