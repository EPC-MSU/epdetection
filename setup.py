from setuptools import find_packages, setup


setup(name="epdetection",
      version="1.0.12",
      description="PCB components detection module",
      url="https://github.com/EPC-MSU/epdetection",
      author="EPC MSU",
      author_email="a.p.marakulin@gmail.com",
      license="CC0-1.0",
      packages=find_packages(),
      python_requires="~=3.6.0",
      install_requires=[
          "numpy==1.18.1",
          "opencv-python",
          "scikit-image==0.16.2",
          "scikit-learn==0.20.1",
          "scipy==1.5.4",
          # "epcore @ git+https://github.com/EPC-MSU/epcore#egg=epcore",
      ],
      package_data={"detection": ["dumps/*"]},
      include_package_data=True,
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: CC0 License",
            "Operating System :: OS Independent",
      ],
      zip_safe=False)
