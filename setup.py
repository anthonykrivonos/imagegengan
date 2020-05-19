import setuptools

# Use README as the long_description
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imagegengan",
    version="1.0.0",
    author="Anthony Krivonos",
    author_email="info@anthonykrivonos.com",
    description="A plug-and-play GAN image generator.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anthonykrivonos/imagegengan",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'opencv-contrib-python',
        'tensorflow',
        'Keras'
    ],
    python_requires='>=3.6',
)