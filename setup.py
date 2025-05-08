from setuptools import setup, find_packages

setup(
    name='manim4ai',
    version='0.1.0',
    author='Your Name',
    author_email='aadyachinubhai@example.com',
    description='A layout and animation static checker built on top of Manim.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/aadya940/manim4ai',  # Update with your repository URL
    packages=find_packages(),
    install_requires=[
        'manim',  # Add any other dependencies here
        'numpy',
        'shapely',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)