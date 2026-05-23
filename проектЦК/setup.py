from setuptools import setup, find_packages

setup(
    name="gesture-moderation",
    version="1.0.0",
    description="Automatic gesture recognition and blocking system for video streams",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
