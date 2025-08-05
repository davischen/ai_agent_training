from setuptools import setup, find_packages

setup(
    name="atlas-agent",
    version="1.0.0", 
    description="ATLAS - Adaptive Training and Learning Automation System",
    author="Your Name",
    author_email="your.email@domain.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "sentence-transformers>=2.2.0", 
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "nltk>=3.6.0",
        "schedule>=1.1.0",
        "requests>=2.25.0",
        "datasets>=2.0.0"
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'atlas-agent=atlas.run_agent:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ]
)
