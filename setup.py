from setuptools import setup, find_packages

setup(
    name="twitter_sentiment_classification",
    version="0.1.0",
    author="Your Name",
    author_email="vijaytakbhate20@gmail.com",
    description="An end-to-end machine learning project for Twitter sentiment classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/twitter_sentiment_classification",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "nltk",
        "regex",
        "joblib",
        "PySpark",
        "mlflow",
        "matplotlib",
        "seaborn",
    ],
    entry_points={
        "console_scripts": [
            "train_model=src.train_pipe:main",
            "predict_model=src.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        # Include any additional files such as configuration files or metadata
        "": ["*.csv", "*.pkl", "*.json"],
    },
)
