"""
Setup script for llmrelic package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llmrelic",
    version="0.1.0",
    author="OVECJOE",
    author_email="vohachor@gmail.com",
    description="A lightweight library for LLM model names and support "
    "definitions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ovecjoe/llmrelic",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
    },
    keywords="llm, ai, models, registry, openai, anthropic, google, cohere, "
    "mistral, meta, huggingface",
    project_urls={
        "Bug Reports": "https://github.com/ovecjoe/llmrelic/issues",
        "Source": "https://github.com/ovecjoe/llmrelic",
    },
)
