"""
Setup script for Omni-Agent: Universal Generic Agent
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="omni-agent",
    version="0.1.0",
    author="All Things Security",
    author_email="contact@allthingssecurity.dev",
    description="Universal AI agent with configurable Theory of Mind and Decision Theory for any domain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/allthingssecurity/omni-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Other/Nonlisted Topic",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.15.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "full": [
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "nltk>=3.6",
            "pandas>=1.3.0",
            "matplotlib>=3.3.0",
            "pydantic>=1.8.0",
            "pyyaml>=5.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "omni-agent=universal_orchestrator:main",
        ],
    },
    keywords=[
        "ai", "agent", "artificial-intelligence", "theory-of-mind", "decision-theory",
        "universal-ai", "configurable-ai", "multi-domain", "business-ai", "research-ai",
        "creative-ai", "personal-assistant", "machine-learning", "intelligent-agent"
    ],
    project_urls={
        "Bug Reports": "https://github.com/allthingssecurity/omni-agent/issues",
        "Source": "https://github.com/allthingssecurity/omni-agent",
        "Documentation": "https://github.com/allthingssecurity/omni-agent#readme",
    },
)