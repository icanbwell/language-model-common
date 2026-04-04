# noinspection Mypy
from typing import Any

from setuptools import setup, find_packages
from os import path, getcwd

# from https://packaging.python.org/tutorials/packaging-projects/

# noinspection SpellCheckingInspection
package_name = "language-model-common"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

try:
    with open(path.join(getcwd(), "VERSION")) as version_file:
        version = version_file.read().strip()
except IOError:
    raise


def fix_setuptools() -> None:
    """Work around bugs in setuptools.

    Some versions of setuptools are broken and raise SandboxViolation for normal
    operations in a virtualenv. We therefore disable the sandbox to avoid these
    issues.
    """
    try:
        from setuptools.sandbox import DirectorySandbox

        # noinspection PyUnusedLocal
        def violation(operation: Any, *args: Any, **_: Any) -> None:
            print("SandboxViolation: %s" % (args,))

        DirectorySandbox._violation = violation
    except ImportError:
        pass


# Fix bugs in setuptools.
fix_setuptools()


# classifiers list is here: https://pypi.org/classifiers/

# create the package setup
setup(
    name=package_name,
    version=version,
    author="Imran",
    author_email="imran.qureshi@bwell.com",
    description="Provides the underlying framework to enhance langchain and add loading configurations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/icanbwell/language-model-common",
    packages=find_packages(),
    install_requires=[
        "boto3>=1.40.21",
        "fastapi>=0.115.8",
        "httpx[http2]>=0.27.2",
        "httpx-sse>=0.4.0",
        "langchain>=1.0.0",
        "langchain-aws>=1.2.0",
        "langchain-community>=0.4",
        "openai>=2.5.0",
        "langchain-openai>=1.1.6",
        "langchain-core>=1.2.5",
        "langgraph>=1.0.0",
        "pydantic>=2.0,<3.0.0",
        "mcp>=1.11.0",
        "langchain-mcp-adapters>=0.2.1",
        "langmem>=0.0.30",
        "langgraph-checkpoint>=3.0.0",
        "langgraph-checkpoint-mongodb>=0.2.1",
        "langgraph-store-mongodb>=0.1.0",
        "tiktoken>=0.12.0",
        "langchain-ai-skills-framework>=1.0.23",
        "simple-container>=1.0.2",
        "pypdf>=6.6.0",
        "markdownify>=0.14.1",
        "beautifulsoup4>=4.12.3",
        "oidcauthlib>=3.0.3",
        "fsspec>=2026.2.0",
        "authlib>=1.6.5",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    dependency_links=[],
    include_package_data=True,
    zip_safe=False,
    package_data={"languagemodelcommon": ["py.typed"]},
)
