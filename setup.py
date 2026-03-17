# noinspection Mypy
from typing import Any

from setuptools import setup, find_packages
from os import path, getcwd

# from https://packaging.python.org/tutorials/packaging-projects/

# noinspection SpellCheckingInspection
package_name = "languagemodelcommon"

with open("README.md", "r") as fh:
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
        "boto3>=1.34.59",
        "fastapi>=0.110.0",
        "httpx[http2]>=0.27.2",
        "langchain>=1.0.0",
        "langchain-ai-skills-framework>=1.0.2",
        "langchain-aws>=1.0.0",
        "langchain-community>=0.4",
        "langchain-core>=1.0.0",
        "langgraph>=1.0.0",
        "loguru>=0.7.3",
        "oidcauthlib>=2.0.11",
        "openai>=2.5.0",
        "pydantic<3.0.0,>=2.0",
        "tiktoken>=0.7.0",
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
