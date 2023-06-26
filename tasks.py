from enum import Enum, unique
from typing import Optional

from invoke import task
from invoke.context import Context

PACKAGE_NAME = "dataquality"
VERSION_FILE = f"{PACKAGE_NAME}/__init__.py"
SOURCES = " ".join(["dataquality", "tests", "tasks.py"])


@task
def clean(ctx: Context) -> None:
    """clean

    Remove all build, test, coverage and Python artifacts.
    """
    ctx.run(
        " ".join(
            [
                "rm -rf",
                ".coverage",
                ".ipynb_checkpoints",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
                "*.egg-info",
                "*.egg",
                "build",
                "coverage.xml",
                "dist",
                "htmlcov",
                "site",
            ]
        ),
        pty=True,
        echo=True,
    )


@task
def install(ctx: Context, extras: Optional[str] = None, editable: bool = True) -> None:
    """install

    Install dependencies.

    Args:
        ctx (Context): The invoke context.
        extras (str, optional): The extras to install. Defaults to None. If None,
            then the default extras are installed. Specify as a comma-separated
            string.
        editable (bool, optional): Whether to install in editable mode. Defaults to
            True.
    """
    if extras is None:
        extras = "dev,test,doc"
    ctx.run(
        "pip install --upgrade pip",
        pty=True,
        echo=True,
    )
    if editable:
        cmd = f"pip install -e '.[{extras}]'"
    else:
        cmd = f"pip install '.[{extras}]'"
    ctx.run(
        cmd,
        pty=True,
        echo=True,
    )


@task
def lint(ctx: Context) -> None:
    """lint

    Check typing and formatting.
    """
    ctx.run(
        "mypy dataquality tasks.py",
        pty=True,
        echo=True,
    )
    ctx.run(
        f"black {SOURCES} --check",
        pty=True,
        echo=True,
    )
    ctx.run(
        f"ruff {SOURCES}",
        pty=True,
        echo=True,
    )


@task
def format(ctx: Context) -> None:
    """format

    Format the code.
    """
    ctx.run(
        f"black {SOURCES}",
        pty=True,
        echo=True,
    )
    ctx.run(
        f"ruff {SOURCES} --fix",
        pty=True,
        echo=True,
    )


@task
def test(ctx: Context) -> None:
    """test

    Run the tests.
    """
    ctx.run(
        "pytest",
        pty=True,
        echo=True,
    )


@task
def build(ctx: Context) -> None:
    """build

    Build the package.
    """
    ctx.run(
        "pip install --upgrade build",
        pty=True,
        echo=True,
    )
    ctx.run(
        "python -m build",
        pty=True,
        echo=True,
    )


@task
def publish(ctx: Context) -> None:
    """publish

    Publish the package.
    """
    ctx.run(
        "pip install --upgrade twine",
        pty=True,
        echo=True,
    )
    ctx.run(
        "twine upload dist/*",
        pty=True,
        echo=True,
    )


@task
def all(ctx: Context) -> None:
    """all

    Run all the tasks that matter for local dev.
    """
    clean(ctx)
    install(ctx)
    format(ctx)
    lint(ctx)
    test(ctx)


@task
def docs_build(ctx: Context) -> None:
    """docs-build

    Build the docs.
    """
    with ctx.cd("docs/autodocs"):
        ctx.run(
            "make markdown",
            pty=True,
            echo=True,
        )


@unique
class BumpType(Enum):
    MAJOR = "major"
    MINOR = "minor"


def _bump_version(version: str, bump: Optional[BumpType] = None) -> str:
    """Bump a version string.

    Args:
        version (str): The version string to bump.
        bump (str): The type of bump to perform.

    Returns:
        str: The bumped version string.
    """
    from packaging.version import Version

    v = Version(version)
    if bump == BumpType.MAJOR:
        v = Version(f"{v.major + 1}.0.0")
    elif bump == BumpType.MINOR:
        v = Version(f"{v.major}.{v.minor + 1}.0")
    else:
        v = Version(f"{v.major}.{v.minor}.{v.micro + 1}")
    return str(v)


@task(aliases=["uv"])
def update_version_number(ctx: Context, part: Optional[BumpType] = None) -> None:
    """update version number

    Specify the part of the version number to bump. The default is to bump the
    micro version number. Other options are major and minor.
    """
    from dataquality import __version__

    print(f"Current version: {__version__}")
    new_version = _bump_version(__version__, part)
    with open(VERSION_FILE, "r") as f:
        lines = f.readlines()
    with open(VERSION_FILE, "w") as f:
        for line in lines:
            if line.startswith("__version__"):
                f.write(f'__version__ = "v{new_version}"\n')
            else:
                f.write(line)
    print(f"New version: {new_version}")
