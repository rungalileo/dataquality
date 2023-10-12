from enum import Enum, unique
from typing import Optional

from invoke import task
from invoke.context import Context

PACKAGE_NAME = "dataquality"
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
def install(ctx: Context, extras: str = "default") -> None:
    """install

    Install dependencies using Poetry.

    Args:
        ctx (Context): The invoke context.
        extras (str, optional): The extras to install. Defaults to "default".
    """
    ctx.run("poetry install", pty=True, echo=True)
    if extras != "default":
        ctx.run(f"poetry install -E {extras}", pty=True, echo=True)


@task
def lint(ctx: Context) -> None:
    """lint

    Check typing and formatting using mypy and black.
    """
    ctx.run("poetry run mypy dataquality tasks.py", pty=True, echo=True)
    ctx.run("poetry run black --check .", pty=True, echo=True)


@task
def format(ctx: Context) -> None:
    """format

    Format the code using black.
    """
    ctx.run("poetry run black .", pty=True, echo=True)


@task
def test(ctx: Context) -> None:
    """test

    Run the tests using pytest.
    """
    ctx.run("poetry run pytest", pty=True, echo=True)


@task
def build(ctx: Context) -> None:
    """build

    Build the package using Poetry.
    """
    ctx.run("poetry build", pty=True, echo=True)


@task
def publish(ctx: Context) -> None:
    """publish

    Publish the package to PyPI using Poetry.
    """
    ctx.run("poetry publish --build", pty=True, echo=True)

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
