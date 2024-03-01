from invoke.context import Context
from invoke.tasks import task


@task
def install(ctx: Context) -> None:
    ctx.run("poetry install --extras minio --extras setfit --with test,dev --no-root", echo=True)


@task
def setup(ctx: Context) -> None:
    print("Installing package dependencies")
    install(ctx)

    print("Setting up pre-commit hooks...")
    ctx.run("poetry run pre-commit install --hook-type pre-push", echo=True)


@task
def test(ctx: Context) -> None:
    ctx.run("poetry run pytest", echo=True)


@task
def type_check(ctx: Context) -> None:
    ctx.run("poetry run mypy --package dataquality --namespace-packages", echo=True)


@task
def docs_build(ctx: Context) -> None:
    ctx.run("poetry run sphinx-apidoc -f -o docs/source dataquality/", echo=True)
    ctx.run("poetry run sphinx-build -M markdown docs/source docs/build/md", echo=True)
    ctx.run("poetry run sphinx-build -b html docs/source/ docs/build/html", echo=True)
