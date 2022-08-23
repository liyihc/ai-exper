import typer
from . import base

app = typer.Typer(no_args_is_help=True)


@app.command("result")
def gather_result(
        dataset: str = typer.Argument(
            ...,
            help=f"choice from {','.join(base.datasets)}"),

):
    print("gather result")


@app.command("custom")
def gather_custom(

):
    print("gather custom")
