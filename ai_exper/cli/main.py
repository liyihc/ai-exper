from pathlib import Path
from . import common
import typer


def main():
    # cwd = Path.cwd()

    # models = importlib.import_module("models", "models.py")
    import importlib
    import importlib.util
    import sys
    sys.path.append(str(Path.cwd()))

    import models

    assert hasattr(models, "procedure")

    common.set_module(models)
    procedure = common.get_procedure()

    common.datasets.update(next(iter(procedure.steps.values())).keys())
    common.models.update(procedure.models.keys())

    app = typer.Typer(no_args_is_help=True)

    from . import run, gather_config, gather_custom

    app.command("run")(run.run)
    app.command("gather-config", no_args_is_help=True)(
        gather_config.gather_config)
    app.command("gather", no_args_is_help=True)(
        gather_custom.gather_custom)

    app()
