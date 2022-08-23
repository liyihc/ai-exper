import importlib
from pathlib import Path
from ai_exper.procedure import Procedure
from . import base
import typer


def main():
    # cwd = Path.cwd()

    path = Path("models.py")
    assert path.exists()

    # models = importlib.import_module("models", "models.py")
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("models", path.absolute())
    models = importlib.util.module_from_spec(spec)
    sys.modules["models"] = models
    spec.loader.exec_module(models)

    assert hasattr(models, "procedure")

    procedure: Procedure = getattr(models, "procedure")

    base.datasets = next(iter(procedure.steps.values())).keys()
    base.models = procedure.models.keys()


    app = typer.Typer(no_args_is_help=True)

    from . import gather, run

    app.command()(run.run)
    app.add_typer(gather.app, name="gather")

    app()
