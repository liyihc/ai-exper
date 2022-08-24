from pathlib import Path
import readline
from typing import List, Optional
import typer
from . import common


def run(
    models: Optional[List[str]] = typer.Argument(
        None,
        help=f"choose from: {', '.join(common.models)}"),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        is_flag=True,
        help="start direct without interactive"),
    file: Optional[Path] = typer.Option(
        None,
        "-f", "--file",
        exists=True, dir_okay=False, readable=True, resolve_path=True
    )
):
    procedure = common.get_procedure()
    if file is not None:
        param = procedure.parse_multi_model_file(Path(file))
    else:
        param = procedure.construct_params()

    procedure.add_models_to_param(param, models)

    if not confirm:
        readline.parse_and_bind("tab: complete")
        while True:
            procedure.print(param)
            readline.set_completer(procedure.get_completer(param))
            command = input("please input command: ").strip()
            if command.startswith("save-to:"):
                path = Path(command.removeprefix("save-to:"))
                if path.exists():
                    confirm = input("override?")
                    if not confirm.lower().startswith('y'):
                        continue
                with path.open('w') as f:
                    f.write(param.json(indent=4))
                print("write to file", path)
            if command == "exit":
                exit()
            if command == "start":
                break
            if "=" in command:
                k, v, *_ = command.split('=')
                try:
                    procedure.update(param, k.strip(), v.strip())
                except Exception as e:
                    print(str(e))
    else:
        procedure.print(param)
    procedure.run(param)
