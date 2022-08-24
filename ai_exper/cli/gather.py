from traceback import format_exc
import typer
from typing import Type
from ai_exper import ResultHandler
from . import common


def do_gather(handler_type: Type[ResultHandler], dataset: str = None, step_name: str = "model"):
    """
    target: step name or "model"
    """
    handler = handler_type()
    procedure = common.get_procedure()
    if dataset:
        dataset_prefix = procedure.steps['dataset'][dataset]().get_prefix() + '*'
    else:
        dataset_prefix = ""
    if step_name == "model":
        pattern = f"[0-9]*{dataset_prefix}/config.json"
    else:
        pattern = f"{dataset_prefix} {step_name}*/config.json"
    errors = []
    for result in procedure.RESULT_PATH.glob(pattern):
        try:
            handler.handle_one(
                result.parent, procedure.parse_model_file(result))
        except:
            errors.append((result.parent, format_exc()))

    handler.gather_result_from_handle_one()
    for p, e in errors:
        print(f"Error while reading {p}")
        print(e)


def dataset_option():
    return typer.Option(
        None,
        "-d", "--dataset",
        help=f"choice from [{', '.join(common.datasets)}]")


def step_name_option():
    return typer.Option(
        "model",
        "-s", "--step",
        help=f"choice from [{', '.join([*common.get_procedure().steps.keys(), 'model'])}]")
