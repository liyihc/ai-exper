from typing import Optional
import typer
from . import common, gather

if common.get_result_handlers():
    help_msg = f"choice from [{', '.join(common.get_result_handlers())}]"
else:
    help_msg = f"please define handlers in models:result_handlers"


def gather_custom(
        handler: str = typer.Argument(
            ...,
            help=help_msg),
        dataset: Optional[str] = gather.dataset_option(),
        step_name: Optional[str] = gather.step_name_option()):
    handler_type = common.get_result_handlers()[handler]
    return gather.do_gather(handler_type, dataset, step_name)