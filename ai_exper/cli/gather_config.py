from pathlib import Path
from typing import Iterable, List, Optional
from ai_exper.procedure import MultiParameters, Parameters
from ai_exper import ResultHandler
from . import common, gather

procedure = common.get_procedure()


class GatherConfig(ResultHandler):
    def __init__(self) -> None:
        super().__init__()
        self.mps: List[MultiParameters] = []

    def handle_one(self, folder_path: Path, params: Parameters):
        for mp in self.mps:
            if params.steps == mp.steps:
                procedure.add_model_to_param(
                    mp, params.model.use, params.model.params)
                break
        else:
            mp = MultiParameters(steps=params.steps)
            self.mps.append(mp)
            procedure.add_model_to_param(
                mp, params.model.use, params.model.params)

    def gather_result_from_handle_one(self):
        for mp in self.mps:
            mp.models = dict(
                sorted(mp.models.items(), key=lambda item: item[0]))
            print(mp.json(indent=4))
            print()


def gather_config(
        dataset: Optional[str] = gather.dataset_option(),
        step_name: Optional[str] = gather.step_name_option()):
    return gather.do_gather(GatherConfig, dataset, step_name)
