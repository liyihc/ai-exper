from pathlib import Path
from typing import Iterable
from ai_exper.procedure import Parameters


class ResultHandler:
    def handle_one(self, folder_path: Path, params: Parameters):
        pass

    def gather_result_from_handle_one(self):
        pass
