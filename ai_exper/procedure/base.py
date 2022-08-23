from pathlib import Path
from pydantic import BaseModel as PydanticBaseModel


class BaseStep(PydanticBaseModel):
    def get_prefix(self) -> str:
        pass

    def handle(self, input_path: Path, output_path: Path):
        pass

    def get_datapath(self, output_path: Path) -> Path:
        return output_path.joinpath("data.hdf5")


class BaseModel(PydanticBaseModel):
    def train(self, input_path: Path, outfile_path: Path):
        pass

    def test(self, input_path: Path, output_path: Path):
        pass

    def handle(self, input_path: Path, output_path: Path):
        self.train(input_path, output_path)
        self.test(input_path, output_path)
