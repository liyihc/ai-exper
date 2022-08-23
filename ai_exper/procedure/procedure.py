from contextlib import contextmanager
from datetime import timedelta, timezone, datetime
from distutils.util import strtobool
from enum import Enum
import json
from pathlib import Path
from random import randbytes
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple, Type, Union
import sty
from pydantic import BaseModel as PydanticBaseModel
from .base import BaseStep, BaseModel

tz = timezone(timedelta(hours=8))
def datetime_now():
    return datetime.now(tz).replace(tzinfo=None)

TIMEFMT = '%Y%m%d-%H%M%S'


KT = Union[str, Enum]


class Procedure:
    def __init__(self, data_path: Path) -> None:
        self.steps: Dict[str, Dict[str, Type[BaseStep]]] = {}
        self.models: Dict[str, Type[BaseModel]] = {}
        self.data_path = data_path
        if not self.DATASET_PATH.exists():
            self.DATASET_PATH.mkdir()
        if not self.RESULT_PATH.exists():
            self.RESULT_PATH.mkdir()
        self.current_step = None
        self.back_hooks: List[Callable[[Path], Any]] = []

    @property
    def DATASET_PATH(self):
        return self.data_path / 'datasets'

    @property
    def RESULT_PATH(self):
        return self.data_path / 'results'

    @contextmanager
    def add_steps(self, step_name: KT):
        if isinstance(step_name, Enum):
            step_name = step_name.value
        assert step_name not in self.steps
        assert step_name != 'model'
        self.steps[step_name] = {}
        self.current_step = step_name
        yield

    def add_step(self, name: KT, step_type: Type[BaseStep]):
        assert self.current_step
        if isinstance(name, Enum):
            name = name.value
        steps = self.steps[self.current_step]
        assert name not in steps
        assert issubclass(step_type, BaseStep)
        steps[name] = step_type

    def add_model(self, model_name: KT, model_type: Type[BaseModel]):
        if isinstance(model_name, Enum):
            model_name = model_name.value
        assert model_name not in self.models
        assert issubclass(model_type, BaseModel)
        self.models[model_name] = model_type

    def print(self, params: 'MultiParameters'):
        print()
        print(sty.fg.li_blue + sty.ef.bold +
              "MODEL PARAMETER DETAIL" + sty.rs.all)

        for step_param in params.steps:
            print_stepmodel_name(
                step_param.step, step_param.use, self.steps[step_param.step].keys())
            print_title_param(
                f"{step_param.step}.{step_param.use}", step_param.params)

        print(sty.fg.li_blue + "models:" + sty.fg.rs)
        for k, model in params.models.items():
            print_title_value(f"  {k}", model.use, readonly=True)
            print_title_param(f"\tmodel.{k}", model.params)

    def get_completer(self, mp: 'MultiParameters'):
        cmds: List[str] = []
        for step_param in mp.steps:
            cmds.append(f"{step_param.step}=")
            cmds.extend(get_pydantic_model_completers(
                step_param.params, f"{step_param.step}.{step_param.use}"))
        for k, model in mp.models.items():
            cmds.extend(get_pydantic_model_completers(
                model.params, f"model.{k}"))
        cmds.extend(["start", "exit", "save-to:"])

        def completer(text, state):
            rets = [c for c in cmds if c.startswith(text)]
            rets.append(-1)
            return rets[state]
        return completer

    def update(self, mp: 'MultiParameters', k: str, v: str) -> bool:
        if not k.startswith("model"):
            if k in self.steps:
                for step_param in mp.steps:
                    if step_param.step != k:
                        continue
                    if v in self.steps[k]:
                        step_param.use = v
                        step_param.params = self.steps[k][v]()
                        return True
            else:
                k, _, *name = k.split('.')
                for step_param in mp.steps:
                    if step_param.step != k:
                        continue
                    return pydantic_set_attr(step_param.params, v, *name)
        else:
            _, k, *name = k.split('.')
            if k in mp.models:
                model_param = mp.models[k]
                return pydantic_set_attr(model_param.params, v, *name)
        return False

    def construct_params(self):
        mp = MultiParameters()
        for name, steps in self.steps.items():
            for step_name, step in steps.items():
                mp.steps.append(StepParameters(
                    step=name,
                    use=step_name,
                    params=step()))
                break
        return mp

    def add_model_to_param(self, mp: 'MultiParameters', model: str, params: Union[BaseModel, None] = None):
        label = 97
        while (tmp_label := f'{model}_{chr(label)}') in mp.models:
            label += 1
        tmp = mp.models[tmp_label] = ModelParameters(
            use=model, params=params or self.models[model]())
        return tmp

    def add_models_to_param(self, mp: 'MultiParameters', models: List[str]):
        if 'all' in models:
            models = list(self.models.keys())
        return [self.add_model_to_param(mp, model) for model in models]

    def parse_model_file(self, json_path: Path):
        return self.parse_model_json(json.loads(json_path.read_text()))

    def parse_multi_model_file(self, json_path: Path):
        return self.parse_multi_model_json(json.loads(json_path.read_text()))

    def parse_model_json(self, j: dict):
        return Parameters(
            steps=self._parse_step_json(j["steps"]),
            model=self._parse_model_json(j["model"])
        )

    def parse_multi_model_json(self, j: dict):
        return MultiParameters(
            steps=self._parse_step_json(j["steps"]),
            models={
                k: self._parse_model_json(m)
                for k, m in j["models"].items()
            }
        )

    def _parse_step_json(self, steps: List[dict]):
        return [
            StepParameters(
                step=step_param['step'],
                use=step_param['use'],
                params=self.steps[step_param['step']][step_param['use']].parse_obj(step_param['params']))
            for step_param in steps
        ]

    def _parse_model_json(self, m: Union[dict, None]):
        if m is None:
            return None
        use = m['use']
        param = m['params']
        return ModelParameters(
            use=use,
            params=self.models[use].parse_obj(param)
        )

    def eq(self, params1: 'Parameters', params2: 'Parameters', to_step: Union[Enum, str, Literal['model']]):
        if isinstance(to_step, Enum):
            to_step = to_step.value
        for step1, step2 in zip(params1.steps, params2.steps):
            if step1 != step2:
                return False
            if step1.step == to_step:
                return True
        # to_step == 'model'
        if len(params1.steps) != len(params2.steps):
            return False
        if params1.model.use != params2.model.use:
            return False

        return not pydantic_diff(params1.model.params, params2.model.params)

    def register_back_hook(self, hook: Callable[[Path], Any]):
        self.back_hooks.append(hook)

    def run(self, mp: 'MultiParameters'):
        prefix = []

        @contextmanager
        def get_path(step_name, p: Parameters) -> Tuple[bool, Path]:
            try:
                if step_name == 'model':
                    pattern = f"* {p.model.use}*/config.json"
                else:
                    tmp_prefix = f"{'-'.join(prefix)} {step_name}"
                    pattern = f"{tmp_prefix}*/config.json"
                for result in self.RESULT_PATH.glob(pattern):
                    if self.eq(p, self.parse_model_file(result), step_name):
                        output(
                            f"find and using {step_name} cache {result.relative_to(self.RESULT_PATH)}")
                        yield True, result.parent
                        return
                if step_name == 'model':
                    path = self.RESULT_PATH / \
                        f"{datetime_now().strftime(TIMEFMT)} {mp.steps[0].params.get_prefix()} {p.model.use}"
                    use = f"{step_name} {p.model.use}"
                else:
                    while True:
                        path = self.RESULT_PATH / \
                            f"{tmp_prefix} {randbytes(2).hex()}"
                        if not path.exists():
                            break
                    use = step_name
                output(f"calculate {use}", clr=sty.bg.da_green)
                if not path.exists():
                    path.mkdir()
                with path.joinpath("config.json").open('w') as f:
                    f.write(p.json(indent=4))
                yield False, path
            except Exception as e:
                target = path.parent.joinpath("error", path.name)
                if not target.parent.exists():
                    target.parent.mkdir()
                path.rename(target)
                import traceback
                with open(target / 'error.txt', 'w') as f:
                    f.write(traceback.format_exc())
                raise

        data_path = self.DATASET_PATH / mp.steps[0].use
        paths = []
        p = Parameters()

        for step in mp.steps:
            p.steps.append(step)
            prefix.append(step.params.get_prefix())
            with get_path(step.step, p) as (had, path):
                paths.append(path)
                if not had:
                    step.params.handle(data_path, path)
                data_path = step.params.get_datapath(path)

        for k, model in mp.models.items():
            p.model = model
            with get_path('model', p) as (had, path):
                if not had:
                    model.params.handle(data_path, path)
                    for hook in self.back_hooks:
                        hook(path)

    def get_different_name(self, param: 'Parameters', other_paraments: Iterable['Parameters']):
        steps = {step.step: Difference(name=step.step) for step in param.steps}
        model = Difference(name=param.model.use)

        for o in other_paraments:
            o_steps = {o_s.step: o_s for o_s in o.steps}
            for step in param.steps:
                store = steps[step.step]
                if step.step not in o_steps:
                    store.force = True
                elif step != (o_s := o_steps[step.step]):
                    if step.use != o_s.use:
                        store.params['model'] = step.use
                    else:
                        store.params.update(
                            pydantic_diff(step.params, o_s.params))
            tm = param.model
            om = o.model
            if tm.use == om.use and tm.params != om.params:
                model.params.update(pydantic_diff(tm.params, om.params))
        diffs = []
        for step in steps.values():
            if step.params:
                diffs.append(
                    f"{step.name} {' '.join(f'{p}={v}' for p, v in step.params.items())}")
        diffs.append(
            f"{model.name} {' '.join(f'{p}={v}' for p, v in model.params.items())}")

        return " ".join(diffs)


class StepParameters(PydanticBaseModel):
    step: str
    use: str
    params: BaseStep


class Difference(PydanticBaseModel):
    name: str
    params: Dict[str, Any] = {}
    force = False


class ModelParameters(PydanticBaseModel):
    use: str
    params: BaseModel


class Parameters(PydanticBaseModel):
    steps: List[StepParameters] = []
    model: Union[ModelParameters, None] = None


class MultiParameters(PydanticBaseModel):
    steps: List[StepParameters] = []
    models: Dict[str, ModelParameters] = {}


def print_stepmodel_name(title: str, value: str, values: Iterable[str]):
    print(title, '=',
          sty.fg.green + value + sty.rs.all,
          sty.fg.grey + sty.ef.italic + "of {" + ', '.join(values) + "}" + sty.rs.all)


def print_title_value(title: str, value: Enum, description: str = "", readonly=False):
    if isinstance(value, Enum):
        description = "of {" + \
            ', '.join(v.value for v in type(value)) + \
            "} " + (description or "")
        value = value.value
    clr = sty.fg.da_green if readonly else sty.fg.green
    print(
        title, '=',
        clr + str(value) + sty.rs.all,
        sty.fg.grey + sty.ef.italic + (description or "") + sty.rs.all)


def print_title_param(title: str, param: PydanticBaseModel):
    for key, field in param.__fields__.items():
        if issubclass(field.outer_type_, PydanticBaseModel):
            print_title_param(f"{title}.{key}", getattr(param, key))
        else:
            print_title_value(f"{title}.{key}", getattr(
                param, key), field.field_info.description)


def output(*args: str, sep=' ', clr=sty.bg.da_blue):
    print(clr + sep.join(map(str, args)) + sty.bg.rs)


def get_pydantic_model_completers(model: PydanticBaseModel, prefix=""):
    for k, f in model.__fields__.items():
        if issubclass(f.outer_type_, PydanticBaseModel):
            yield from get_pydantic_model_completers(getattr(model, k), f"{prefix}.{k}")
        else:
            yield f"{prefix}.{k}="


def pydantic_set_attr(model: PydanticBaseModel, value, name: str, *names: str):
    field = model.__fields__[name]
    if field.outer_type_ is bool:
        setattr(model, name, bool(strtobool(value)))
    elif issubclass(field.outer_type_, PydanticBaseModel):
        pydantic_set_attr(getattr(model, name), value, *names)
    else:
        setattr(model, name, model.__fields__[name].outer_type_(value))


def pydantic_diff(model: PydanticBaseModel, compared: PydanticBaseModel):
    return {
        key: value
        for key, field in model.__fields__.items()
        if field.field_info.extra.get('compare', True) and (value := getattr(model, key)) != getattr(compared, key, None)}
