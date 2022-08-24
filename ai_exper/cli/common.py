from typing import Dict, List, Optional, Set
from ai_exper import Procedure, ResultHandler


datasets: Set[str] = set()
models: Set[str] = set()

_module = None


def set_module(module):
    global _module
    _module = module


def get_procedure() -> Procedure:
    return getattr(_module, "procedure")


def get_result_handlers() -> Dict[str, ResultHandler]:
    return getattr(_module, "result_handlers", {})
