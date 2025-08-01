__all__ = ("LazyStr", "blur", "blur_later")

from typing import Any, Callable

from pydantic import BaseModel

class LazyStr:
    """Call 'f' while calling its '_str_'."""
    def __init__(self, f:Callable[[],str]):
        self._f = f
        self.__val = None

    def __str__(self):
        if self.__val is None:
            self.__val = self._f()
        return self.__val



def blur_later(obj):
    return LazyStr(lambda: blur(obj))



def blur(obj) -> str:

    if obj is None:
        return "None"
    for t, f in _blur_map:
        if isinstance(obj, t):
            return f(obj)
    return f"...({type(obj).__name__})"

def _blur_dict(d : dict) -> str:
    contents = ','.join(f"{k!r} : {blur(v)}" for k, v in d.items())
    return '{' + contents + '}'

_blur_map : list[tuple[type, Callable[[Any], str]]] = [
    (dict, _blur_dict),
    (list, lambda li : '[' + ','.join(blur(v) for v in li) + ']'),
    (BaseModel, lambda b : blur(b.model_dump(by_alias=True))),
]