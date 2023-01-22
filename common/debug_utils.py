import sys
import math

from typing import NamedTuple
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def _getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

def as_Bytes(size_byte, precision=2):
    SUBFIX = ("B", "KB", "MB", "GB")
    power_indice = min(int(math.log(size_byte, 1024)), len(SUBFIX) - 1)
    _byte = SUBFIX[power_indice]
    
    _size = round(size_byte / 1024**power_indice, precision)
    return Bytes(_size, _byte, precision=precision)

class Bytes(NamedTuple):
    size: int
    subfix: str
    precision: int = 2
    
    def __str__(self): 
        _format = "{:." + str(self.precision) + "f} {}"
        return _format.format(self.size, self.subfix)
    
    @staticmethod
    def getsize(obj): return _getsize(obj)