# Typed, attribute-accessible replacement for the plain nested dicts that used
# to carry simulation data (parameters/types/activities/technologies/agents/policies).
from types import SimpleNamespace


class Struct(SimpleNamespace):
    """A data structure that supports both `obj.field` and `obj['field']`.

    The simulation mutates these objects throughout mod1-mod5 (e.g.
    `technologies['balancers']['use'] = {...}` gets added well after the
    initial load), so this keeps the old bracket syntax working for code
    that hasn't been converted yet, while new code can use attribute
    access. It is not a dict: iteration, `dict(...)`, etc. are not
    supported on purpose - `.keys()/.values()/.items()` cover the cases
    the simulation code actually needs.
    """

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def get(self, key, default=None):
        return self.__dict__.get(key, default)
