import datetime
from typing import List

class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


def extract_date(fpath: str) -> str:
    # TODO: make it OS agnostic
    # https://stackoverflow.com/questions/4579908/cross-platform-splitting-of-path-in-python
    d = fpath.split('/')
    date = [e for e in d if all([ee.isnumeric() for ee in e]) and len(e) > 0]
    if len(date) == 0:
        raise ValueError('no date in the path')

    if len(date) > 1:
        raise ValueError('multiple candidates for date in a path')

    return date[0]


def to_date(array) -> List[datetime.date]:
    return [e.to_pydatetime().date() for e in array]