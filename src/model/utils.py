import datetime
from typing import Iterable, List

import pandas as pd


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
    d = [e.replace('-', '') for e in fpath.split('/')]
    date = [e for e in d if all([ee.isnumeric() for ee in e]) and len(e) > 0]
    if len(date) == 0:
        raise ValueError('no date in the path')

    if len(date) > 1:
        raise ValueError('multiple candidates for date in a path')

    return date[0]


def to_date(array: Iterable) -> List[datetime.date]:
    return [e.to_pydatetime().date() for e in array]


def pd_anti_join(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """find df1 rows that are not in df2

    Args:
        df1 (pd.DataFrame): first dataframe
        df2 (pd.DataFrame): second dataframe

    Returns:
        pd.DataFrame: anti-join df
    """
    outer = df1.merge(df2, how='outer', indicator=True)
    anti_join = outer \
        .loc[(outer._merge=='left_only'), :] \
        .drop('_merge', axis=1)

    return anti_join