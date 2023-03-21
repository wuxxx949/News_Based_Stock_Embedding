import os
import pickle
from typing import Any, List


STOPWORDS = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once',
             'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for',
             'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is',
             's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until',
             'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above',
             'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before',
             'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',
             'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now',
             'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
             'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

def pickle_results(dir: str, name: str, obj: Any) -> None:
    """pickle an object
    """
    dbfile = open(os.path.join(dir, name), 'wb')
    pickle.dump(obj, dbfile)
    dbfile.close()

def load_pickled_obj(dir: str, name: str):
    file = open(os.path.join(dir, name), 'rb')
    data = pickle.load(file)
    # close the file
    file.close()

    return data

def get_path(dir_path: str) -> List[str]:
    """get all news article file paths

    Args:
        dir_path (str): directory contains all news

    Returns:
        List[str]: news file paths
    """
    path_lst = []
    subdirs = os.listdir(dir_path)
    for subdir in subdirs:
        # avoid hidden file
        if subdir.startswith('.'):
            continue
        # assume two layer folder structure
        news_paths = os.listdir(os.path.join(dir_path, subdir))
        for path in news_paths:
            path_lst.append(os.path.join(dir_path, subdir, path))

    return path_lst
