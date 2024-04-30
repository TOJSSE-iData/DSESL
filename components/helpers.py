from sklearn.metrics import f1_score

import numpy as np
import datetime
import json
import os

'''
args
'''


def datetime_now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


'''
output
'''


def Tprint(*args) -> str:
    result = f"[{datetime_now()}]"
    result += " ".join([str(x) for x in args])
    print(result)
    return result


def TLprint(*args, width=88, lchar='-', head_break=False, silent=False):
    result = ("\n" if head_break else "") + lchar * width + "\n"
    result += " ".join([str(x) for x in args]) + "\n"
    result += lchar * width
    if not silent:
        print(result)
    return result


COLOR_DICT = {
    "black": 30, "red": 31, "green": 32, "yellow": 33, "blue": 34,
    "purple": 35, "lightblue": 36, "white": 37,
}


def colorful_print(text: str, color="white"):
    return "\033[{}m".format(COLOR_DICT[color]) + text + "\033[0m"


'''
path
'''


def root_abs_path(rel_path: str):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), rel_path)


def create_dir_if_not_exists(*paths, suffix_dot_clear=False):
    result = []
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
        while suffix_dot_clear and path.endswith("."):
            path = path[:-1]
        result.append(path)
    result = result[0] if len(result) == 1 else result
    return result


def get_abs_paths_in_folder(
        folder_path, include_file=True, include_folder=False,
        filter_func=None, sort_key=None, with_is_dir=False,
):
    assert include_folder or include_file
    assert not with_is_dir or (include_file and include_folder)
    result = []
    # different path types
    for _path in os.listdir(folder_path):
        _path = os.path.abspath(os.path.join(folder_path, _path))
        _is_dir = os.path.isdir(_path)
        if not include_folder and _is_dir:
            continue
        if not include_file and os.path.isfile(_path):
            continue
        if filter_func and not filter_func(_path):
            continue
        result.append((_is_dir, _path) if with_is_dir else _path)
    # sort function
    if sort_key is None:
        sort_key = lambda p: p
    # sort
    result.sort(key=lambda x: sort_key(x[1]) if with_is_dir else sort_key(x))
    return result


'''
file
'''


def touch_file(path: str):
    create_dir_if_not_exists(os.path.dirname(path))
    if not os.path.exists(path):
        f = open(path, "w")
        f.close()
    return path


def load_json_file(path) -> dict:
    with open(path, "rt", encoding="utf8") as f:
        result = json.load(f)
    return result


def write_json_to_disk(data, dest_path: str):
    create_dir_if_not_exists(os.path.dirname(dest_path))
    with open(dest_path, "wt", encoding="utf8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return dest_path


'''
calc metrics
'''


# pred score: float -> int
def _get_labels(pred_probas, threshold):
    return (pred_probas >= threshold).astype(int)


# F1-score under one threshold
def calculate_f1(target, pred_label, threshold: float) -> float:
    assert 0.0 <= threshold <= 1.0
    return f1_score(y_true=target, y_pred=_get_labels(pred_label, threshold))


# F1-score under different thresholds
def calculate_max_f1(target, pred_label, thresholds=np.arange(0, 1, 0.02)):
    max_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        labels = _get_labels(pred_label, threshold)
        current_f1 = f1_score(target, labels)
        if current_f1 > max_f1:
            max_f1 = current_f1
            best_threshold = threshold

    return max_f1, best_threshold


if __name__ == '__main__':
    Tprint("test")
