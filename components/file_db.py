from components.helpers import load_json_file, write_json_to_disk, touch_file
import os


def force_str_key(data: dict):
    result = dict()
    for key, value in data.items():
        result[str(key)] = value
    return result


class FileDB:
    def __init__(self, path: str):
        self.path = touch_file(path)
        # values
        self._data = self._load_from_disk(verify=True)
        self._mount_key = None

    '''
    protected
    '''

    def _load_from_disk(self, verify=False):
        result = {}
        if verify and os.stat(self.path).st_size == 0:
            write_json_to_disk(result, self.path)
        else:
            result = load_json_file(self.path)
        return result

    '''
    public: row-action
    '''

    def has(self, key, flush=False):
        return str(key) in self.get_all(flush=flush)

    def get(self, key, default_value=None, flush=True):
        return self.get_all(flush=flush).get(str(key), default_value)

    def set(self, key, value, only_different=False, flush=True):
        key = str(key)
        data = self.get_all(flush=False)
        if only_different and data.get(key, None) == value:
            return None
        data.update({key: value})
        if flush:
            self.flush()

    def update_many(self, _dict, flush=True):
        self.get_all().update(force_str_key(_dict))
        if flush:
            self.flush()

    def pop(self, key, default_value=None, w_flush=True, r_flush=False):
        data = self.get_all(flush=r_flush)
        if key not in data:
            return default_value
        result = data.pop(str(key))
        if w_flush:
            self.flush()
        return result

    '''
    public: data keys
    '''

    def get_all(self, flush=False):
        if flush:
            self._data = self._load_from_disk()
        return self._data[self._mount_key] if self._mount_key is not None else self._data

    def set_all(self, data: dict, flush=True):
        if self._mount_key is None:
            self._data = data
        else:
            self._data[self._mount_key] = data
        if flush:
            self.flush()

    def reload_from_disk(self):
        result = self.get_all(flush=True)
        self.set_all(result, flush=False)
        return result

    def items(self, flush=False):
        return self.get_all(flush).items()

    '''
    public: mount
    '''

    def is_mounting(self):
        return self._mount_key is not None

    def mount_on(self, key):
        assert not self.is_mounting()
        self._mount_key = key
        if key not in self._data:
            self._data[key] = dict()

    def unmount(self, clear_m_key=False, flush=False):
        assert self.is_mounting()
        if clear_m_key:
            self._data.pop(self._mount_key)
            if flush:
                self.flush()
        self._mount_key = None

    '''
    public: file-action
    '''

    def flush(self):
        self._data = force_str_key(self._data)
        return write_json_to_disk(self._data, self.path)

    def clear(self):
        self.set_all(dict(), flush=True)
