"""torch が無い環境で torch.save の .pt (zip 形式) を numpy として読む最小の読み込み器。

対応: dict / list / tuple / 数値 / 文字列 / numpy 配列 と、torch のテンソル (_rebuild_tensor_v2)。
未知の torch 型は None に置き換える。信頼できる自前のファイル (harness の出力) だけに使うこと。
usage (library): from zh_ptload import load_pt; d = load_pt(path)
"""
import io
import pickle
import zipfile

import numpy as np

_DT = {
    "HalfStorage": np.float16, "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "BFloat16Storage": np.uint16, "IntStorage": np.int32, "LongStorage": np.int64,
    "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.name = name
        self.dtype = _DT.get(name)


def _rebuild_tensor_v2(storage, offset, size, stride, *args, **kw):
    arr = storage
    if len(size) == 0:
        return arr[offset].copy()
    itemsize = arr.dtype.itemsize
    v = np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride))
    return np.array(v)


class _Unpickler(pickle._Unpickler):  # 純 Python 版 (C 版は persistent_load の戻り値の扱いで失敗する)
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf = zf
        self.prefix = prefix
        self.cache = {}

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *a, **k: data
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module.startswith("numpy"):
            return super().find_class(module, name)
        if module in ("builtins", "_codecs"):
            return super().find_class(module, name)
        if module == "torch" and name in ("float16", "float32", "int32", "int64", "bfloat16", "bool"):
            return name
        return lambda *a, **k: None

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, st, key, _loc, numel = pid
        if key not in self.cache:
            dt = st.dtype if isinstance(st, _StorageType) else getattr(st, "dtype", np.uint8)
            if isinstance(st, _StorageType) and st.name == "UntypedStorage":
                dt = np.uint8
            raw = self.zf.read(f"{self.prefix}/data/{key}")
            self.cache[key] = np.frombuffer(raw, dtype=dt)
        return self.cache[key]


def load_pt(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pk = [n for n in names if n.endswith("/data.pkl")][0]
    prefix = pk[: -len("/data.pkl")]
    return _Unpickler(io.BytesIO(zf.read(pk)), zf, prefix).load()
