import json
import math
import os
import tempfile
from pathlib import Path


def read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('JSON 字段重复：' + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('非有限数值')))


def sanitize(value):
    """把非有限浮点转为 JSON null，避免 NaN/Infinity 混入业务结果。

    未定义的估计值（如无支持候选的点估计）按既有约定用 null 表达，不用零替代。
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=target.parent, prefix='.' + target.name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            # 仍保持严格：非有限数值不允许写出。业务层应先用 sanitize 转为 null。
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)
