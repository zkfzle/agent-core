def safe_get(data: dict, path, default=None):
    if data is None:
        return default

    keys = path.split('.')
    current = data

    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and key.isdigit():
            index = int(key)
            if 0 <= index < len(current):
                current = current[index]
            else:
                return default
        else:
            return default

        if current is None:
            return default

    return current


def create_nested_dict(path, value=None, separator='.'):
    if not path:
        return value

    keys = path.split(separator)
    result = current = {}

    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            current[key] = value
        else:
            current[key] = {}
            current = current[key]

    return result