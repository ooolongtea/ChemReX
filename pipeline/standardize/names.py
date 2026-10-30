"""Normalize name containers without turning arbitrary objects into chemicals."""


def string_names(values):
    if isinstance(values, str):
        value = values.strip()
        return [value] if value else []
    if isinstance(values, (list, tuple)):
        return list(dict.fromkeys(name for value in values for name in string_names(value)))
    return []
