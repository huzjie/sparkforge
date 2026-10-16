"""Simple backend registry."""
_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    return deco


def get_backend(name, config):
    if name not in _BACKENDS:
        raise KeyError("unknown backend '%s' (available: %s)" % (name, list(_BACKENDS)))
    return _BACKENDS[name](config)


def list_backends():
    return list(_BACKENDS.keys())
