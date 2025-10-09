# workshops_ai/utils/compat.py
from functools import wraps
import logging
log = logging.getLogger(__name__)

def accepts_org_id_kw(fn):
    @wraps(fn)
    def _wrap(*args, **kwargs):
        if "org_id" in kwargs:
            kwargs.pop("org_id", None)
            log.debug({"event":"compat.org_id_ignored", "fn": fn.__name__})
        return fn(*args, **kwargs)
    return _wrap
