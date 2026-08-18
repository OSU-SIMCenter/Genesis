"""Typed reads of the ``AGF_*`` environment knobs.

Every knob is read through here so that a malformed value fails with a message
naming the variable, rather than a bare ``ValueError`` raised from inside a
class body during import. Two behaviours matter:

* **Unset, empty, or whitespace-only means "use the default."** ``AGF_MAX_FORCE=``
  is what a shell leaves behind after ``export AGF_MAX_FORCE=`` or an unset
  variable expanded into a wrapper script, and it previously crashed the import
  with ``could not convert string to float: ''``.
* **A malformed value is fatal, and says so.** Silently falling back to the
  default there would hide a typo'd knob behind a run that looks fine, which is
  the failure mode this sim keeps hitting.

Defaults are returned unchanged (not round-tripped through ``str``), so an
unset knob yields the exact literal written at the call site.
"""

import os

__all__ = ["env_str", "env_float", "env_int", "env_bool"]


class EnvKnobError(ValueError):
    """A raw ``AGF_*`` value could not be read as the type its knob expects."""


def _raw(name):
    """The knob's value, or ``None`` if unset/blank -- blank means default."""
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


def _fail(name, raw, want, extra=""):
    raise EnvKnobError(
        f"{name}={raw!r} is not {want}. Unset it (or set it empty) to use the "
        f"default{extra}."
    )


def env_str(name, default):
    v = _raw(name)
    return default if v is None else v


def env_float(name, default):
    v = _raw(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        _fail(name, v, "a number", f" ({default})")


def env_int(name, default):
    v = _raw(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        _fail(name, v, "a whole number", f" ({default})")


def env_bool(name, default):
    """Accepts the historical ``0``/``1`` spelling plus the obvious words."""
    v = _raw(name)
    if v is None:
        return default
    low = v.lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    _fail(name, v, "a boolean (0/1, true/false, yes/no, on/off)",
          f" ({'1' if default else '0'})")
