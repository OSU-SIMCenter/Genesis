"""Typed reads of the ``AGF_*`` environment knobs.

Every knob is read through here so that a malformed value fails with a message
naming the variable, rather than a bare ``ValueError`` raised from inside a
class body during import. Two behaviours matter for ordinary knobs:

* **Unset, empty, or whitespace-only means "use the default."** ``AGF_MAX_FORCE=``
  is what a shell leaves behind after ``export AGF_MAX_FORCE=`` or an unset
  variable expanded into a wrapper script, and it previously crashed the import
  with ``could not convert string to float: ''``.
* **A malformed value is fatal, and says so.** Silently falling back to the
  default there would hide a typo'd knob behind a run that looks fine, which is
  the failure mode this sim keeps hitting.

Pins and guards pass ``blank_ok=False``. For those, **unset** still means
default (production derives ``AGF_ROBOT_TIME_TO_SECONDS`` from dt), but
**empty/whitespace is fatal**. Treating blank as default there silently unpins a
CFL sweep and makes it a joint controller sweep — the failure class this helper
exists to prevent. Do not advertise "set it empty to use the default" on a pin.

Defaults are returned unchanged (not round-tripped through ``str``), so an
unset knob yields the exact literal written at the call site.
"""

import os

__all__ = ["env_str", "env_float", "env_int", "env_bool"]


class EnvKnobError(ValueError):
    """A raw ``AGF_*`` value could not be read as the type its knob expects."""


def _raw(name, blank_ok=True):
    """The knob's value, or ``None`` if the default should be used.

    Unset always means default. Empty/whitespace means default only when
    ``blank_ok`` is true.
    """
    v = os.environ.get(name)
    if v is None:
        return None
    stripped = v.strip()
    if stripped:
        return stripped
    if blank_ok:
        return None
    raise EnvKnobError(
        f"{name}={v!r} is empty. Unset the variable to use the default, "
        f"or set a real value. Empty is not a pin."
    )


def _fail(name, raw, want, extra="", blank_ok=True):
    if blank_ok:
        hint = "Unset it (or set it empty) to use the default"
    else:
        hint = "Unset it to use the default (empty is invalid for this knob)"
    raise EnvKnobError(
        f"{name}={raw!r} is not {want}. {hint}{extra}."
    )


def env_str(name, default, blank_ok=True):
    v = _raw(name, blank_ok=blank_ok)
    return default if v is None else v


def env_float(name, default, blank_ok=True):
    v = _raw(name, blank_ok=blank_ok)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        _fail(name, v, "a number", f" ({default})", blank_ok=blank_ok)


def env_int(name, default, blank_ok=True):
    v = _raw(name, blank_ok=blank_ok)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        _fail(name, v, "a whole number", f" ({default})", blank_ok=blank_ok)


def env_bool(name, default, blank_ok=True):
    """Accepts the historical ``0``/``1`` spelling plus the obvious words."""
    v = _raw(name, blank_ok=blank_ok)
    if v is None:
        return default
    low = v.lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    _fail(name, v, "a boolean (0/1, true/false, yes/no, on/off)",
          f" ({'1' if default else '0'})", blank_ok=blank_ok)
