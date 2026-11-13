"""Is the machine on mains power?

Long model runs are developed on a laptop whose GTX 1650 and CPU, flat out, can draw
more than its 41 Wh battery delivers. Observed: a corpus run started on battery cut the
power within 20 seconds of loading the model (Windows Kernel-Power event 41, no bug
check) - five such shutdowns in 60 days. So model-heavy commands check before starting,
and corpus runs check again before every document.
"""

from __future__ import annotations

import ctypes
import sys


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),  # 0 = battery, 1 = mains, 255 = unknown
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def on_battery() -> bool:
    """True only when the OS positively reports battery power. Desktops, other
    platforms and unknown states return False, so the check never blocks them."""
    if sys.platform != "win32":
        return False
    status = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return False
    return status.ACLineStatus == 0
