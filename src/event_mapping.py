"""Sysmon Event ID → human-readable name mapping."""

EVENT_MAPPING = {
    1: "Process Creation",
    2: "File Creation Time Changed",
    3: "Network Connection",
    4: "Sysmon Service State Changed",
    5: "Process Terminated",
    6: "Driver Loaded",
    7: "Image Loaded",
    8: "CreateRemoteThread",
    10: "ProcessAccess",
    11: "FileCreate",
    12: "RegistryEvent (Create/Delete)",
    13: "RegistryEvent (Value Set)",
    15: "FileCreateStreamHash",
    16: "Sysmon Config Changed",
    17: "PipeEvent (Created)",
    18: "PipeEvent (Connected)",
    22: "DNSEvent",
    23: "FileDelete",
    255: "Error",
}
