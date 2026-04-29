try:
    import pyrealsense2 as rs
except Exception as e:
    rs = None
    REALSENSE_IMPORT_ERROR = e
else:
    REALSENSE_IMPORT_ERROR = None

try:
    import pyorbbecsdk as ob
except Exception as e:
    ob = None
    ORBBEC_IMPORT_ERROR = e
else:
    ORBBEC_IMPORT_ERROR = None

# List the connected Intel RealSense cameras and print their serial numbers.
def list_realsense_devices():
    if rs is None:
        print(f"------------RealSense SDK unavailable: {REALSENSE_IMPORT_ERROR}------------")
        return

    try:
        ctx = rs.context()
        devices = ctx.devices
    except Exception as e:
        print(f"------------RealSense device query failed: {e}------------")
        return

    num_devices = len(devices)
    print(f"------------Detected {num_devices} RealSense device------------")

    if num_devices == 0:
        return

    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"Device {i}: Name={name}, Serial={serial}")


def _orbbec_device_info_value(device_info, method_name):
    try:
        return getattr(device_info, method_name)()
    except Exception:
        return None


def list_orbbec_devices():
    if ob is None:
        print(f"------------Orbbec SDK unavailable: {ORBBEC_IMPORT_ERROR}------------")
        return

    try:
        ctx = ob.Context()
        devices = ctx.query_devices()
    except Exception as e:
        print(f"------------Orbbec device query failed: {e}------------")
        return

    num_devices = devices.get_count()
    print(f"------------Detected {num_devices} Orbbec device------------")

    if num_devices == 0:
        return

    for i in range(num_devices):
        dev = devices.get_device_by_index(i)
        info = dev.get_device_info()
        name = _orbbec_device_info_value(info, "get_name")
        serial = _orbbec_device_info_value(info, "get_serial_number")
        uid = _orbbec_device_info_value(info, "get_uid")
        connection_type = _orbbec_device_info_value(info, "get_connection_type")
        print(f"Device {i}: Name={name}, Serial={serial}, UID={uid}, Connection={connection_type}")


def main():
    list_realsense_devices()
    list_orbbec_devices()


if __name__ == "__main__":
    main()
