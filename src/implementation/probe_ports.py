import glob
import serial


for port in sorted(glob.glob("/dev/cu.usbserial-*")):
    try:
        with serial.Serial(port, 115200, timeout=1.0) as s:
            s.reset_input_buffer()
            s.write(b"\n")
            reply = s.read_until(b"mmwDemo:/>")
        kind = "CLI  <--" if b"mmwDemo" in reply else "no reply (probably the data port)"
    except Exception as e:
        kind = f"error: {e}"
    print(f"{port:36s} {kind}")
