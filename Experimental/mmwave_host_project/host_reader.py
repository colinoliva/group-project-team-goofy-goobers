import time
import csv

def parse_line(line):
    fields = line.strip().split(",")
    data = {}

    for field in fields:
        key, value = field.split(":")
        data[key] = value

    return data

print("Starting host-side reader...\n")

with open("mock_ti_data.txt", "r") as infile, open("log.csv", "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["frame", "range", "azimuth", "doppler", "presence"])

    for line in infile:
        frame = parse_line(line)

        frame_id = int(frame["FRAME"])
        range_val = float(frame["RANGE"])
        azimuth = float(frame["AZIMUTH"])
        doppler = float(frame["DOPPLER"])
        presence = int(frame["PRESENCE"])

        writer.writerow([frame_id, range_val, azimuth, doppler, presence])

        print(f"[Frame {frame_id}] Logged")

        if presence == 1:
            print("Target detected")
            print(f"Range: {range_val} m")
            print(f"Azimuth: {azimuth} deg")
            print(f"Doppler: {doppler} m/s")
        else:
            print("No target detected")

        print("-" * 30)
        time.sleep(1)