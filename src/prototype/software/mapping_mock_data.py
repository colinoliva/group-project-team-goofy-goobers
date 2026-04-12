import numpy as np
import matplotlib.pyplot as plt

# Sensor positions (EDIT THESE to match your setup)
S1 = np.array([0, 0])
S2 = np.array([20, 0])
S3 = np.array([10, 15])

# Your data (parsed)
data = [
    (6.9, 13.9, 4.4),
    (7.8, 8.8, None),
    (1.3, 9.2, 6.7),
    (1.8, 24.3, 6.2),
    (26.1, 21.8, 10.2),
    (2.0, 12.8, 3.8),
    (6.1, 18.7, 8.1),
    (5.9, 24.5, 24.4),
    (1.4, 20.2, 4.4),
    (1.8, 11.7, 4.2),
    (3.7, 15.7, 3.7),
    (12.4, 24.7, 18.6),
    (108.9, 22.7, 43.9),
    (16.2, 12.6, None),
    (8.3, 13.6, None),
    (12.7, 12.3, None),
]

def trilaterate(p1, p2, p3, r1, r2, r3):
    try:
        # Standard Linear system of equations for trilateration
        A = 2 * (p2 - p1)
        B = 2 * (p3 - p1)
        C = r1**2 - r2**2 - np.dot(p1, p1) + np.dot(p2, p2)
        D = r1**2 - r3**2 - np.dot(p1, p1) + np.dot(p3, p3)

        mat = np.array([A, B])
        vec = np.array([C, D])

        # Solve for the best fit position
        pos = np.linalg.lstsq(mat, vec, rcond=None)[0]
        return pos
    except Exception:
        return None

positions = []

for d1, d2, d3 in data:
    if d1 is not None and d2 is not None and d3 is not None:
        pos = trilaterate(S1, S2, S3, d1, d2, d3)
        if pos is not None:
            positions.append(pos)

positions = np.array(positions)

# Plot
if len(positions) > 0:
    plt.figure(figsize=(10, 8))
    plt.scatter(positions[:, 0], positions[:, 1], label="Object Positions", alpha=0.6)
    
    # Plot Sensors
    sensors = np.array([S1, S2, S3])
    plt.scatter(sensors[:, 0], sensors[:, 1], marker='x', color='red', s=100, label="Sensors")
    
    for i, txt in enumerate(["S1", "S2", "S3"]):
        plt.text(sensors[i, 0] + 0.5, sensors[i, 1] + 0.5, txt, fontweight='bold')

    plt.title("Ultrasonic Trilateration Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.show()
else:
    print("No valid positions could be calculated. Check if your data points have all three sensor distances.")