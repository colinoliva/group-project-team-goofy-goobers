Purpose:

We are establishing Communication Between a Texas Instruments Device and Rapsberry Pi 4 Using Mock mmWave Data.

Overview:
This project focuses on developing the host-side software for communication between Texas Instruments Devices and a Rapsberry Pi 4.

Since there has been difficulties with the hardware integration, (i.e Brandon is waiting on parts to set up the environment correctly), Mock data has been used to simulate the mmWave sensor output. The purpose is to build and validate the data pipeline before connecting to the actual hardware.

System Architecture:
TI Device (MSPM0G3507 / IWR6843 mmWave Sensor)
--> UART / Serial Communication
--> Raspberry Pi 4
--> Python Processing Script

Features implemented:
- Mock sensor data generation
- Host side data reading and parsing
- Extraction of
    - Frame ID
    - Range (meters)
    - Azimuth (degrees)
    - Doppler (velocity)
    - Presence detecting

- Real Time interpretation of the sensor data
- CSV logging of the processed frames.