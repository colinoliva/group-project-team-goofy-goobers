System Summary:
This system uses a TI MSPM0G3507 microcontroller to interface three HC-SR04 ultrasonic sensors. Each sensor measures distance independently. The microcontroller collects and outputs the data over UART. 

The purpose of this setup is to simulate an end to end system. Specifically the Sensor, Embedded, and Communication Layers.

Sensor Layer ==> Embedded Layer ==> Communication Layer ==> Processing Layer ==> Application / Visualization Layer



Components Used:
- 1 x TI MSPM0G3507 LaunchPad
- 3 x HC-SR04 Ultrasonic Sensors
- 6 x Resistors. (1k and 2k)
- Breadboard + Jumper Wires



Breakdown of the System:
Each Ultrasonic Sensor measures the distance using ultrasonic sound waves:

1. Microcontroller sends a short pulse to the TRIG pin.
2. Sensor emits an ultrasonic burst.
3. When echo returns, the sensor sets the ECHO pin HIGH.
4. The duration of the HIGH signal is used to calculate the distance.



Multi Sensor Design

To support multiple sensors:
- Each sensor is assigned unique GPIO pins for TRIG and ECHO.
- Sensors are triggered sequentially.
- A Delay is added between the readings to prevent errors.
- Schematics are also added to showcase the wiring design.
    (Since HC-SR04 ECHO pin outputs 5V, and the GPIO pins are only 3.3V tolerant, to prevent damage, each ECHO line uses a voltage divider.)



Data Flow:
System uses two different communication methods:

Sensor Interface (Timing Based)
Each HC-SR04 sensor communicated with the microcontroller using a trigger pulse and timed echo response.

Data Output (UART)
After computing the distance, the microcontroller sends a formatted data to a computer using UART serial communication. This allows real time monitoring through a terminal.

Example Output:
S1,dist=13.2
S2,dist=25.7
S3,dist=42.1

While our group waits for the incoming TI sensors, this implementation serves as a validation of the embedded data pipeline and hardware integration required for multisensor systems.