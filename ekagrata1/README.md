# EKAGRATA: AI-Powered Athlete Digital Twin

EKAGRATA is a real-time motion analysis system for badminton smash technique. It tracks player motion using a webcam and a wearable ESP32+BNO055 IMU, computes joint angles, scores the smash, and visualizes the movement as a live 3D digital twin in Blender.

## Setup Instructions

1. **Install Python Dependencies:**
   Ensure you have Python 3.11 installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Settings:**
   Edit `config.yaml` to set your camera index, IMU IP/port, and Blender settings.

3. **Flash ESP32 Firmware:**
   - Open `firmware/esp32_imu/esp32_imu.ino` in the Arduino IDE.
   - Install the `Adafruit BNO055` and `Adafruit Unified Sensor` libraries.
   - Update the Wi-Fi credentials and destination IP/port at the top of the file.
   - Flash to your ESP32.

## Running the Application

1. **Start the Blender Receiver:**
   - Open Blender.
   - Go to the Scripting workspace.
   - Open `blender/ekagrata_rig_receiver.py` and run it (Alt+P). It will listen for incoming data.

2. **Start the Python Main Loop:**
   - Run the main application:
     ```bash
     python src/main.py
     ```
   - The system will attempt to connect to the IMU and Blender. If either fails, it will gracefully fall back to camera-only mode.
   - Perform a smash motion!
   - Press the `q` key on the OpenCV window to exit and generate a 1-page PDF session report.

## Architecture

- **Camera (MediaPipe):** Captures full body pose landmarks.
- **IMU (ESP32+BNO055):** Provides high-speed orientation of the racket arm.
- **Fusion:** A complementary filter combines both data sources to reduce drift and occlusion errors.
- **Analysis:** Detects peak wrist velocity as the impact moment and computes a technique score.
- **Visualization:** Streams data via local sockets to a Blender rigged model.
