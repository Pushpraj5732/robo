#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// ---------------------------------------------------------
// Configuration
// ---------------------------------------------------------
const char* ssid = "YOUR_WIFI_SSID";           // TODO: Replace with your Wi-Fi SSID
const char* password = "YOUR_WIFI_PASSWORD";   // TODO: Replace with your Wi-Fi Password

// IP of the computer running the Python script
const char* udpAddress = "192.168.1.100";      // TODO: Replace with your PC's IP address
const int udpPort = 8888;                      // Must match config.yaml

// ---------------------------------------------------------

// BNO055 I2C address is usually 0x28 or 0x29
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

WiFiUDP udp;

void setup() {
  Serial.begin(115200);
  Serial.println("EKAGRATA IMU Firmware Starting...");

  // Initialize Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP address: ");
  Serial.println(WiFi.localIP());

  // Initialize BNO055
  if (!bno.begin(Adafruit_BNO055::OPERATION_MODE_NDOF)) {
    Serial.println("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while (1);
  }

  // Use external crystal for better accuracy
  bno.setExtCrystalUse(true);

  Serial.println("BNO055 Initialized. Streaming quaternions over UDP at ~100Hz...");
}

void loop() {
  // Read quaternion data
  imu::Quaternion quat = bno.getQuat();

  // Create JSON payload
  // Format: {"timestamp": 12345, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
  String payload = "{";
  payload += "\"timestamp\":" + String(millis()) + ",";
  payload += "\"qw\":" + String(quat.w(), 4) + ",";
  payload += "\"qx\":" + String(quat.x(), 4) + ",";
  payload += "\"qy\":" + String(quat.y(), 4) + ",";
  payload += "\"qz\":" + String(quat.z(), 4);
  payload += "}";

  // Send via UDP
  udp.beginPacket(udpAddress, udpPort);
  udp.print(payload);
  udp.endPacket();

  // Aim for roughly 100Hz (10ms total loop time)
  delay(10); 
}
