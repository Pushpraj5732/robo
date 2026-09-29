import socket
import json
import threading
import time

class ImuReceiver:
    def __init__(self, host="0.0.0.0", port=8888, buffer_size=1024):
        """
        Initializes the UDP listener for ESP32 IMU packets.
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.latest_data = None
        self.running = False
        self.thread = None
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Non-blocking allows us to check for stop signals in the loop
        self.sock.settimeout(0.1) 
        
        try:
            self.sock.bind((self.host, self.port))
            print(f"IMU Receiver listening on UDP {self.host}:{self.port}")
        except Exception as e:
            print(f"Warning: Could not bind UDP socket: {e}")
            self.sock = None

    def start(self):
        """Starts the background listening thread."""
        if self.sock is None:
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        return True

    def _listen_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.buffer_size)
                try:
                    # Expecting JSON: {"timestamp": 123, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
                    parsed_data = json.loads(data.decode('utf-8'))
                    self.latest_data = parsed_data
                except json.JSONDecodeError:
                    pass # Ignore malformed packets
            except socket.timeout:
                continue # Normal timeout, just loop again
            except Exception as e:
                if self.running:
                    print(f"IMU socket error: {e}")

    def get_latest(self):
        """
        Returns the latest parsed quaternion dictionary or None.
        Format: {'timestamp': int, 'qw': float, 'qx': float, 'qy': float, 'qz': float}
        """
        return self.latest_data

    def stop(self):
        """Stops the listening thread and cleans up."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.sock:
            self.sock.close()
