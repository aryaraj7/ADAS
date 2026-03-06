# =============================================================
# ADAS — modules/ultrasonic.py
# Reads distance from ESP32 + HC-SR04 ultrasonic sensor via USB serial.
# ESP32 sends JSON lines: {"dist_cm": 123.4}
# =============================================================

import json
import threading
import time

import serial
import serial.tools.list_ports


class UltrasonicReader:
    """
    Background serial reader for ESP32 ultrasonic sensor.
    Continuously reads distance and exposes latest value via get_distance().
    """

    def __init__(self, port=None, baudrate=115200):
        self._port = port
        self._baudrate = baudrate
        self._serial = None
        self._distance_cm = -1.0  # -1 means no valid reading
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._connected = False

    def start(self):
        """Open serial port and start background reading thread."""
        port = self._port or self._auto_detect_port()
        if port is None:
            print("[Ultrasonic] No ESP32 serial port found.")
            return False

        try:
            self._serial = serial.Serial(port, self._baudrate, timeout=0.1)
            time.sleep(2)  # Wait for ESP32 reset after serial open
            self._connected = True
            print(f"[Ultrasonic] Connected on {port}")
        except serial.SerialException as e:
            print(f"[Ultrasonic] Failed to open {port}: {e}")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop reading and close serial port."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._connected = False
        self._distance_cm = -1.0
        print("[Ultrasonic] Stopped.")

    def get_distance_cm(self) -> float:
        """Return latest distance in cm. Returns -1 if no valid reading."""
        with self._lock:
            return self._distance_cm

    def get_distance_m(self) -> float:
        """Return latest distance in metres. Returns -1 if no valid reading."""
        with self._lock:
            d = self._distance_cm
        if d < 0:
            return -1.0
        return d / 100.0

    @property
    def connected(self) -> bool:
        return self._connected

    def _read_loop(self):
        """Background thread: read serial lines and parse JSON distance."""
        while not self._stop_event.is_set():
            try:
                if not self._serial or not self._serial.is_open:
                    break
                line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("{"):
                    data = json.loads(line)
                    dist = float(data.get("dist_cm", -1))
                    with self._lock:
                        self._distance_cm = dist
            except (json.JSONDecodeError, ValueError):
                continue
            except serial.SerialException:
                print("[Ultrasonic] Serial connection lost.")
                self._connected = False
                break

    @staticmethod
    def _auto_detect_port() -> str | None:
        """Try to find an ESP32 USB serial port automatically."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or "").lower()
            vid = p.vid or 0
            # Common ESP32 USB chips: CP210x (VID 0x10C4), CH340 (VID 0x1A86),
            # FTDI (VID 0x0403), ESP32-S2/S3 native USB (VID 0x303A)
            if vid in (0x10C4, 0x1A86, 0x0403, 0x303A):
                print(f"[Ultrasonic] Auto-detected ESP32 on {p.device} ({p.description})")
                return p.device
            if "cp210" in desc or "ch340" in desc or "esp32" in desc:
                print(f"[Ultrasonic] Auto-detected ESP32 on {p.device} ({p.description})")
                return p.device
        return None

    @staticmethod
    def list_ports() -> list[str]:
        """List all available serial ports."""
        return [p.device for p in serial.tools.list_ports.comports()]
