import pigpio
import time

# GPIO pins
PAN_PIN = 18
TILT_PIN = 19

# connect to pigpio daemon
pi = pigpio.pi()
if not pi.connected:
    exit()

# Function to move servo (angle 0–180)
def set_servo(pin, angle):
    pulsewidth = 500 + (angle / 180.0) * 2000  # 500–2500 µs range
    pi.set_servo_pulsewidth(pin, pulsewidth)

try:
    while True:
        # Example motion sweep
        for angle in range(0, 181, 5):
            set_servo(PAN_PIN, angle)
            set_servo(TILT_PIN, 180 - angle)
            time.sleep(0.02)

        for angle in range(180, -1, -5):
            set_servo(PAN_PIN, angle)
            set_servo(TILT_PIN, 180 - angle)
            time.sleep(0.02)

except KeyboardInterrupt:
    pass

# Clean up
pi.set_servo_pulsewidth(PAN_PIN, 0)
pi.set_servo_pulsewidth(TILT_PIN, 0)
pi.stop()
