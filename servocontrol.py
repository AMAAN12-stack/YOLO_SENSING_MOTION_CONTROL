import RPi.GPIO as GPIO
import time

PAN_PIN = 18
TILT_PIN = 19

GPIO.setmode(GPIO.BCM)
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

# 50 Hz PWM for servos
pan = GPIO.PWM(PAN_PIN, 50)
tilt = GPIO.PWM(TILT_PIN, 50)

pan.start(0)
tilt.start(0)

# This range (2.5–12.5) works for most servos and reduces jitter
def set_angle(pwm, angle):
    duty = 2.5 + (angle / 180.0) * 10.0  # 2.5% → 0°, 12.5% → 180°
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.02)  # small delay helps stabilize movement
    pwm.ChangeDutyCycle(0)  # stop sending PWM to reduce shiver

try:
    # Example: sweep motion
    while True:
        for angle in range(0, 181, 5):
            set_angle(pan, angle)
            set_angle(tilt, 180 - angle)

        for angle in range(180, -1, -5):
            set_angle(pan, angle)
            set_angle(tilt, 180 - angle)

except KeyboardInterrupt:
    pass

pan.stop()
tilt.stop()
GPIO.cleanup()
