import RPi.GPIO as GPIO
import time
from pynput import keyboard

# Servo pins (BCM numbering)
PAN_PIN = 18
TILT_PIN = 19

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan_pwm = GPIO.PWM(PAN_PIN, 50)   # 50 Hz
tilt_pwm = GPIO.PWM(TILT_PIN, 50)

pan_pwm.start(0)
tilt_pwm.start(0)

# Starting positions
pan_angle = 90
tilt_angle = 90

# Arduino-like "servo.write()" equivalent
def move_servo(pwm, angle):
    if angle < 0: angle = 0
    if angle > 180: angle = 180

    duty = 2.5 + (angle / 180.0) * 10.0   # 2.5–12.5% pulse width
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.02)
    pwm.ChangeDutyCycle(0)  # stops jitter
    return angle

# Keyboard handler
def on_press(key):
    global pan_angle, tilt_angle

    try:
        if key == keyboard.Key.left:
            pan_angle -= 5
            pan_angle = move_servo(pan_pwm, pan_angle)
            print("Pan:", pan_angle)

        elif key == keyboard.Key.right:
            pan_angle += 5
            pan_angle = move_servo(pan_pwm, pan_angle)
            print("Pan:", pan_angle)

        elif key == keyboard.Key.up:
            tilt_angle += 5
            tilt_angle = move_servo(tilt_pwm, tilt_angle)
            print("Tilt:", tilt_angle)

        elif key == keyboard.Key.down:
            tilt_angle -= 5
            tilt_angle = move_servo(tilt_pwm, tilt_angle)
            print("Tilt:", tilt_angle)

        elif key.char == 'q':
            print("Quit.")
            return False

    except AttributeError:
        pass

def main():
    print("Use arrow keys to control the servos.")
    print("Press q to quit.\n")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    pan_pwm.stop()
    tilt_pwm.stop()
    GPIO.cleanup()

if __name__ == "__main__":
    main()
