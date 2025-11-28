import RPi.GPIO as GPIO
import time
from pynput import keyboard

# Servo pin numbers (BCM)
PAN_PIN = 18
TILT_PIN = 19

GPIO.setmode(GPIO.BCM)
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan_pwm = GPIO.PWM(PAN_PIN, 50)
tilt_pwm = GPIO.PWM(TILT_PIN, 50)

pan_pwm.start(0)
tilt_pwm.start(0)

# Start angles
pan_angle = 90
tilt_angle = 90

# Track which keys are currently held
keys_held = set()

# Arduino-like servo.write(angle)
def move_servo(pwm, angle):
    angle = max(0, min(180, angle))
    duty = 2.5 + (angle / 180.0) * 10.0
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.02)
    pwm.ChangeDutyCycle(0)  # reduce jitter
    return angle

def on_press(key):
    global pan_angle, tilt_angle

    if key in keys_held:
        return  # ignore repeated events!

    keys_held.add(key)

    try:
        if key == keyboard.Key.left:
            pan_angle -= 5
            pan_angle = move_servo(pan_pwm, pan_angle)
            print("Pan:", pan_angle)

        if key == keyboard.Key.right:
            pan_angle += 5
            pan_angle = move_servo(pan_pwm, pan_angle)
            print("Pan:", pan_angle)

        if key == keyboard.Key.up:
            tilt_angle += 5
            tilt_angle = move_servo(tilt_pwm, tilt_angle)
            print("Tilt:", tilt_angle)

        if key == keyboard.Key.down:
            tilt_angle -= 5
            tilt_angle = move_servo(tilt_pwm, tilt_angle)
            print("Tilt:", tilt_angle)

        if hasattr(key, "char") and key.char == 'q':
            print("Quit.")
            return False

    except:
        pass

def on_release(key):
    if key in keys_held:
        keys_held.remove(key)

def main():
    print("Use arrow keys to move servos ONE step per press.")
    print("Press 'q' to quit.\n")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    pan_pwm.stop()
    tilt_pwm.stop()
    GPIO.cleanup()

if __name__ == "__main__":
    main()
