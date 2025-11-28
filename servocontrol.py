import RPi.GPIO as GPIO
import time
from pynput import keyboard

PAN_PIN = 18
TILT_PIN = 19

GPIO.setmode(GPIO.BCM)
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan = GPIO.PWM(PAN_PIN, 50)
tilt = GPIO.PWM(TILT_PIN, 50)

pan.start(0)
tilt.start(0)

# Starting angles
pan_angle = 90
tilt_angle = 90

def move_servo(pwm, angle):
    if angle < 0: angle = 0
    if angle > 180: angle = 180
    duty = 2.5 + (angle / 180.0) * 10
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.02)
    pwm.ChangeDutyCycle(0)  # reduce jitter
    return angle

def on_press(key):
    global pan_angle, tilt_angle

    try:
        if key == keyboard.Key.left:
            pan_angle -= 5
            pan_angle = move_servo(pan, pan_angle)

        elif key == keyboard.Key.right:
            pan_angle += 5
            pan_angle = move_servo(pan, pan_angle)

        elif key == keyboard.Key.up:
            tilt_angle += 5
            tilt_angle = move_servo(tilt, tilt_angle)

        elif key == keyboard.Key.down:
            tilt_angle -= 5
            tilt_angle = move_servo(tilt, tilt_angle)

        elif key.char == 'q':
            print("Quitting...")
            return False  # stop listener

    except AttributeError:
        pass  # ignore unknown keys

def main():
    print("Use arrow keys to control pan/tilt")
    print("Press 'q' to quit.\n")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    pan.stop()
    tilt.stop()
    GPIO.cleanup()

if __name__ == "__main__":
    main()
