import RPi.GPIO as GPIO
import cv2
from pynput import keyboard

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)

# PWM setup for servos
pwm_17 = GPIO.PWM(17, 50)  # 50Hz frequency
pwm_27 = GPIO.PWM(27, 50)
pwm_17.start(7.5)  # Center position (90 degrees)
pwm_27.start(7.5)

# Servo positions
servo_x = 7.5
servo_y = 7.5
step = 0.5  # Adjustment step

def on_press(key):
    global servo_x, servo_y
    
    try:
        if key == keyboard.Key.up:
            servo_y = max(3, servo_y - step)  # Move up
            pwm_27.ChangeDutyCycle(servo_y)
            print(f"Up - Servo Y: {servo_y:.2f}")
        elif key == keyboard.Key.down:
            servo_y = min(12, servo_y + step)  # Move down
            pwm_27.ChangeDutyCycle(servo_y)
            print(f"Down - Servo Y: {servo_y:.2f}")
        elif key == keyboard.Key.left:
            servo_x = max(3, servo_x - step)  # Move left
            pwm_17.ChangeDutyCycle(servo_x)
            print(f"Left - Servo X: {servo_x:.2f}")
        elif key == keyboard.Key.right:
            servo_x = min(12, servo_x + step)  # Move right
            pwm_17.ChangeDutyCycle(servo_x)
            print(f"Right - Servo X: {servo_x:.2f}")
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        return False

try:
    print("Arrow keys to control servos. Press ESC to exit.")
    print("GPIO 17 (X-axis) / GPIO 27 (Y-axis)")
    
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

except KeyboardInterrupt:
    pass

finally:
    pwm_17.stop()
    pwm_27.stop()
    GPIO.cleanup()