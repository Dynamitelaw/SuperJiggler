import time
import board
import digitalio

import usb_hid
from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keycode import Keycode

time.sleep(5)

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

for i in range(10):
	# Set up a keyboard device.
	kbd = Keyboard(usb_hid.devices)

	# Type lowercase 'a'. Presses the 'a' key and releases it.
	kbd.send(Keycode.A)

	# Type capital 'A'.
	kbd.send(Keycode.SHIFT, Keycode.A)

	# Type control-x.
	kbd.send(Keycode.CONTROL, Keycode.X)

	# You can also control press and release actions separately.
	kbd.press(Keycode.CONTROL, Keycode.X)
	kbd.release_all()

	# Press and hold the shifted '1' key to get '!' (exclamation mark).
	kbd.press(Keycode.SHIFT, Keycode.ONE)
	# Release the ONE key and send another report.
	kbd.release(Keycode.ONE)
	# Press shifted '2' to get '@'.
	kbd.press(Keycode.TWO)
	# Release all keys.
	kbd.release_all()

	time.sleep(1)

while True:
	led.value = True
	time.sleep(0.1)
	led.value = False
	time.sleep(0.1)
	led.value = True
	time.sleep(0.1)
	led.value = False
	time.sleep(0.1)
	led.value = True
	time.sleep(0.1)
	led.value = False
	time.sleep(0.1)

	time.sleep(1)