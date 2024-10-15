import time
import board
import digitalio

import usb_hid
from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keycode import Keycode

time.sleep(5)

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

# Set up a keyboard device.
g_kbd = Keyboard(usb_hid.devices)

#Keycode = https://docs.circuitpython.org/projects/hid/en/latest/api.html#adafruit-hid-keycode-keycode

#Loop through all alphanumeric keys to test
keycodeList = [
	Keycode.A,
	Keycode.B,
	Keycode.C,
	Keycode.D,
	Keycode.E,
	Keycode.F,
	Keycode.G,
	Keycode.H,
	Keycode.I,
	Keycode.J,
	Keycode.K,
	Keycode.L,
	Keycode.M,
	Keycode.N,
	Keycode.O,
	Keycode.P,
	Keycode.Q,
	Keycode.R,
	Keycode.S,
	Keycode.T,
	Keycode.U,
	Keycode.V,
	Keycode.W,
	Keycode.X,
	Keycode.Y,
	Keycode.Z,
	Keycode.ZERO,
	Keycode.ONE,
	Keycode.TWO,
	Keycode.THREE,
	Keycode.FOUR,
	Keycode.FIVE,
	Keycode.SIX,
	Keycode.SEVEN,
	Keycode.EIGHT,
	Keycode.NINE,
	]

for key in keycodeList:
	#Unshifted
	g_kbd.press(key)
	time.sleep(0.06)
	g_kbd.release(key)

	time.sleep(0.2)

	#Shifted
	g_kbd.press(Keycode.SHIFT)
	g_kbd.press(key)
	time.sleep(0.100)
	g_kbd.release(key)
	g_kbd.release(Keycode.SHIFT)

	time.sleep(0.2)


#Start LED heartbeat to indicate test completion
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