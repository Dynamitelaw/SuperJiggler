import time
import board
from math import sin
from math import cos
import digitalio

import usb_hid
from adafruit_hid.mouse import Mouse

time.sleep(5)

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT

# Set up a mouse device
m = Mouse(usb_hid.devices)

# Click the left mouse button.
#m.click(Mouse.LEFT_BUTTON)

# Roll the mouse wheel away from the user one unit.
# Amount scrolled depends on the host.
#m.move(0, 0, -1)

#Move mouse in a spiral pattern
radius = 1
angle = 0
for step in range(500):
	#Move mouse pointer
	xDelta = int(radius*cos(angle))
	yDelta = int(-1*radius*sin(angle))

	m.move(x=xDelta, y=yDelta)

	#Increment angle and radius
	radius += 0.1
	angle += 0.1

	time.sleep(0.01)


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