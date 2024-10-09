import os
import time
from pynput import mouse as pmouse

mouse_controller = pmouse.Controller()

def on_move(x, y):
	print('Pointer moved to {0}'.format((x, y)))
	#return False

def on_click(x, y, button, pressed):
	if (pressed):
		print("Pressed")
	#return False

def on_scroll(x, y, dx, dy):
	print('Scrolled {0} at {1}'.format('down' if dy < 0 else 'up',(x, y)))
	#return False

listener = pmouse.Listener(on_move=None, on_click=on_click, on_scroll=None)
listener.start()

while (True):
	print(mouse_controller.position)
	time.sleep(0.1)

listener.stop()
