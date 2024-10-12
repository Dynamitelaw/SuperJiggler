import os
import time
from pynput import mouse as pmouse
import threading
import multiprocessing
import pickle
import shutil
import bz2
import gzip
import lzma



g_log_save_time = 120
g_newEventQueue = multiprocessing.Queue()
g_prev_mouse_event_lock = threading.Lock()
g_prev_mouse_event = None


class MouseEvent():
	"""docstring for ClassName"""
	def __init__(self, x_pos, y_pos, button=None, button_pressed=False, scroll_dx=None, scroll_dy=None):
		self.x_pos = x_pos
		self.y_pos = y_pos
		self.time = time.time()
		self.button = button
		self.button_pressed = button_pressed
		self.scroll_dx = scroll_dx
		self.scroll_dy = scroll_dy

	def is_pos_only(self):
		pos_only = True

		if (not (self.button is None)) or (not (self.scroll_dx is None)) or (not (self.scroll_dy is None)):
			pos_only = False

		return pos_only

	def to_json(self):
		rep_dict = {"x": self.x_pos, "y": self.y_pos, "time": self.time}

		if not (self.button is None):
			rep_dict["button"] = self.button
			rep_dict["button_pressed"] = self.button_pressed

		if not (self.scroll_dx is None):
			rep_dict["scroll_dx"] = self.scroll_dx

		if not (self.scroll_dy is None):
			rep_dict["scroll_dy"] = self.scroll_dy

		return rep_dict

	def __str__(self):
		return str(self.to_json())

	def __repr__(self):
		return str(self)


def on_move(x, y):
	if (g_prev_mouse_event.is_pos_only()):
		if ((g_prev_mouse_event.x_pos == x) and (g_prev_mouse_event.y_pos == y)):
			return

	event_obj = MouseEvent(x, y)
	appendMouseEvent(event_obj)


def on_click(x, y, button, pressed):
	event_obj = MouseEvent(x, y, button=button, button_pressed=pressed)
	appendMouseEvent(event_obj)


def on_scroll(x, y, dx, dy):
	event_obj = MouseEvent(x, y, scroll_dx=dx, scroll_dy=dy)
	appendMouseEvent(event_obj)


def appendMouseEvent(event_obj):
	global g_prev_mouse_event
	global g_newEventQueue

	g_prev_mouse_event_lock.acquire()
	g_prev_mouse_event = event_obj
	g_newEventQueue.put(event_obj)
	g_prev_mouse_event_lock.release()


def saveMouseEvents(filename, new_event_queue):
	#global g_mouseEvents
	filepath = "{}.gz".format(filename)

	while True:
		time.sleep(g_log_save_time)

		#Get new mouse events from queue
		new_events = []
		while (not new_event_queue.empty()):
			new_events.append(new_event_queue.get())

		#Skip save if there are no new events
		if (len(new_events) == 0):
			continue

		#Open previous compressed pickle file if it exists
		saved_events = []
		if (os.path.exists(filepath)):
			with gzip.open(filepath, "rb") as prev_events_pickle_file:
				saved_events = pickle.load(prev_events_pickle_file)

		#Write updated compressed pickle with updated event list
		print(type(saved_events))
		print(type(saved_events[-1]))
		with gzip.open(filepath, "wb") as events_pickle_file:
			saved_events += new_events
			pickle.dump(saved_events, events_pickle_file)


def main():
	mouse_controller = pmouse.Controller()

	x_pos, y_pos = mouse_controller.position
	event_obj = MouseEvent(x_pos, y_pos)
	appendMouseEvent(event_obj)

	listener = pmouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
	listener.start()

	data_save_proc = multiprocessing.Process(target=saveMouseEvents, args=("mouse_data.pickle", g_newEventQueue))
	data_save_proc.start()
	data_save_proc.join()
	listener.stop()


if __name__ == '__main__':
	main()
