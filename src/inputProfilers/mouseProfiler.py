import os
import time
from pynput import mouse as pmouse
import threading
import pickle
import shutil
import bz2
import gzip
import lzma


g_mouseEvents = []
g_mouseEvents_lock = threading.Lock()
g_prevPos = (0,0)
g_prevPos_lock = threading.Lock()
g_log_save_time = 120


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
	g_mouseEvents_lock.acquire()
	prev_event = g_mouseEvents[-1]
	g_mouseEvents_lock.release()
	if (prev_event.is_pos_only()):
		if ((prev_event.x_pos == x) and (prev_event.y_pos == y)):
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
	global g_mouseEvents

	g_mouseEvents_lock.acquire()
	g_mouseEvents.append(event_obj)
	g_mouseEvents_lock.release()


def saveMouseEvents(file_name):
	global g_mouseEvents

	filepath = "{}.gz".format(file_name)

	#Skip save if there are no new events
	if (len(g_mouseEvents) < 2):
		return

	#Copy previous pickle dump into backup file
	if (os.path.exists(filepath)):
		backup_path = "{}_bak.gz".format(file_name)
		shutil.copyfile(filepath, backup_path)
		time.sleep(1)

	#Open previous pickle file if it exists
	saved_events = []
	if (os.path.exists(filepath)):
		with gzip.open(filepath, "rb") as events_pickle_file:
			saved_events = pickle.load(events_pickle_file)

	#Write updated pickle with updated event list
	saved_events = []
	with gzip.open(filepath, "wb") as events_pickle_file:
		g_mouseEvents_lock.acquire()
		saved_events += g_mouseEvents
		pickle.dump(saved_events, events_pickle_file)
		g_mouseEvents = [g_mouseEvents[-1]]
		g_mouseEvents_lock.release()


def main():
	mouse_controller = pmouse.Controller()

	x_pos, y_pos = mouse_controller.position
	event_obj = MouseEvent(x_pos, y_pos)
	appendMouseEvent(event_obj)

	listener = pmouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
	listener.start()

	while (True):
		time.sleep(g_log_save_time)
		saveMouseEvents("mouse_data.pickle")

	listener.stop()

main()
#saveMouseEvents("mouse_log.pickle")