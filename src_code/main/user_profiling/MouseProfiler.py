import os
import time

import threading
import multiprocessing

from enum import Enum, auto
import pickle
import shutil
import bz2
import gzip
import lzma

import numpy as np
import math

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from pynput import mouse as pmouse
import progressbar


##############################
# Globals
##############################
g_newEventQueue = multiprocessing.Queue()
g_prev_mouse_event_lock = threading.Lock()
g_prev_mouse_event = None
g_num_clicks = 0

##############################
# Mouse data collection
##############################
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
	global g_num_clicks
	g_num_clicks += 1

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


def saveMouseEvents(filename, new_event_queue, kill_process, log_save_time=60):
	filepath = "{}.gz".format(filename)
	backup_filepath = "{}_bak.gz".format(filename)
	new_events = []

	while (not kill_process.is_set()):
		time.sleep(log_save_time/2)

		#Get new mouse events from queue
		while (not new_event_queue.empty()):
			new_events.append(new_event_queue.get())

		#Skip save if there are no new events
		if (len(new_events) == 0):
			if (not kill_process.is_set()):
				time.sleep(log_save_time/2)
			continue

		#Open previous compressed pickle file if it exists
		saved_events = []
		if (os.path.exists(filepath)):
			with gzip.open(filepath, "rb") as prev_events_pickle_file:
				try:
					saved_events = pickle.load(prev_events_pickle_file)
				except:
					shutil.copyfile(backup_filepath, filepath)  #Override corrupted file with backup
					continue

		#Write updated compressed pickle with updated event list
		with gzip.open(filepath, "wb") as events_pickle_file:
			saved_events += new_events
			pickle.dump(saved_events, events_pickle_file)

		#Update backup file
		if (not kill_process.is_set()):
			time.sleep(log_save_time/2)
		shutil.copyfile(filepath, backup_filepath)

		#Clear new events list
		new_events = []


def collectMouseData(mouse_data_filepath="mouse_data.pickle", collection_time_max=1200, target_click_num=1500):
	print("Collecting mouse data...")
	mouse_controller = pmouse.Controller()

	x_pos, y_pos = mouse_controller.position
	event_obj = MouseEvent(x_pos, y_pos)
	appendMouseEvent(event_obj)

	listener = pmouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
	listener.start()

	kill_process = multiprocessing.Event()
	data_save_proc = multiprocessing.Process(target=saveMouseEvents, args=(mouse_data_filepath, g_newEventQueue, kill_process))
	data_save_proc.start()

	widgets = [' [', progressbar.Percentage(), '] ', progressbar.Bar('*')]
	progress_bar = progressbar.ProgressBar(max_value=100, widgets=widgets).start()

	start_time = time.time()
	while True:
		time.sleep(1)
		elapsed_time = time.time() - start_time
		if (g_num_clicks > target_click_num):
			break
		if (elapsed_time > collection_time_max):
			print("\nMax collection time exceeded. Stopping data collection")
			break

		progress = max(int(100*max(elapsed_time/collection_time_max, float(g_num_clicks)/target_click_num))-1, 0)
		progress_bar.update(progress)
	progress_bar.update(100)
	
	listener.stop()
	kill_process.set()
	data_save_proc.join()
	listener.join()

	print("\nMouse data collected")

	return "{}.gz".format(mouse_data_filepath)

##########################################
# Mouse movement analysis
##########################################

class MousePathType(Enum):
	UNKNOWN = auto()
	PRE_CLICK = auto()
	POST_CLICK = auto()
	SCROLL = auto()
	DRAG = auto()
	AIMLESS = auto()


class MouseDataPoint:
	def __init__(self, mouse_event_obj):
		self.x_pos = mouse_event_obj.x_pos
		self.y_pos = mouse_event_obj.y_pos
		self.timestamp = mouse_event_obj.time
		self.button = mouse_event_obj.button
		self.button_pressed = mouse_event_obj.button_pressed
		self.scroll_dx = mouse_event_obj.scroll_dx
		self.scroll_dy = mouse_event_obj.scroll_dy

		self.x_velocity = None
		self.y_velocity = None
		self.velocity_magnitude = None

		self.x_acceleration = None
		self.y_acceleration = None
		self.acceleration_magnitude = None

		self.idle = False

		self.path_type = MousePathType.UNKNOWN

		self.path_start_x = None
		self.path_start_y = None

		self.path_end_x = None
		self.path_end_y = None

		self.straight_path_length = None
		self.end_start_angle = None
		self.end_dist = None


	def calcPathData(self, start_x, start_y, end_x, end_y):
		self.path_start_x = start_x
		self.path_start_y = start_y
		self.path_end_x = end_x
		self.path_end_y = end_y

		dx_start = self.x_pos - start_x
		dy_start = self.y_pos - start_y
		dx_end = end_x - self.x_pos
		dy_end = end_y - self.y_pos

		self.straight_path_length = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
		self.end_dist = np.sqrt((dx_end)**2 + (dy_end)**2)
		self.start_dist = np.sqrt((dx_start)**2 + (dy_start)**2)
		
		self.path_progress = 1-(self.end_dist/self.straight_path_length)
		if (self.path_progress < 0):
			self.path_progress = self.end_dist / (self.end_dist+self.start_dist)

		if (self.end_dist == 0) or (self.start_dist == 0):
			self.end_start_angle = 180
		else:
			self.end_start_angle = np.degrees(calcAngle([start_x,start_y], [self.x_pos,self.y_pos], [end_x,end_y]))
		self.straight_path_deviation = (180-self.end_start_angle)/180


def calcAngle(p_a, p_b, p_c):
	ba = [p_a[i] - p_b[i] for i in range(len(p_a))]
	bc = [p_c[i] - p_b[i] for i in range(len(p_c))]

	cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
	if (cosine_angle < -1):
		cosine_angle = -0.99
	if (cosine_angle > 1):
		cosine_angle = 0.99
	angle = np.arccos(cosine_angle)

	return angle


def cartesianDistance(p1_x, p1_y, p2_x, p2_y):
	return np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)


def getVelocityDatapoints(mouse_events, smoothing_width=15):
	data_points = []
	for indx in range(smoothing_width, len(mouse_events)-smoothing_width):
		x_delta = mouse_events[indx+smoothing_width].x_pos - mouse_events[indx-smoothing_width].x_pos
		y_delta = mouse_events[indx+smoothing_width].y_pos - mouse_events[indx-smoothing_width].y_pos
		t_delta = mouse_events[indx+smoothing_width].time - mouse_events[indx-smoothing_width].time
		
		x_velocity = x_delta/t_delta
		y_velocity = y_delta/t_delta
		velocity_magnitude = np.sqrt(x_velocity**2 + y_velocity**2)

		data_point_obj = MouseDataPoint(mouse_events[indx])
		data_point_obj.x_velocity = x_velocity
		data_point_obj.y_velocity = y_velocity
		data_point_obj.velocity_magnitude = velocity_magnitude

		data_points.append(data_point_obj)

	return data_points


def populateAccelerationData(data_points, smoothing_width=50):
	accel_data_points = []
	for indx in range(smoothing_width, len(data_points)-smoothing_width):
		xv_delta = data_points[indx+smoothing_width].x_velocity - data_points[indx-smoothing_width].x_velocity
		yv_delta = data_points[indx+smoothing_width].y_velocity - data_points[indx-smoothing_width].y_velocity
		t_delta = data_points[indx+smoothing_width].timestamp - data_points[indx-smoothing_width].timestamp

		x_acceleration = xv_delta/t_delta
		y_acceleration = yv_delta/t_delta

		data_point_obj = data_points[indx]
		data_point_obj.x_acceleration = x_acceleration
		data_point_obj.y_acceleration = y_acceleration
		data_point_obj.acceleration_magnitude = np.sqrt(x_acceleration**2 + y_acceleration**2)

		accel_data_points.append(data_point_obj)

	return accel_data_points


def populateIdleData(data_points, smoothing_width=6, velocity_cuttoff=150, acceleration_cuttoff=15000):  #TODO: Scale velocity cuttoff
	idle_data_points = []
	for indx in range(smoothing_width, len(data_points)-smoothing_width):
		all_neighbors_idle = True
		for j in range(indx-smoothing_width, indx+smoothing_width+1):
			all_neighbors_idle = all_neighbors_idle and (data_points[j].velocity_magnitude < velocity_cuttoff) and (data_points[j].acceleration_magnitude < acceleration_cuttoff) and (not data_points[j].button_pressed) and (not data_points[j].scroll_dy)
		
		data_point_obj = data_points[indx]
		data_point_obj.idle = all_neighbors_idle
		idle_data_points.append(data_point_obj)

	return idle_data_points


def splitPaths(data_points, distance_cutoff=50):
	#Determine where each path begins/ends
	path_split_indxs = []
	for indx in range(1, len(data_points)-1):
		split_path = False
		if (data_points[indx].button_pressed and (not data_points[indx-1].button_pressed)):
			split_path = True
		if (data_points[indx].idle and ((not data_points[indx-1].idle) or (not data_points[indx+1].idle))):
			split_path = True
		if ((data_points[indx].timestamp - data_points[indx-1].timestamp) > 0.5):
			split_path = True
		if (not (data_points[indx].scroll_dx is None)) or (not (data_points[indx].scroll_dy is None)):
			split_path = True

		if (split_path):
			path_split_indxs.append(indx)

	if (len(path_split_indxs) < 2):
		return []

	#Split data point list into paths
	paths = []
	for j in range(1,len(path_split_indxs)):
		start_indx = path_split_indxs[j-1]
		end_indx = path_split_indxs[j]

		paths.append(data_points[start_indx:end_indx+1])

	#Filter paths based on length
	filtered_paths = []
	for path in paths:
		start_x = path[0].x_pos
		start_y = path[0].y_pos

		end_x = path[-1].x_pos
		end_y = path[-1].y_pos

		straight_distance = np.sqrt((start_x-end_x)**2 + (start_y-end_y)**2)
		if (straight_distance > distance_cutoff):
			filtered_paths.append(path)

	return filtered_paths


def populatePathData(data_points):
	path_list = splitPaths(data_points)
	filtered_datapoints = []
	for path in path_list:
		filtered_datapoints += path

		#Get start and end coord of path
		start_point = path[0]
		end_point = path[-1]

		start_x = start_point.x_pos
		start_y = start_point.y_pos

		end_x = end_point.x_pos
		end_y = end_point.y_pos

		#Determine path type
		path_type = MousePathType.UNKNOWN
		if ((not start_point.button_pressed) and end_point.button_pressed):
			path_type = MousePathType.PRE_CLICK
		if (start_point.button_pressed and end_point.idle):
			path_type = MousePathType.POST_CLICK
		if (start_point.idle and end_point.idle):
			path_type = MousePathType.AIMLESS
		if (start_point.button_pressed and end_point.button_pressed):
			path_type = MousePathType.DRAG

		#Populate path data for all points in path
		for data_point_obj in path:
			data_point_obj.calcPathData(start_x, start_y, end_x, end_y)
			data_point_obj.path_type = path_type

	return filtered_datapoints

def filterPaths(data_points, target_path_type):
	filtered_data = []
	for data_point_obj in data_points:
		if (data_point_obj.path_type == target_path_type):
			filtered_data.append(data_point_obj)

	return filtered_data
		

##########################################
# Mouse profile creation
##########################################

def generateMouseProfile(mouse_data_filepath, mouse_profile_path="bespoke_mouse_profile.pickle"):
	#Get mouse events
	mouse_events = []
	print(mouse_data_filepath)
	with gzip.open(mouse_data_filepath, "rb") as prev_events_pickle_file:
		mouse_events = pickle.load(prev_events_pickle_file)

	#Populate datapoints from mouse events
	print("Extracting mouse data points")
	data_points = getVelocityDatapoints(mouse_events)
	data_points = populateAccelerationData(data_points)
	data_points = populateIdleData(data_points)
	data_points = populatePathData(data_points)

	#Only fit movement model on PRE_CLICK paths
	data_points = filterPaths(data_points, MousePathType.PRE_CLICK)

	#Remove upper 1% of velocity data points
	velocities = [i.velocity_magnitude for i in data_points]
	velocities.sort()
	max_v_indx = math.floor(len(velocities)*0.99)
	max_velocity = velocities[max_v_indx]
	data_points = [i for i in data_points if (i.velocity_magnitude < max_velocity)]
	
	# Generate polynomial features for velocity model fit
	data_in = np.array([[i.path_progress, i.end_start_angle, i.straight_path_length, i.straight_path_deviation] for i in data_points])
	velocity_to_fit = np.array([i.velocity_magnitude for i in data_points])
	poly = PolynomialFeatures(degree=3)
	data_in_poly = poly.fit_transform(data_in)

	# Fit the model
	print("Fitting mouse velocity model")
	velocity_model = LinearRegression()
	velocity_model.fit(data_in_poly, velocity_to_fit)

	# Make predictions
	v_predicted = velocity_model.predict(data_in_poly)

	# Calculate the R-squared score
	r2 = r2_score(velocity_to_fit, v_predicted)
	print("R2 = {}".format(r2))

	#Determine model mismatch distribution
	diff_ratios = (velocity_to_fit - v_predicted)/v_predicted
	diff_ratios.sort()
	lower_bound_indx = math.ceil(len(diff_ratios)*0.03)
	upper_bound_indx = math.floor(len(diff_ratios)*0.97)
	diff_ratios = diff_ratios[lower_bound_indx:upper_bound_indx]
	v_diff_ratio_avg = np.average(diff_ratios)
	v_diff_ratio_std = np.std(diff_ratios)

	#Save model to file
	#Currently use pickle rather than json since I need to save the LinearRegression object. Need to find a way to spawn a regression object using known coefficients
	print("Writing mouse profile")
	mouse_profile = {}
	mouse_profile["coefficients"] = velocity_model.coef_
	mouse_profile["intercept"] = velocity_model.intercept_
	mouse_profile["v_diff_ratio_avg"] = v_diff_ratio_avg
	mouse_profile["v_diff_ratio_std"] = v_diff_ratio_std
	mouse_profile["model"] = velocity_model

	with open(mouse_profile_path, "wb") as mouse_profile_pickle_file:
		pickle.dump(mouse_profile, mouse_profile_pickle_file)


def createUserProfile():
	#Get current path
	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_output_dir = os.path.join(script_dir, "data")
	profile_output_dir = os.path.join(script_dir, "profiles")

	#Collect mouse data
	mouse_data_filepath = os.path.join(data_output_dir, "mouse_data_new.pickle")
	mouse_data_filepath = collectMouseData(mouse_data_filepath=mouse_data_filepath)

	#Create mouse profile
	mouse_profile_path = os.path.join(profile_output_dir, "bespoke_mouse_profile_new.pickle")
	generateMouseProfile(mouse_data_filepath, mouse_profile_path=mouse_profile_path)

if __name__ == '__main__':
	mouse_data_filepath = collectMouseData()
	# start_time = time.time()
	# mouse_data_filepath = "mouse_data.pickle.gz"
	# generateMouseProfile(mouse_data_filepath)
	# elapsed_time = time.time() - start_time
	# print(int(elapsed_time))
