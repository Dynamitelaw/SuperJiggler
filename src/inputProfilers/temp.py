from mouseProfiler import *
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import math
import matplotlib.animation as animation
from enum import Enum, auto
import sys
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle



def cartesianDistance(p1_x, p1_y, p2_x, p2_y):
	return np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)

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
		

class MousePathGenerator(object):
	"""docstring for ClassName"""
	def __init__(self):
		self.start_x = 0
		self.start_y = 0

		self.target_x = 0
		self.target_y = 0

		self.target_start_angle = 0

		self.current_x = 0
		self.current_y = 0

		self.path_length = 0

		self.v_x = 0
		self.v_y = 0

		self.wind_x = 0
		self.wind_y = 0
		self.wind_v_dampening = np.sqrt(3)
		self.wind_a_dampening = np.sqrt(5)

		self.GRAV_0 = 800
		#self.WIND_0 = 4000
		self.WIND_0 = 3000

		self.last_iter_time = 0
		self.path_progress = 0

		mouse_profile_path = "mouse_profile.pickle"
		with open(mouse_profile_path, "rb") as mouse_profile_pickle_file:
			mouse_profile = pickle.load(mouse_profile_pickle_file)
			self.velocity_model = mouse_profile["model"]
			self.v_diff_ratio_avg = mouse_profile["v_diff_ratio_avg"]
			self.v_diff_ratio_std = mouse_profile["v_diff_ratio_std"]
	
		self.poly = PolynomialFeatures(degree=3)
		self.v_diff_ratio = 0

	def startNewPath(self, start_x, start_y, target_x, target_y):
		self.start_x = start_x
		self.start_y = start_y

		self.target_x = target_x
		self.target_y = target_y

		self.path_length = cartesianDistance(start_x, start_y, target_x, target_y)

		self.last_iter_time = 0

		self.current_x = start_x
		self.current_y = start_y

		self.v_diff_ratio = max(np.random.normal(self.v_diff_ratio_avg, self.v_diff_ratio_std, 1)[0], -0.8)

	def getNextPosition(self):
		start_dist = cartesianDistance(self.current_x, self.current_y, self.start_x, self.start_y)
		target_dist = cartesianDistance(self.current_x, self.current_y, self.target_x, self.target_y)

		self.path_progress = 1-(target_dist/self.path_length)
		if (self.path_progress < 0):
			self.path_progress = target_dist/(target_dist+start_dist)

		if (start_dist == 0) or (target_dist == 0):
			self.target_start_angle = 180
		else:
			self.target_start_angle = np.degrees(calcAngle([self.start_x,self.start_y], [self.current_x,self.current_y], [self.target_x,self.target_y]))

		self.path_deviation = (180-self.target_start_angle)/180

		W_mag = min(self.WIND_0*(1-self.path_deviation), target_dist*100)
		self.wind_x = self.wind_x/self.wind_v_dampening + (2*np.random.random()-1)*W_mag/self.wind_a_dampening
		self.wind_y = self.wind_y/self.wind_v_dampening + (2*np.random.random()-1)*W_mag/self.wind_a_dampening

		grav_x = (self.GRAV_0*(self.target_x-self.current_x))/(target_dist*(1-self.path_progress)*(1-self.path_deviation))
		grav_y = (self.GRAV_0*(self.target_y-self.current_y))/(target_dist*(1-self.path_progress)*(1-self.path_deviation))

		self.v_x += self.wind_x + grav_x
		self.v_y += self.wind_y + grav_y

		v_mag = np.hypot(self.v_x, self.v_y)
		v_max = self.velocity_model.predict(self.poly.fit_transform([[self.path_progress, self.target_start_angle, self.path_length, self.path_deviation]]))[0]
		v_max = v_max*(1+self.v_diff_ratio)
		if (v_max < 50):
			v_max = 50

		if v_mag > v_max:
			v_clip = v_max/2 + np.random.random()*v_max/2
			self.v_x = (self.v_x/v_mag) * v_clip
			self.v_y = (self.v_y/v_mag) * v_clip

		if (self.last_iter_time == 0):
			self.last_iter_time = time.time()
		else:
			current_time = time.time()
			time_delta = current_time - self.last_iter_time

			self.current_x += self.v_x*time_delta
			self.current_y += self.v_y*time_delta

			self.last_iter_time = current_time

		return self.current_x, self.current_y

		
def genMousePath(start_x, start_y, target_x, target_y):
	current_x = start_x
	current_y = start_y

	points = []
	pathGen = MousePathGenerator()
	pathGen.startNewPath(start_x, start_y, target_x, target_y)
	for i in range(20000):
		points.append((current_x, current_y))
		time.sleep(0.001)
		target_dist = np.sqrt((current_x-target_x)**2 + (current_y-target_y)**2)
		if (target_dist < 15):
			break

		current_x, current_y = pathGen.getNextPosition()

	return points


def main():
	#Get mouse events
	mouse_events = []
	#filepath = "mouse_data_small.pickle.gz"
	#filepath = "mouse_data.pickle_bak - Copy.gz"
	filepath = "mouse_data.pickle.gz"
	with gzip.open(filepath, "rb") as prev_events_pickle_file:
		mouse_events = pickle.load(prev_events_pickle_file)

	#Populate datapoints from mouse events
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
	velocity_model = LinearRegression()
	velocity_model.fit(data_in_poly, velocity_to_fit)

	# Make predictions
	v_predicted = velocity_model.predict(data_in_poly)

	# Calculate the R-squared score
	r2 = r2_score(velocity_to_fit, v_predicted)

	#Determine mismatch distribution
	diff_ratios = (velocity_to_fit - v_predicted)/v_predicted
	diff_ratios.sort()
	lower_bound_indx = math.ceil(len(diff_ratios)*0.03)
	upper_bound_indx = math.floor(len(diff_ratios)*0.97)
	diff_ratios = diff_ratios[lower_bound_indx:upper_bound_indx]
	plt.hist(diff_ratios, bins=80)
	plt.show() 
	v_diff_ratio_avg = np.average(diff_ratios)
	v_diff_ratio_std = np.std(diff_ratios)

	#Save model to file
	mouse_profile = {}
	mouse_profile["coefficients"] = velocity_model.coef_
	mouse_profile["intercept"] = velocity_model.intercept_
	mouse_profile["v_diff_ratio_avg"] = v_diff_ratio_avg
	mouse_profile["v_diff_ratio_std"] = v_diff_ratio_std
	mouse_profile["model"] = velocity_model

	mouse_profile_path = "mouse_profile.pickle"  #currently use pickle rather than json since I need to save the LinearRegression object. Need to find a way to spawn a regression object using known coefficients
	with open(mouse_profile_path, "wb") as mouse_profile_pickle_file:
		pickle.dump(mouse_profile, mouse_profile_pickle_file)


main()

# start_pos = (0,0,0)
# mid_pos = (0.3,-1,-5)
# other_mid_pos = (0.7,2,-8)
# end_pos = (1,-10,-10)

# points = [start_pos, mid_pos, other_mid_pos, end_pos]
# path_progress = [i[0] for i in points]
# x_pos = [i[1] for i in points]
# y_pos = [i[2] for i in points]
#https://ben.land/post/2021/04/25/windmouse-human-mouse-movement/

while True:
	points = genMousePath(200,300,1300,1400)

	x_pos = [i[0] for i in points]
	y_pos = [i[1] for i in points]

	t = []
	v = []
	for indx in range(1,len(points)):
		t.append(indx*0.001)
		v_x = x_pos[indx] - x_pos[indx-1]
		v_y = y_pos[indx] - y_pos[indx-1]
		v_mag = np.sqrt(v_x**2 + v_y**2)/0.001
		v.append(v_mag)

	fig, ax = plt.subplots(2)
	ax[0].plot(x_pos, y_pos)
	ax[1].plot(t, v)
	plt.show()


'''
Current observed patters
PRE_CLICK:
	v_mag = f(path_progress, path_length) ~= path_length*(-1*(path_progress-0.5)**2)
	curvature = f(path_progress) ~= A(path_progress-0.5)**2) + B
	curvature = f(v_mag) ~= A/v_mag


'''