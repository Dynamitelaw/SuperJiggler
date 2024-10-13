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


def cartesianToPolar(x, y):
	rho = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)

	return rho, phi

def polarToCartesian(rho, phi):
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return(x, y)

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
		self.velocity_rho = None
		self.velocity_phi = None
		self.velocity_end_phi = None

		self.x_acceleration = None
		self.y_acceleration = None
		self.acceleration_magnitude = None
		self.acceleration_rho = None
		self.acceleration_phi = None
		self.acceleration_end_phi = None

		self.idle = False

		self.path_type = MousePathType.UNKNOWN

		self.path_start_x = None
		self.path_start_y = None
		self.path_start_rho = None
		self.path_start_phi = None

		self.path_end_x = None
		self.path_end_y = None
		self.path_end_rho = None
		self.path_end_phi = None
		self.straight_path_length = None
		self.end_start_angle = None
		self.end_dist = None

		self.curvature = 0

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

		self.path_start_rho, self.path_start_phi = cartesianToPolar(dx_start, dy_start)
		self.path_end_rho, self.path_end_phi = cartesianToPolar(dx_end, dy_end)

		self.velocity_rho, self.velocity_phi = cartesianToPolar(self.x_velocity, self.y_velocity)
		self.acceleration_rho, self.acceleration_phi = cartesianToPolar(self.x_acceleration, self.y_acceleration)

		
		self.path_progress = 1-(self.end_dist/self.straight_path_length)
		if (self.path_progress < 0):
			self.path_progress = self.path_end_rho / (self.path_end_rho+self.path_start_rho)

		if (self.end_dist == 0) or (self.start_dist == 0):
			self.end_start_angle = 180
		else:
			self.end_start_angle = np.degrees(calcAngle([start_x,start_y], [self.x_pos,self.y_pos], [end_x,end_y]))
		self.straight_path_deviation = (180-self.end_start_angle)/180

		self.velocity_end_phi = self.velocity_phi - self.path_end_phi
		self.acceleration_end_phi = self.acceleration_phi - self.path_end_phi



def calcMengerCurvature(p1, p2, p3):
	"""Calculates the Menger curvature of three points.

	Args:
	    p1, p2, p3: 2D or 3D points represented as tuples or lists.

	Returns:
	    The Menger curvature (reciprocal of the circumradius).
	"""

	x1, y1 = p1.x_pos, p1.y_pos
	x2, y2 = p2.x_pos, p2.y_pos
	x3, y3 = p3.x_pos, p3.y_pos

	# Calculate the area of the triangle formed by the three points
	area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

	# Calculate the side lengths of the triangle
	a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
	b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
	c = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)

	# If the area is zero, the points are collinear, and the curvature is zero
	if area == 0:
	    return 0

	# Calculate the Menger curvature
	curvature = (4 * area) / (a * b * c)

	return curvature


def calcAngle(p_a, p_b, p_c):
	ba = [p_a[i] - p_b[i] for i in range(len(p_a))]
	bc = [p_c[i] - p_b[i] for i in range(len(p_c))]

	cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
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


def populateCurvatureData(data_points, smoothing_width=20):
	for indx in range(smoothing_width, len(data_points)-smoothing_width):
		curvature = calcMengerCurvature(data_points[indx-smoothing_width], data_points[indx], data_points[indx+smoothing_width])
		data_points[indx].curvature = curvature

	return data_points


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
		if (data_point_obj.path_type == target_type):
			filtered_data.append(data_point_obj)

	return filtered_data
		

def getNormalSample(mean, std, onlyPositive=True):
	'''
	Returns a single sample from a normal distribution
	'''
	sample = np.random.normal(mean, std, 1)[0]
	if (onlyPositive):
		while (sample < 0):
			sample = np.random.normal(mean, std, 1)[0]
	return sample


def getNextMousePosition(start_x, start_y, target_x, target_y, current_x, current_y, current_vx, current_vy):
	start_dist = np.sqrt((current_x-start_x)**2 + (current_y-start_y)**2)
	target_dist, target_phi = cartesianToPolar((target_x-current_x), (target_y-current_y))

	path_progress = start_dist/(start_dist+target_dist)
	path_length = np.sqrt((target_x-start_x)**2 + (target_y-start_y)**2)

	scale_factor = 0.1
	a_mag = (path_length -2*path_length*path_progress)*scale_factor
	print((path_progress, a_mag))
	a_mag = a_mag*getNormalSample(1, 0.01)
	a_phi = target_phi*getNormalSample(1, 0.03)
	a_x, a_y = polarToCartesian(a_mag, a_phi)

	next_vx = current_vx + a_x
	next_vy = current_vy + a_y
	next_x = current_x+next_vx
	next_y = current_y+next_vy

	return next_x, next_y, next_vx, next_vy


class MousePathGenerator(object):
	"""docstring for ClassName"""
	def __init__(self):
		self.start_x = 0
		self.start_y = 0

		self.target_x = 0
		self.target_y = 0

		self.current_x = 0
		self.current_y = 0

		self.path_length = 0

		self.v_x = 0
		self.v_y = 0

		self.wind_x = 0
		self.wind_y = 0
		self.wind_v_dampening = np.sqrt(3)
		self.wind_a_dampening = np.sqrt(5)

		self.GRAV_0 = 9
		self.WIND_0 = 100
		self.M_0 = 30
		self.D_0 = 12

		self.last_iter_time = 0

	def startNewPath(self, start_x, start_y, target_x, target_y):
		self.start_x = start_x
		self.start_y = start_y

		self.target_x = target_x
		self.target_y = target_y

		self.path_length = cartesianDistance(start_x, start_y, target_x, target_y)

		self.last_iter_time = 0

		self.current_x = start_x
		self.current_y = start_y

	def getNextPosition(self):
		start_dist = cartesianDistance(self.current_x, self.current_y, self.start_x, self.start_y)
		target_dist = cartesianDistance(self.current_x, self.current_y, self.target_x, self.target_y)

		path_progress = 1-(target_dist/self.path_length)
		target_start_angle = np.degrees(calcAngle([1,0], [0,0], [1,40]))
		if (target_start_angle < 0):
			raise ValueError("Nergative angle!")
		path_deviation = (180-target_start_angle)/180

		W_mag = min(self.WIND_0*(1-path_deviation), target_dist)
		self.wind_x = self.wind_x/self.wind_v_dampening + (2*np.random.random()-1)*W_mag/self.wind_a_dampening
		self.wind_y = self.wind_y/self.wind_v_dampening + (2*np.random.random()-1)*W_mag/self.wind_a_dampening
		'''
		if target_dist >= self.D_0:
			self.wind_x = self.wind_x/self.wind_v_dampening + (2*np.random.random()-1)*W_mag/self.wind_a_dampening
			self.wind_y = self.wind_y/self.wind_v_dampening + (2*np.random.random()-1)*W_mag/self.wind_a_dampening
		else:
			self.wind_x = self.wind_x/self.wind_v_dampening
			self.wind_y = self.wind_y/self.wind_v_dampening
			if self.M_0 < 3:
				self.M_0 = np.random.random()*3 + 3
			else:
				self.M_0 = self.M_0/np.sqrt(5)
		'''
		grav_x = (self.GRAV_0*(self.target_x-self.current_x))/(target_dist*(1-path_progress)*(1-path_deviation))
		grav_y = (self.GRAV_0*(self.target_y-self.current_y))/(target_dist*(1-path_progress)*(1-path_deviation))
		self.v_x += self.wind_x + grav_x
		self.v_y += self.wind_y + grav_y
		v_mag = np.hypot(self.v_x, self.v_y)
		v_max = 50*(1.25 - (4*(0.5-path_progress)**2)) + 20*path_deviation
		#v_max = self.M_0
		if v_mag > v_max:
			v_clip = v_max/2 + np.random.random()*v_max/2
			self.v_x = (self.v_x/v_mag) * v_clip
			self.v_y = (self.v_y/v_mag) * v_clip

		self.current_x += self.v_x
		self.current_y += self.v_y

		return self.current_x, self.current_y

		
def genMousePath(start_x, start_y, target_x, target_y):
	current_x = start_x
	current_y = start_y

	points = []
	pathGen = MousePathGenerator()
	pathGen.startNewPath(start_x, start_y, target_x, target_y)
	for i in range(200):
		points.append((current_x, current_y))
		target_dist = np.sqrt((current_x-target_x)**2 + (current_y-target_y)**2)
		if (target_dist < 15):
			break

		current_x, current_y = pathGen.getNextPosition()

	return points


def main():
	#Get mouse events
	mouse_events = []
	#filepath = "mouse_data_small.pickle.gz"
	filepath = "mouse_data.pickle_bak - Copy.gz"
	with gzip.open(filepath, "rb") as prev_events_pickle_file:
		mouse_events = pickle.load(prev_events_pickle_file)

	#Calculate datapoints from mouse events
	sample_size = 80000
	data_points = getVelocityDatapoints(mouse_events[0:sample_size])
	data_points = populateAccelerationData(data_points)
	data_points = populateCurvatureData(data_points)
	data_points = populateIdleData(data_points)
	data_points = populatePathData(data_points)

	#Only fit movement model on PRE_CLICK paths
	data_points = filterPaths(data_points, MousePathType.PRE_CLICK)
	

	# Generate polynomial features up to degree 2
	x_to_fit = np.array([[0.5-i.path_progress, i.end_start_angle, i.straight_path_length, i.straight_path_deviation] for i in data_points])
	y_to_fit = np.array([i.velocity_magnitude for i in data_points])
	poly = PolynomialFeatures(degree=2)
	X_poly = poly.fit_transform(x_to_fit)

	# Fit the model
	model = LinearRegression()
	model.fit(X_poly, y_to_fit)

	# Make predictions
	y_pred = model.predict(X_poly)

	# Calculate the R-squared score
	r2 = r2_score(y_to_fit, y_pred)
	diff = y_to_fit - y_pred
	plt.hist(diff, bins=20)
	plt.show() 

	# Print the R-squared score
	print("R-squared:", r2)

	# Extract coefficients and intercept
	coefficients = model.coef_
	intercept = model.intercept_

	# Construct and print the equation
	equation = f"v_mag = {intercept:.2f} + {coefficients[0]:.2f} * x1 + {coefficients[1]:.2f} * x2"
	print(coefficients)
	#print(equation)
	#sys.exit()

	# fig, axs = plt.subplots(2)
	# fig.suptitle('Vertically stacked subplots')
	# axs[0].scatter(path_progress, v_mag)
	# axs[1].scatter(path_progress, accel_mag)

	# plt.scatter(path_progress, curvature)

	fig = plt.figure(figsize=(12, 12))
	ax = fig.add_subplot(projection='3d')
	ax.scatter(path_length, path_progress, y_pred, color="r")
	ax.scatter(path_length, path_progress, v_mag, color="b")
	ax.set_xlabel("path_length")
	ax.set_ylabel("path_progress")
	ax.set_zlabel("y_pred")
	#plt.show()

	colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
	color_indx = 0
	for path in paths:
		start_x = path[0].x_pos
		start_y = path[0].y_pos
		x = np.array([i.x_pos-start_x for i in path])
		y = np.array([i.y_pos-start_y for i in path])
		#plt.plot(x, y, "tab:{}".format(colors[color_indx]))
		color_indx = (color_indx+1)%(len(colors))

	#plt.show()
	sys.exit()

	fig, ax = plt.subplots(2)
	t = np.linspace(0, 3, 40)
	g = -9.81
	v0 = 12
	z = g * t**2 / 2 + v0 * t

	v02 = 5
	z2 = g * t**2 / 2 + v02 * t

	data_points = paths[4] #2
	path_x = np.array([i.x_pos for i in data_points])
	path_y = np.array([i.y_pos for i in data_points])
	path_plot = ax[0].plot(path_x, path_y)
	current_data = data_points[0]
	current_pos = ax[0].scatter([current_data.x_pos], [current_data.y_pos])

	telemetry_scale = 10000
	current_v_telemetry = ax[1].quiver([0, 0], current_data.x_velocity, current_data.y_velocity)
	current_a_telemetry = ax[1].quiver([0, 0], current_data.x_acceleration, current_data.y_acceleration)
	ax[1].set_xlim(-1*telemetry_scale, telemetry_scale)
	ax[1].set_ylim(-1*telemetry_scale, telemetry_scale)
	#ax.legend()

	plt.quiver([0, 0, 0], [0, 0, 0], [1, -2, 4], [1, 2, -7], angles='xy', scale_units='xy', scale=1)
	def update(frame):
		#current_pos.set_xdata([data_points[frame].x_pos])
		#current_pos.set_ydata([data_points[frame].y_pos])
		current_data = data_points[frame]
		data = np.hstack(([current_data.x_pos], [current_data.y_pos]))
		current_pos.set_offsets(data)
		current_v_telemetry.set_UVC(current_data.x_velocity, current_data.y_velocity)
		current_a_telemetry.set_UVC(current_data.x_acceleration, current_data.y_acceleration)
		return (path_plot, current_pos, current_v_telemetry, current_a_telemetry)


	ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data_points), interval=5)
	plt.show()

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

angle = calcAngle([1,0], [0,0], [1,40])
print(np.degrees(angle))
#sys.exit()

#cs, points = generateSplinePath(0,0,10,10)
points = genMousePath(1500,1500,200,300)

x_pos = [i[0] for i in points]
y_pos = [i[1] for i in points]

t = []
v = []
for indx in range(1,len(points)):
	t.append(indx)
	v_x = x_pos[indx] - x_pos[indx-1]
	v_y = y_pos[indx] - y_pos[indx-1]
	v_mag = np.sqrt(v_x**2 + v_y**2)
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