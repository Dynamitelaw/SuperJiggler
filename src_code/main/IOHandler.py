##########################################
# Mouse movement generator
##########################################

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