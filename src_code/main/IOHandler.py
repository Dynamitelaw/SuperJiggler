from pynput import keyboard, mouse
from pynput.mouse import Button
import pickle
import json
import os
import time
from scipy import stats as scistats
import random
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from ScreenHandler import *


class KeyboardHandler():
	def __init__(self, local_machine=False):
		self.local_machine = local_machine

		self.keyboard_controller = None
		if (self.local_machine):
			self.keyboard_controller = keyboard.Controller()
		else:
			raise ValueError("Not implemented yet")

		script_dir = os.path.dirname(os.path.abspath(__file__))
		profile_dir = os.path.join(script_dir, "user_profiling", "profiles")

		typing_profile_path = None
		if (os.path.exists(os.path.join(profile_dir, "bespoke_typing_profile.json"))):
			typing_profile_path = os.path.join(profile_dir, "bespoke_typing_profile.json")
		elif (os.path.exists(os.path.join(profile_dir, "generic_typing_profile.json"))):
			typing_profile_path = os.path.join(profile_dir, "generic_typing_profile.json")
		else:
			raise FileNotFoundError("Cannot find a typing profile")

		typing_profile_file = open(typing_profile_path, "r")
		self.typing_profile = json.load(typing_profile_file)
		typing_profile_file.close()

		self.key_shift_dict = {
			"!": "1",
			"@": "2",
			"#": "3",
			"$": "4",
			"%": "5",
			"^": "6",
			"&": "7",
			"*": "8",
			"(": "9",
			")": "0",
			"_": "-",
			"+": "=",
			"{": "[",
			"}": "]",
			":": ";",
			"\"": "'",
			"<": ",",
			">": ".",
			"?": "/",
			"~": "`"
		}

		self.char_replace_dict = {
			" ": "space",
			"\n": "enter",
			"\t": "tab"
		}

		self.local_key_replace_dict = {
			"backspace" : keyboard.Key.backspace,
			"caps lock" : keyboard.Key.caps_lock,
			"enter" : keyboard.Key.enter,
			"esc" : keyboard.Key.esc,
			"space" : keyboard.Key.space,
			"tab" : keyboard.Key.tab,
			"shift" : keyboard.Key.shift,
			"right shift" : keyboard.Key.shift_r,
			"alt" : keyboard.Key.alt,
			"right alt" : keyboard.Key.alt_r,
			"ctrl" : keyboard.Key.ctrl,
			"right ctrl" : keyboard.Key.ctrl_r,
		}


	def getKeyDelays(self, key_name, prev_key=None):
		#Get delay distributions
		hit_delay_dist = self.typing_profile["overall"]["hit_delay"]
		hold_time_dist = self.typing_profile["overall"]["hold_time"]

		if (key_name in self.typing_profile["per_key"]):
			hit_delay_dist = self.typing_profile["per_key"][key_name]["hit_delay"]["overall"]
			hold_time_dist = self.typing_profile["per_key"][key_name]["hold_time"]
			if (not (prev_key is None)) and (prev_key in self.typing_profile["per_key"][key_name]["hit_delay"]["per_prev_key"]):
				hit_delay_dist = self.typing_profile["per_key"][key_name]["hit_delay"]["per_prev_key"][prev_key]

		#Sample distributions for delays
		hit_delay = -1
		while(hit_delay < 0):
			hit_delay = min(scistats.alpha.rvs(hit_delay_dist["alpha"], hit_delay_dist["loc"], hit_delay_dist["scale"], size=1)[0], 1.5)


		hold_time = -1
		while(hold_time < 0):
			hold_time = min(scistats.alpha.rvs(hold_time_dist["alpha"], hold_time_dist["loc"], hold_time_dist["scale"], size=1)[0], 0.5)


		return hit_delay, hold_time


	def genKeypressSequence(self, text):
		#Convert text block into sequence of key presses
		letterSequence = list(text)

		shiftPressed = False
		keyPresses = []
		for l in letterSequence:
			key_name = l
			if (l.isupper()):
				key_name = l.lower()
				if (not shiftPressed):
					keyPresses.append("SHIFT_PRESS")
					shiftPressed = True
				keyPresses.append(key_name)

			elif (key_name in self.key_shift_dict):
				if (not shiftPressed):
					keyPresses.append("SHIFT_PRESS")
					shiftPressed = True

				key_name = self.key_shift_dict[key_name]
				keyPresses.append(key_name)

			elif (key_name in self.char_replace_dict):
				if (shiftPressed and l.islower()):
					keyPresses.append("SHIFT_RELEASE")
					shiftPressed = False

				key_name = self.char_replace_dict[key_name]
				keyPresses.append(key_name)

			else:
				if (shiftPressed and l.islower()):
					keyPresses.append("SHIFT_RELEASE")
					shiftPressed = False

				keyPresses.append(key_name)

		return keyPresses


	def insertTypingErrors(self, keyPresses):
		#Insert typing errors
		keyPressesImperfect = []
		error_rate = self.typing_profile["overall"]["error_rate"]
		error_length_dist = self.typing_profile["overall"]["error_overshoot_len"]
		for indx in range(len(keyPresses)):
			key_name = keyPresses[indx]
			if not (key_name in self.typing_profile["per_key"]):
				keyPressesImperfect.append(key_name)
				continue

			#Ensure this key has potential miss keys
			error_keys = self.typing_profile["per_key"][key_name]["error_keys"]
			if (len(error_keys) == 0):
				keyPressesImperfect.append(key_name)
				continue

			#Determine if we insert an error on before this key press
			insert_error = (random.random() < error_rate)
			if (not insert_error):
				keyPressesImperfect.append(key_name)
				continue

			#Determine error length and error type
			error_length = min(1, int(np.random.normal(error_length_dist["avg"], error_length_dist["std"], 1)))

			error_type_seed = random.random()
			error_type = None
			#52.9% substitution, 25.6% omission, 21.5% insertion #Dhakal, V.; Feit, A.M.; Kristensson, P.O.; Oulasvirta, A. Observations on typing from 136 million keystrokes. In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems (CHI’18), Montreal, QC, Canada, 21–26 April 2018; pp. 6.
			if (error_type_seed < 0.215):
				error_type = "insertion"
			elif (error_type_seed < 0.471):
				error_type = "omission"
			else:	
				error_type = "substitution"

			#Generate error key presses
			remaining_text_len = len(keyPresses) - indx - 1
			error_presses = []
			if (error_type == "insertion"):
				error_presses.append(random.choice(error_keys))
				error_presses.append(key_name)
				remaining_err_len = error_length - 2
				if (remaining_err_len > 0) and (remaining_err_len < remaining_text_len):
					error_presses += [k for k in keyPresses[indx+1:indx+1+remaining_err_len] if not ("SHIFT" in k)]

			if (error_type == "omission") and (remaining_text_len > error_length):
				error_presses += [k for k in keyPresses[indx+1:indx+1+error_length] if not ("SHIFT" in k)]

			if (error_type == "substitution"):
				error_presses.append(random.choice(error_keys))
				remaining_err_len = error_length - 1
				if (remaining_err_len > 0) and (remaining_err_len < remaining_text_len):
					error_presses += [k for k in keyPresses[indx+1:indx+1+remaining_err_len] if not ("SHIFT" in k)]

			#Correct error
			num_backspace = len(error_presses)
			for i in range(num_backspace):
				error_presses.append("backspace")
			
			keyPressesImperfect += error_presses
			keyPressesImperfect.append(key_name)


		return keyPressesImperfect


	def genKeypressTimes(self, keyPresses):
		#Generate key delays and hold times
		key_times = []
		for keyIndx in range(len(keyPresses)):
			key_name = keyPresses[keyIndx].replace("SHIFT_PRESS", "shift")
			prev_key = None
			if (keyIndx > 0):
				prev_key = keyPresses[keyIndx-1].replace("SHIFT_PRESS", "shift")

			hit_delay, hold_time = self.getKeyDelays(key_name, prev_key=prev_key)
			key_times.append((key_name, hit_delay, hold_time))

		#Generate list of key events with absolute timestamps starting at 0
		key_events = []
		current_time = 0
		for key_press in key_times:
			key_name, hit_delay, hold_time = key_press
			current_time += hit_delay
			if (key_name == "shift"):
				press_event = (current_time, "shift", "press")
				key_events.append(press_event)
			elif (key_name == "SHIFT_RELEASE"):
				release_event = (current_time, "shift", "release")
				key_events.append(release_event)
			else:
				press_event = (current_time, key_name, "press")
				key_events.append(press_event)
				release_event = (current_time+hold_time, key_name, "release")
				key_events.append(release_event)

		key_events.sort()  #sort times to account for rollover

		#Convert absolute timestamps to relative delays
		key_event_delays = []
		for indx in range(len(key_events)):
			prev_time = 0
			if (indx > 0):
				prev_time = key_events[indx-1][0]

			event_time = key_events[indx][0]
			delay = event_time - prev_time
			delay_event = (delay, key_events[indx][1], key_events[indx][2])
			key_event_delays.append(delay_event)


		return key_event_delays

	def pressKey(self, key):
		self.keyboard_controller.press(key)

	def releaseKey(self, key):
		self.keyboard_controller.release(key)

	def typeText(self, text):
		keyPresses = self.genKeypressSequence(text)
		keyPresses = self.insertTypingErrors(keyPresses)
		key_event_delays = self.genKeypressTimes(keyPresses)

		#Type text
		for key_event in key_event_delays:
			delay, key, event_type = key_event
			if (key in self.local_key_replace_dict):
				key = self.local_key_replace_dict[key]
			time.sleep(delay)
			if (event_type == "press"):
				self.keyboard_controller.press(key)
			elif (event_type == "release"):
				self.keyboard_controller.release(key)


def cartesianDistance(p1_x, p1_y, p2_x, p2_y):
	return np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)

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

class MouseHandler():
	def __init__(self, local_machine=False, polling_rate=200):
		self.local_machine = local_machine

		self.start_x = 0
		self.start_y = 0

		self.target_x = 0
		self.target_y = 0

		self.target_start_angle = 0

		self.current_x = None
		self.current_y = None

		self.path_length = 0

		self.v_x = 0
		self.v_y = 0

		self.wind_x = 0
		self.wind_y = 0
		self.wind_v_dampening = np.sqrt(3)
		self.wind_a_dampening = np.sqrt(5)

		self.GRAV_0 = 800
		self.GRAV_MAX = 1000000
		self.WIND_0 = 1000

		self.last_iter_time = 0
		self.path_progress = 0

		script_dir = os.path.dirname(os.path.abspath(__file__))
		profile_dir = os.path.join(script_dir, "user_profiling", "profiles")

		mouse_profile_path = None
		if (os.path.exists(os.path.join(profile_dir, "bespoke_mouse_profile.pickle"))):
			mouse_profile_path = os.path.join(profile_dir, "bespoke_mouse_profile.pickle")
		elif (os.path.exists(os.path.join(profile_dir, "generic_mouse_profile.pickle"))):
			mouse_profile_path = os.path.join(profile_dir, "generic_mouse_profile.pickle")
		else:
			raise FileNotFoundError("Cannot find a mouse profile")
		with open(mouse_profile_path, "rb") as mouse_profile_pickle_file:
			mouse_profile = pickle.load(mouse_profile_pickle_file)
			self.velocity_model = mouse_profile["model"]
			self.v_diff_ratio_avg = mouse_profile["v_diff_ratio_avg"]
			self.v_diff_ratio_std = mouse_profile["v_diff_ratio_std"]
	
		self.poly = PolynomialFeatures(degree=3)
		self.v_diff_ratio = 0

		self.polling_period = 1/float(polling_rate)

		self.holt_time_avg = 0.03
		self.holt_time_std = 0.005

		self.mouse_controller = None
		if (self.local_machine):
			self.mouse_controller = mouse.Controller()
		else:
			raise ValueError("Not implemented yet")

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

	def takeMovementStep(self):
		#Get current distance from start point and target
		start_dist = cartesianDistance(self.current_x, self.current_y, self.start_x, self.start_y)
		target_dist = cartesianDistance(self.current_x, self.current_y, self.target_x, self.target_y)

		#Calculate path progress
		self.path_progress = 1-(target_dist/self.path_length)
		if (self.path_progress < 0):
			self.path_progress = target_dist/(target_dist+start_dist)

		#Calculate angle between current position, start point, and target
		if (start_dist == 0) or (target_dist == 0):
			self.target_start_angle = 180
		else:
			self.target_start_angle = np.degrees(calcAngle([self.start_x,self.start_y], [self.current_x,self.current_y], [self.target_x,self.target_y]))

		self.path_deviation = (180-self.target_start_angle)/180

		#Determine wind speed and direction
		W_mag = min(self.WIND_0*(1-self.path_deviation), target_dist*100)
		self.wind_x = self.wind_x/self.wind_v_dampening + (2*np.random.random()-1)*W_mag/self.wind_a_dampening
		self.wind_y = self.wind_y/self.wind_v_dampening + (2*np.random.random()-1)*W_mag/self.wind_a_dampening

		#Determine gravitational force towards target
		grav_x = (self.GRAV_0*(self.target_x-self.current_x))/(target_dist*(1-self.path_progress)*(1-self.path_deviation))
		grav_y = (self.GRAV_0*(self.target_y-self.current_y))/(target_dist*(1-self.path_progress)*(1-self.path_deviation))

		#Clip gravitational force
		if (np.isnan(grav_x)):
			grav_x = 0
		grav_x = min(grav_x, self.GRAV_MAX)
		grav_x = max(grav_x, -1*self.GRAV_MAX)

		if (np.isnan(grav_y)):
			grav_y = 0
		grav_y = min(grav_y, self.GRAV_MAX)
		grav_y = max(grav_y, -1*self.GRAV_MAX)

		#Determine new x and y velocity
		self.v_x += self.wind_x + grav_x
		if (np.isnan(self.v_x)):
			self.v_x = self.wind_x + grav_x

		self.v_y += self.wind_y + grav_y
		if (np.isnan(self.v_y)):
			self.v_y = self.wind_y + grav_y

		#Clip velocity magnitude
		v_mag = np.hypot(self.v_x, self.v_y)			
		v_max = self.velocity_model.predict(self.poly.fit_transform([[self.path_progress, self.target_start_angle, self.path_length, self.path_deviation]]))[0]
		v_max = v_max*(1+self.v_diff_ratio)
		v_max = max(v_max, 200*(1+self.v_diff_ratio))

		v_clip = None
		if v_mag > v_max:
			v_clip = v_max/2 + np.random.random()*v_max/2
			self.v_x = (self.v_x/v_mag) * v_clip
			self.v_y = (self.v_y/v_mag) * v_clip

		#Move mouse
		if (self.last_iter_time == 0):
			self.last_iter_time = time.time()
		else:
			current_time = time.time()
			time_delta = current_time - self.last_iter_time

			x_delta =  int(self.v_x*time_delta)
			y_delta =  int(self.v_y*time_delta)

			if (abs(x_delta) > 0) or (abs(y_delta) > 0):
				self.current_x += x_delta
				self.current_y += y_delta

				self.mouse_controller.move(x_delta, y_delta)

				self.last_iter_time = current_time


	def moveMouse(self, x_pos, y_pos, pixel_tolerance=4):
		#Update current mouse position
		start_x = self.current_x
		start_y = self.current_y

		if (self.local_machine):
			start_x, start_y = self.mouse_controller.position
		else:
			raise ValueError("Not implemented yet")

		#Move to target position
		self.startNewPath(start_x, start_y, x_pos, y_pos)
		targetReached = (self.current_x == x_pos) and (self.current_y == y_pos)
		prev_x, prev_y = self.current_x, self.current_y
		same_pos_cnt = 0
		while (not targetReached):
			self.takeMovementStep()
			targetReached = (self.current_x == x_pos) and (self.current_y == y_pos)
			time.sleep(self.polling_period)
			if ((prev_x == self.current_x) and (prev_x == self.current_x)):
				same_pos_cnt += 1
			else:
				same_pos_cnt = 0

			if (same_pos_cnt > 10):
				targetReached = (abs(self.current_x - x_pos) <= pixel_tolerance) and (abs(self.current_y - y_pos) <= pixel_tolerance)

			prev_x, prev_y = self.current_x, self.current_y

	def leftClickMouse(self):
		if (self.local_machine):
			self.mouse_controller.press(Button.left)
			time.sleep(max(np.random.normal(self.holt_time_avg, self.holt_time_std, 1)[0], 0.01))
			self.mouse_controller.release(Button.left)
		else:
			raise ValueError("Not implemented yet")

	def rightClickMouse(self):
		if (self.local_machine):
			self.mouse_controller.press(Button.right)
			time.sleep(max(np.random.normal(self.holt_time_avg, self.holt_time_std, 1)[0], 0.01))
			self.mouse_controller.release(Button.right)
		else:
			raise ValueError("Not implemented yet")

	def getMousePostion(self):
		if (self.local_machine):
			return self.mouse_controller.position
		else:
			raise ValueError("Not implemented yet")


class IOHandler():
	def __init__(self, local_machine=False):
		self.local_machine = local_machine

		self.keyboardHandler = KeyboardHandler(local_machine=local_machine)
		self.mouseHandler = MouseHandler(local_machine=local_machine)
		self.screen_handler = ScreenHandler(local_machine=local_machine)

		self.mouse_switch_time_avg = 0.7
		self.mouse_switch_time_std = 0.15

		self.button_click_boundary = 0.8

	def typeText(self, text):
		self.keyboardHandler.typeText(text)
		time.sleep(max(np.random.normal(self.mouse_switch_time_avg, self.mouse_switch_time_std, 1)[0], 0.01))

	def pressKey(self, key):
		self.keyboardHandler.pressKey(key)

	def releaseKey(self, key):
		self.keyboardHandler.releaseKey(key)

	def moveMouse(self, x_pos, y_pos):
		self.mouseHandler.moveMouse(x_pos, y_pos)

	def leftClickMouse(self):
		self.mouseHandler.leftClickMouse()

	def rightClickMouse(self):
		self.mouseHandler.rightClickMouse()

	def clickButton(self, button_img_path=None, button_text=None, bounding_box=None):
		#Get location of button
		x_pos, y_pos, x_width, y_height = (None, None, None, None)
		if not (button_img_path is None):
			x_pos, y_pos, x_width, y_height = self.screen_handler.getImageLocation(button_img_path, bounding_box=bounding_box)
		elif not (button_text is None) and (len(button_text.strip()) > 0):
			x_pos, y_pos, x_width, y_height = self.screen_handler.getTextLocation(button_text, bounding_box=bounding_box)
		else:
			raise ValueError("Missing input argument. Either button_img_path or button_text must be specified")

		#Chose random position near center of button
		x_offset = np.random.normal(0, x_width/6, 1)[0]
		while(abs(x_offset) > x_width*self.button_click_boundary*0.5):
			x_offset = np.random.normal(0, x_width/6, 1)[0]

		y_offset = np.random.normal(0, y_height/6, 1)[0]
		while(abs(y_offset) > y_height*self.button_click_boundary*0.5):
			y_offset = np.random.normal(0, y_height/6, 1)[0]

		x_pos += int(x_offset)
		y_pos += int(y_offset)

		#Move to button position and click
		self.moveMouse(x_pos, y_pos)
		self.leftClickMouse()

		return (x_pos, y_pos)


def main():
	time.sleep(5)
	ioHandler = IOHandler(local_machine=True)
	text = "HELLO darkness my old friend!\nIt's great to see you again. I missed you"
	ioHandler.typeText(text)

	ioHandler.clickButton(button_img_path="button.png")


if __name__ == '__main__':
	#main()
	ioHandler = IOHandler(local_machine=True)
	while True:
		print(ioHandler.mouseHandler.getMousePostion())
		time.sleep(0.3)