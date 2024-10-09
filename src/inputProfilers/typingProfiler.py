import time
import keyboard
import json
import numpy as np
import math
import os


g_ordered_key_events = []
g_numEvents = 0

g_key_name_list = [
	"a","b","c","d","e","f","g","h","i","j","k","l","m",
	"n","o","p","q","r","s","t","u","v","w","x","y","z",
	"1","2","3","4","5","6","7","8","9","0",
	",",".","/",";","'","[","]","\\","-","=",
	"esc","tab","caps lock","shift","right shift",
	"ctrl","right ctrl","alt","right alt",
	"space","enter","backspace"
]

g_keyEventCount = {}

def logKeyEvent(event):
	global g_ordered_key_events
	global g_numEvents
	global g_keyEventCount

	g_ordered_key_events.append(event)
	g_numEvents += 1

	if (event.name in g_keyEventCount):
		g_keyEventCount[event.name] += 1


def captureKeystrokeData():
	global g_keyEventCount

	#Prepopulate g_keyEventCount
	for keyName in g_key_name_list:
		g_keyEventCount[keyName] = 0

	#Collect typing data
	print("Collecting typing data...")
	hook = keyboard.hook(logKeyEvent)

	enoughSamplesCollected = False
	startTime = time.time()
	while (not enoughSamplesCollected):
		doneTemp = True
		for key in g_keyEventCount:
			if (g_keyEventCount[key] < 10) and (len(key) < 2):
				doneTemp = False
				break
		enoughSamplesCollected = doneTemp

		elapsedTime = time.time() - startTime
		if (elapsedTime > 500):
			enoughSamplesCollected = True

	print("Data collection complete")
	keyboard.unhook(hook)
	

def extractKeystrokeData(ordered_key_events):
	print("Extracting keystroke data")
	#Create a list of only down (press) events
	press_events = [event for event in ordered_key_events if (event.event_type == "down")]
	if (len(press_events) < 3):
		raise ValueError("List of key events is too short")

	#Get time it took to hit each key, given what the previous key was
	key_hit_times = {}
	for indx in range(1, len(press_events)):
		key = press_events[indx]
		if not (key.name in key_hit_times):
			key_hit_times[key.name] = {}

		prev_key = press_events[indx-1]
		if not (prev_key.name in key_hit_times[key.name]):
			key_hit_times[key.name][prev_key.name] = []

		hitTime = key.time - prev_key.time
		if (hitTime < 60):
			key_hit_times[key.name][prev_key.name].append(hitTime)

	#Get the times each key is held for
	press_time_dict = {}
	hold_time_dict = {}
	for event in ordered_key_events:
		keyName = event.name
		if not (keyName in press_time_dict):
			press_time_dict[keyName] = None

		if (event.event_type == "down"):
			press_time_dict[keyName] = event.time

		if (event.event_type == "up"):
			pressTime = press_time_dict[keyName]
			if (pressTime is None):
				continue

			releaseTime = event.time
			holdTime = releaseTime - pressTime
			if (holdTime > 0) and (holdTime < 60):
				if not (keyName in hold_time_dict):
					hold_time_dict[keyName] = []

				hold_time_dict[keyName].append(holdTime)

	#Get all backspace sequences
	backspace_sequences = []
	current_sequence_length = None
	in_sequence = False
	for key_press in press_events:
		if (key_press.name == "backspace"):
			if (in_sequence):
				current_sequence_length += 1
			else:
				in_sequence = True
				current_sequence_length = 1
		else:
			if (in_sequence):
				in_sequence = False
				backspace_sequences.append(current_sequence_length)
				current_sequence_length = None

	#Return data
	data = {}
	data["hit_delays"] = key_hit_times
	data["hold_times"] = hold_time_dict
	data["backspace_sequences"] = backspace_sequences

	return data


def fineTuneDistribution(global_average, global_std, new_data, blend_factor=5):
	#Fine tune the global distribution to bring it closer to the new data
	#The more new data points there are, the more the new distribution will be affected by the new data

	if (len(new_data) < 2):
		return global_average, global_std

	new_avg = np.average(new_data)
	new_std = np.std(new_data)
	num_new_data_points = len(new_data)

	new_ratio = 1 - (1/(2**(num_new_data_points/blend_factor)))

	fine_tuned_avg = (global_average*(1-new_ratio)) + (new_avg*new_ratio)
	fine_tuned_std = (global_std*(1-new_ratio)) + (new_avg*new_std)

	return fine_tuned_avg, fine_tuned_std


def analyzeKeystrokes(ordered_key_events):
	#Extract keystroke data
	key_data = extractKeystrokeData(ordered_key_events)

	print("Analyzing keystroke data")
	#Get list of all hit delays
	all_hit_delays = []
	for keyName in key_data["hit_delays"]:
		for prev_key in key_data["hit_delays"][keyName]:
			all_hit_delays += key_data["hit_delays"][keyName][prev_key]

	#Remove upper and lower 5% of hit delay data points
	all_hit_delays.sort()
	lower_bound_indx = math.ceil(len(all_hit_delays)*0.05)
	upper_bound_indx = math.floor(len(all_hit_delays)*0.95)
	if (upper_bound_indx-lower_bound_indx < 3):
		raise ValueError("Insufficient key hit delay data")

	all_hit_delays = all_hit_delays[lower_bound_indx:upper_bound_indx]

	#Calculate mean and standard deviation of hit delays
	global_prev_delay_avg = np.average(all_hit_delays)
	global_prev_delay_std = np.std(all_hit_delays)

	#Get list of all hold times
	all_hold_times = []
	for keyName in key_data["hold_times"]:
		if (keyName != "shift"):
			all_hold_times += key_data["hold_times"][keyName]

	#Remove upper and lower 5% of hold time data points
	all_hold_times.sort()
	lower_bound_indx = math.ceil(len(all_hold_times)*0.05)
	upper_bound_indx = math.floor(len(all_hold_times)*0.95)
	if (upper_bound_indx-lower_bound_indx < 3):
		raise ValueError("Insufficient key hold time data")

	all_hold_times = all_hold_times[lower_bound_indx:upper_bound_indx]

	#Calculate mean and standard deviation of hold times
	global_hold_time_avg = np.average(all_hold_times)
	global_hold_time_std = np.std(all_hold_times)

	#Calculate error rate and overshoot stats
	num_backspace = 0
	if ("backspace" in key_data["hold_times"]):
		num_backspace = len(key_data["hold_times"]["backspace"])
	error_rate = len(key_data["backspace_sequences"])/(len(all_hold_times)-num_backspace)

	error_overshoot_len_avg = (0.1/global_prev_delay_avg)*3
	error_overshoot_len_std = 1
	if (len(key_data["backspace_sequences"]) > 2):
		error_overshoot_len_avg = np.average(key_data["backspace_sequences"])
		error_overshoot_len_std = np.std(key_data["backspace_sequences"])

	#Create stats dict for each key
	key_stats = {}
	for keyName in g_key_name_list:
		key_stats[keyName] = {}

	#Calculate hit delay stats for each key
	for keyName in key_stats:
		hit_delay_stats = {}

		hit_delay_data = {}
		if keyName in key_data["hit_delays"]:
			hit_delay_data = key_data["hit_delays"][keyName]

		#Get fine tuned net hit delay stats for this key
		combined_delays = []
		for prev_key in hit_delay_data:
			combined_delays += hit_delay_data[prev_key]

		key_delay_avg, key_delay_std = fineTuneDistribution(global_prev_delay_avg, global_prev_delay_std, combined_delays)

		#Get fine tuned hit delay stats based on previous keys
		for prev_key in g_key_name_list:
			prev_delay_avg = key_delay_avg
			prev_delay_std = key_delay_std

			if (prev_key in hit_delay_data):
				prev_key_data = hit_delay_data[prev_key]
				prev_delay_avg, prev_delay_std = fineTuneDistribution(prev_delay_avg, prev_delay_std, prev_key_data)

			hit_delay_stats[prev_key] = {"avg": prev_delay_avg, "std": prev_delay_std}

		key_stats[keyName]["hit_delays"] = hit_delay_stats

	#Calculate hold time stats for each key
	for keyName in key_stats:
		hold_delay_stats = {}

		hold_delay_data = {}
		if keyName in key_data["hold_times"]:
			hold_delay_data = key_data["hold_times"][keyName]

		#Get fine tuned hold time stats for this key
		key_hold_avg, key_hold_std = fineTuneDistribution(global_hold_time_avg, global_hold_time_std, hold_delay_data)
		key_stats[keyName]["hold_time"] = {"avg": key_hold_avg, "std": key_hold_std}

	#Return final stats
	stats = {}
	stats["overall"] = {
		"hit_delay": {"avg": global_prev_delay_avg, "std": global_prev_delay_std},
		"hold_time": {"avg": global_hold_time_avg, "std": global_hold_time_std},
		"error_rate": error_rate,
		"error_overshoot_len": {"avg": error_overshoot_len_avg, "std": error_overshoot_len_std}
	}
	stats["per_key"] = key_stats

	return stats


def saveRawData(output_path="typing_data.log"):
	if not (os.path.exists(output_path)):
		output_file = open(output_path, "w")
		output_file.write("")
		output_file.close()

	output_file = open(output_path, "a")
	output_file.write("#Session {}\n".format(time.time()))
	for event in g_ordered_key_events:
		output_file.write("{}\n".format(event.to_json()))
	output_file.close()


def importRawData(data_path="typing_data.log"):
	global g_ordered_key_events

	if not (os.path.exists(data_path)):
		raise ValueError("Path \"{}\" does not exist".format(data_path))

	data_file = open(data_path, "r")
	line = data_file.readline()
	while(line):
		if not ("#Session" in line):
			try:
				event_dict = json.loads(line.strip())
				event_obj = keyboard.KeyboardEvent(event_dict["event_type"], event_dict["scan_code"], name=event_dict["name"], time=event_dict["time"], is_keypad=event_dict["is_keypad"])
				g_ordered_key_events.append(event_obj)
			except Exception as e:
				pass

		line = data_file.readline()

	data_file.close()


def main():
	importRawData()
	#captureKeystrokeData()
	#saveRawData()

	typing_profile = analyzeKeystrokes(g_ordered_key_events)
	
	print("Writing keystroke profile")
	output_file = open("typing_profile.json", "w")
	output_file.write(json.dumps(typing_profile, indent=2, sort_keys=True))
	output_file.close()

main()