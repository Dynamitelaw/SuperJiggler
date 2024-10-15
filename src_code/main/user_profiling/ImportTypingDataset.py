import os
import pandas as pd
import keyboard
import multiprocessing as mproc
import time
import pickle



g_letter_translate_dict = {
	" ": "space",
	"bksp": "backspace",
	"<": ",",
	">": ".",
	"?": "/",
	":": ";",
	"\"": "'",
	"{": "[",
	"}": "]",
	"|": "\\",
	"_": "-",
	"+": "=",
	"!": "1",
	"@": "2",
	"$": "3",
	"%": "4",
	"^": "5",
	"&": "6",
	"*": "7",
	"9": "8",
	")": "9"
}

g_key_name_list = [
	"a","b","c","d","e","f","g","h","i","j","k","l","m",
	"n","o","p","q","r","s","t","u","v","w","x","y","z",
	"1","2","3","4","5","6","7","8","9","0",
	",",".","/",";","'","[","]","\\","-","=",
	"esc","tab","caps lock","shift","right shift",
	"ctrl","right ctrl","alt","right alt",
	"space","enter","backspace"
]

def convertData(args):
	pid, output_dir, dataset_path = args


	raw_data_path = os.path.join(dataset_path, "{}_keystrokes.txt".format(pid))
	if (not os.path.exists(raw_data_path)):
		return
	raw_data_file = open(raw_data_path, "r")

	if (not os.path.exists(output_dir)):
		os.mkdir(output_dir)

	output_file_path = os.path.join(output_dir, "{}.event_list.pickle".format(pid))
	#output_file = open(output_file_path, "w")

	events = []

	line = raw_data_file.readline()
	if ("PARTICIPANT_ID" in line):
		line = raw_data_file.readline()
	while (line):
		try:
			line_list = line.strip().split("\t")
			#PARTICIPANT_ID, TEST_SECTION_ID, SENTENCE, USER_INPUT, KEYSTROKE_ID, PRESS_TIME, RELEASE_TIME, LETTER, KEYCODE

			letter = line_list[7].strip().lower()
			if (letter in g_letter_translate_dict):
				letter = g_letter_translate_dict[letter]
			elif (len(letter) == 0):
				letter = "space"
			elif ((len(letter) > 1) and (not (letter in g_key_name_list))):
				raise ValueError("Unkown letter \"{}\"\n{}".format(letter, line_list))

			event_obj_press = keyboard.KeyboardEvent("down", 0, name=letter, time=float(line_list[5])/1000, device=None, modifiers=None, is_keypad=False)
			event_obj_release = keyboard.KeyboardEvent("up", 0, name=letter, time=float(line_list[6])/1000, is_keypad=False)
			
			#output_file.write("{}\n".format(event_obj_press.to_json()))
			#output_file.write("{}\n".format(event_obj_release.to_json()))

			events.append(event_obj_press)
			events.append(event_obj_release)
		except:
			pass

		line = raw_data_file.readline()

	raw_data_file.close()
	#output_file.close()

	with open(output_file_path, "wb") as events_pickle_file:
		pickle.dump(events, events_pickle_file)


def main():
	#Get dataset metadata
	dataset_path = "C:\\Users\\Dynamitelaw\\Downloads\\Keystrokes\\Keystrokes\\files"
	metadata_path = os.path.join(dataset_path, "metadata_participants.txt")

	metadata_df = pd.read_csv(metadata_path, sep='\t', header=0)

	#Filter out undesired participants
	metadata_df = metadata_df[metadata_df['LAYOUT'] == "qwerty"]
	metadata_df = metadata_df[metadata_df['KEYBOARD_TYPE'] != "on-screen"]
	metadata_df = metadata_df[metadata_df['NATIVE_LANGUAGE'] == "en"]
	metadata_df = metadata_df[metadata_df['AVG_WPM_15'] > 30]
	metadata_df = metadata_df[metadata_df['AVG_WPM_15'] < 60]

	id_list = list(metadata_df['PARTICIPANT_ID'])

	#print(id_list)
	#print(id_list[0])
	output_path = "C:\\Users\\Dynamitelaw\\Downloads\\Keystrokes\\Keystrokes\\files_converted"
	args = []
	for pid in id_list:
		args.append((pid, output_path, dataset_path))

	start_time = time.time()
	with mproc.Pool(40) as proc_pool:
		proc_pool.map(convertData, args)

	elapsed_time = time.time() - start_time
	print(elapsed_time/60)

if __name__ == '__main__':
	main()