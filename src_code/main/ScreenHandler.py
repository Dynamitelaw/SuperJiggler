from PIL import ImageGrab
import time
import cv2
import numpy as np
import sys
import os
import easyocr
from difflib import SequenceMatcher


class ScreenHandler():
	def __init__(self, local_machine=False, ocr_languange="en"):
		self.local_machine = local_machine

		script_dir = os.path.dirname(os.path.abspath(__file__))
		self.screenshot_dir = os.path.join(script_dir, "temp")
		if not (os.path.exists(self.screenshot_dir)):
			os.mkdir(self.screenshot_dir)

		self.ocr_reader = easyocr.Reader([ocr_languange])

	def getScreenshot(self, bounding_box=None):
		output_path = os.path.join(self.screenshot_dir, "screenshot.png")

		if (self.local_machine):
			screenshot = ImageGrab.grab(bbox=bounding_box)
			screenshot.save(output_path)
			screenshot.close()
		else:
			raise ValueError("Not implemented yet")

		return output_path

	def getImageLocation(self, image_path, bounding_box=None):
		#Check search image path
		if not (os.path.exists(image_path)):
			raise FileNotFoundError("Could not find button image \"{}\"".format(image_path))

		#Get image of screen
		screen_img = cv2.imread(self.getScreenshot(bounding_box=bounding_box))
		#Open search image
		template = cv2.imread(image_path)
		template_width, template_height = template.shape[0:2] 

		#Find matches on screen
		match_scores = cv2.matchTemplate(screen_img, template, cv2.TM_CCOEFF_NORMED) 
		threshold = 0.8
		match_locations = np.where(match_scores >= threshold) 

		#Return first match location and dimensions
		first_match = None
		try:
			first_match = (match_locations[1][0]+int(template_height/2), match_locations[0][0]+int(template_width/2), template_width, template_height)
		except:
			raise ValueError("Could not find image \"{}\" on screen".format(image_path))

		return first_match

	def getTextLocation(self, search_text, bounding_box=None, threshold=0.8):
		start_time = time.time()
		#Get image of screen
		screen_img = self.getScreenshot(bounding_box=bounding_box)

		#Extract text with ocr
		results = self.ocr_reader.readtext(screen_img)

		#Find best match for search text
		best_match_rect = None
		best_match_score = None
		for (bounding_box, detected_text, confidence) in results:
			match_score = SequenceMatcher(None, search_text, detected_text).ratio()
			if (best_match_score is None) or (match_score > best_match_score):
				best_match_score = match_score
				best_match_rect = bounding_box

		if (best_match_score < threshold):
			raise ValueError("Could not find text \"{}\" on screen".format(search_text))
		
		#Convert bounding box points to center position and dimenstions
		best_match_rect = np.transpose(best_match_rect)
		x_pos = int(np.average(best_match_rect[0]))
		y_pos = int(np.average(best_match_rect[1]))

		x_width = int(np.max(best_match_rect[0]) - np.min(best_match_rect[0]))
		y_height = int(np.max(best_match_rect[1]) - np.min(best_match_rect[1]))

		elapsed_time = time.time() - start_time
		print(elapsed_time)
		
		return (x_pos, y_pos, x_width, y_height)

if __name__ == '__main__':
	screen_handler = ScreenHandler(local_machine=True)
	k = screen_handler.getTextLocation("Preferences")

	print(k)