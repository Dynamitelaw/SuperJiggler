from PIL import ImageGrab
import time
import cv2
import numpy as np
import sys
import os


class ScreenHandler():
	def __init__(self, local_machine=False):
		self.local_machine = local_machine

		script_dir = os.path.dirname(os.path.abspath(__file__))
		self.screenshot_dir = os.path.join(script_dir, "temp")
		if not (os.path.exists(self.screenshot_dir)):
			os.mkdir(self.screenshot_dir)

	def getScreenshot(self):
		output_path = os.path.join(self.screenshot_dir, "screenshot.png")

		if (self.local_machine):
			screenshot = ImageGrab.grab()
			screenshot.save(output_path)
			screenshot.close()
		else:
			raise ValueError("Not implemented yet")

		return output_path

	def getImageLocation(self, image_path):
		#Get image of screen
		screen_img = cv2.imread(self.getScreenshot())
		#Open search image
		template = cv2.imread(image_path)
		template_width, template_height = template.shape[0:2] 

		#Find matches on screen
		match_scores = cv2.matchTemplate(screen_img, template, cv2.TM_CCOEFF_NORMED) 
		threshold = 0.8
		match_locations = np.where(match_scores >= threshold) 

		#Return first match location and dimensions
		first_match = (match_locations[1][0]+int(template_height/2), match_locations[0][0]+int(template_width/2), template_width, template_height)

		return first_match

if __name__ == '__main__':
	screen_handler = ScreenHandler(local_machine=True)
	pos = screen_handler.getImageLocation("button.png")
	print(pos)