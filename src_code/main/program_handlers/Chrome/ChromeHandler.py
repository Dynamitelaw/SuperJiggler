import os
from random import choice

from IOHandler import *


class ChromeHandler:
	def __init__(self, ioHandler):
		self.ioHandler = ioHandler

		#File Paths
		handlerDir = os.path.dirname(os.path.abspath(__file__))
		self.imageDir = os.path.join(handlerDir, "img")

		#Search queries and reddit pages
		self.googleQueries = ["Our lord and savior Lightning McQueen"]

		self.redditPages = [
			"https://www.reddit.com/r/ProgrammerHumor/",
			"https://www.reddit.com/r/wallstreetbets/"
		]


	def randomAction(self):
		#Run random chrome action
		actionName = choice(self.actionDistribution)

		if (actionName == "browseReddit"):
			self.browseReddit()
		elif (actionName == "googleSearch"):
			self.googleSearch()

	def openChrome(self):
		#Select Chrome from taskbar
		print("Selecting Chrome from taskbar")
		self.ioHandler.clickButton(button_img_path=os.path.join(self.imageDir,"WindowsTaskbar.PNG"))
		
	def browseReddit(self):
		print("Browsing Reddit", debugLevel=1)
		

	def googleSearch(self):
		print("Performing google search")
		try:
			self.openChrome()
		except Exception as e:
			print(e)
			return

		#Enter random search query
		time.sleep(0.6)
		print("Entering google query")
		try:
			self.ioHandler.clickButton(button_img_path=os.path.join(self.imageDir1,"SearchBar.PNG"))
		except:
			self.ioHandler.clickButton(button_text="Search Google or type a URL")
		time.sleep(0.3)
		searchQuery = choice(self.googleQueries)
		self.ioHandler.typeText(searchQuery)
		time.sleep(0.5)
		self.ioHandler.pressKey(keyboard.Key.enter)
		time.sleep(0.6)
		buttonLocation = self.ioHandler.clickButton(button_text="Reddit", bounding_box=(0,0,1100,1400))

		