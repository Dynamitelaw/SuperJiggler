import os

from IOHandler import *


class ChromeHandler:
	def __init__(self, ioHandler):
		self.ioHandler = ioHandler

		#File Paths
		handlerDir = os.path.dirname(os.path.abspath(__file__))
		self.imageDir = os.path.join(handlerDir, "img")

		#Search queries and reddit pages
		self.googleQueries = ["Our lord and savior Lighting McQueen"]

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
		self.ioHandler.clickButton(os.path.join(self.imageDir,"WindowsTaskbar.PNG"))
		
	def browseReddit(self):
		print("Browsing Reddit", debugLevel=1)
		try:
			self.openChrome()
		except Exception as e:
			print(e)
			return

		#Go to random reddit page
		print("Entering reddit url", debugLevel=2)
		buttonLocation = clickButton(os.path.join(self.buttonImages,"searchBar.PNG"), hardCodedCoord=self.buttonCoordinates["searchBar"])
		if not (buttonLocation):
			pressKeyWithModifier('l', Key.ctrl)
		time.sleep(getNormalSample(0.3, 0.03))
		redditAddress = choice(self.redditPages)
		typeText(redditAddress)
		time.sleep(getNormalSample(0.5, 0.1))
		pressKey(Key.enter)
		time.sleep(getNormalSample(5, 0.2))

		#Scroll around for a bit
		print("Scrolling through reddit page", debugLevel=2)
		moveMouseTo((1897,540), scaleResolution=True)
		for i in range(int(getNormalSample(13, 2))):
			makeMouseMovement(20)
			time.sleep(getNormalSample(3, 0.3))
			moveMouseTo((1897,540), scaleResolution=True)
			slowScroll(randrange(-10,-1,1))

		#Close chrome
		closeCurrentWindow()

	def googleSearch(self):
		print("Performing google search", debugLevel=1)
		try:
			self.openChrome()
		except Exception as e:
			print(e)
			return

		#Enter random search query
		print("Entering google query", debugLevel=2)
		buttonLocation = clickButton(os.path.join(self.buttonImages,"searchBar.PNG"), hardCodedCoord=self.buttonCoordinates["searchBar"])
		if not (buttonLocation):
			pressKeyWithModifier('l', Key.ctrl)
		time.sleep(getNormalSample(0.3, 0.03))
		searchQuery = choice(self.googleQueries)
		typeText(searchQuery)
		time.sleep(getNormalSample(0.5, 0.1))
		pressKey(Key.enter)
		time.sleep(getNormalSample(10, 0.2))
		makeMouseMovement(40)

		#Go to stack overflow result
		print("Opening stack overflow result", debugLevel=2)
		buttonLocation = clickButton(os.path.join(self.buttonImages,"stackOverflowResult.PNG"), hardCodedCoord=self.buttonCoordinates["stackOverflowResult"])
		if not (buttonLocation):
			print("Could not find stack overflow result. Will scroll through results instead")
		time.sleep(getNormalSample(1, 0.1))

		#Scroll around for a bit
		print("Scroll through result", debugLevel=2)
		moveMouseTo((1897,540), scaleResolution=True)
		for i in range(int(getNormalSample(5, 1))):
			makeMouseMovement(20)
			time.sleep(getNormalSample(7, 0.3))
			moveMouseTo((1897,540), scaleResolution=True)
			slowScroll(randrange(-10,-1,1))

		#Close chrome
		closeCurrentWindow()