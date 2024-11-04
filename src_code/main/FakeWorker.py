import platform

from program_handlers import *
from IOHandler import *


def runActivitySession(sessionLength, sessionProgramName):
	'''
	Run an activity session for a single program
	'''
	startTime = time.time()
	print("Starting {} session".format(sessionProgramName))

	while(time.time()-startTime < sessionLength):
		makeMouseMovement(10)		
		if (sessionProgramName == "Outlook"):
			g_outlookHandler.randomAction()
		elif (sessionProgramName == "VNC"):
			g_vncHandler.randomAction()
		elif (sessionProgramName == "Chrome"):
			g_chromeHandler.randomAction()


def main():
	# Setup command line arguments
	helpdesc = "This script will keep your usage statistics up and your status active while you're away"
	parser = argparse.ArgumentParser(description=helpdesc)
	parser.add_argument('-settings', action="store", dest='settings', default='usageSettings.json', help='Filepath for json file containing script settings. Defaults t0 \"usageSettings.json\" in cwd.')
	parser.add_argument('-duration', action="store", dest='duration', help='Specify duration of makeBusy activity in minutes. Script will quit after specified time has passed')
	arguments = parser.parse_args()

	settingsPath = arguments.settings
	duration = arguments.duration

	#Read settings file
	global g_settings
	g_settings = jsonToDict(settingsPath)

	#Setup global constants
	global g_mouseSpeed
	g_mouseSpeed = g_settings["inputSettings"]["mouseSpeed"]
	global g_keypressLength
	g_keypressLength = g_settings["inputSettings"]["meanKeypressLength"]/5
	global g_keypressSeperator
	g_keypressSeperator = g_settings["inputSettings"]["meanKeypressSeperator"]/5

	global g_monitorWidthScale
	g_monitorWidthScale = g_settings["monitorSettings"]["width"] / 2160  #this program was calibrated on 1440p monitor. So this allows you to scale hard-coded coordinates for different resolutions
	global g_monitorHeightScale
	g_monitorHeightScale = g_settings["monitorSettings"]["height"] / 1440

	#Set up session settings
	sessionProbability = g_settings["activitySettings"]["sessionProbability"]
	meanSessionLength = g_settings["activitySettings"]["meanSessionLength"]  #in seconds
	sessionLengthStd = g_settings["activitySettings"]["sessionLengthStd"]
	meanSessionPollTime = g_settings["activitySettings"]["meanSessionPollTime"]
	sessionPollTimeStd = g_settings["activitySettings"]["sessionPollTimeStd"]

	meanBreakLength = g_settings["breakSettings"]["meanBreakLength"]
	breakLengthStd = g_settings["breakSettings"]["breakLengthStd"]
	breakProbability = g_settings["breakSettings"]["breakProbability"]

	#Set up program usage distribution
	programSettings = g_settings["programSettings"]
	programDistribution = []
	for programName in programSettings:
		programDistribution += [programName] * int(1000*programSettings[programName]["usageProbability"])

	#Instantiate program handlers
	global g_outlookHandler
	g_outlookHandler = OutlookHandler()
	global g_vncHandler
	g_vncHandler = VncHandler()
	global g_chromeHandler
	g_chromeHandler = ChromeHandler()

	#Start main loop
	time.sleep(5)
	seed(time.time())
	makeMouseMovement(20)
	startTime = time.time()

	while(True):
		sessionRandom = random()
		if (sessionRandom < sessionProbability):
			breakRandom = random()
			if (breakRandom < breakProbability):
				print("Start break")
				breakLength = getNormalSample(meanBreakLength, breakLengthStd)
				time.sleep(breakLength)
				print("End break")
			else:
				print("Start activity session")
				moveMouseTo((1000, 400), scaleResolution=True)
				sessionLength = getNormalSample(meanSessionLength, sessionLengthStd)
				sessionProgramName = choice(programDistribution)
				runActivitySession(sessionLength, sessionProgramName)
				print("End activity session")

		sessionPollTime = getNormalSample(meanSessionPollTime, sessionPollTimeStd)
		moveMouseTo((1189, 605), scaleResolution=True)
		time.sleep(sessionPollTime)
		makeMouseMovement(20)

		if (duration):
			print("Checking time allotment")
			currentTime = time.time()
			elapsedTime = (currentTime - startTime)/60
			if (elapsedTime > int(duration)):
				print("Specified time has passed. Ending program")
				sys.exit()


if __name__ == '__main__':
	#Determine if we are running on target machine, or on a Raspberry Pi
	local_machine = True
	if ("aarch" in platform.machine()) and ("Linux" in platform.system()) and ("rpt-rpi" in system.release()):
		local_machine = False
	print(local_machine)
	
	ioHandler = IOHandler(local_machine=local_machine)
	chrome = ChromeHandler(ioHandler)
	#chrome.openChrome()
	#chrome.googleSearch()