import os
import TypingProfiler
import MouseProfiler
from MouseProfiler import MouseEvent
import sys


def main():
	print("This script will collect keystroke and mouse movement data to create a user interface profile")
	input("Press Enter to continue:")

	#Create keyboard profile
	print("The script needs ~10 min of typing data for the average user.\nGo to a type test website and start typing.")
	print("The typing test you use should use complete sentences that have various letter cases and punctuation.")
	print("https://www.typingtest.com/ is recommended.")
	#TypingProfiler.createUserProfile()
	print("Typing profile complete")

	#Create mouse profile
	print("The script needs ~10 min of mouse movement data for the average user")
	print("https://mouseaccuracy.com/")
	print("Go to the website above, and play a sequency of 90 second sessions on EASY difficulty")
	MouseProfiler.createUserProfile()

	print("Interface profiles complete!")

if __name__ == '__main__':
	main()
	sys.exit()
