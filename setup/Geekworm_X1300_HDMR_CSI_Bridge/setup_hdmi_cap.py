import subprocess
import re
import time


def runCommand(cmd):
	print("# {}".format(cmd))
	cmd_output = subprocess.check_output(cmd, shell=True, text=True)
	print(cmd_output)
	return cmd_output

#Get media name
cmd = "v4l2-ctl --list-devices"
cmd_output = runCommand(cmd)

in_csi_section = False
media_name = None
for line in cmd_output.split("\n"):
	if (in_csi_section) and (re.search("/dev/media", line)):
		media_name = line.strip()
		print(media_name)
		break
	if (re.search("rp1-cfe.*csi", line)):
		in_csi_section = True


if (media_name is None):
	raise Exception("Could not find media name from v4l2 device list. Make sure overlays are setup properly in /boot/firmware/config.txt")

#Load EDID data
edid_path = "/home/jose/gitRepos/SuperJiggler/setup/Geekworm_X1300_HDMR_CSI_Bridge/1080P60EDID.txt"
cmd = "v4l2-ctl -d /dev/v4l-subdev2 --set-edid=file={} --fix-edid-checksums".format(edid_path)
cmd_output = runCommand(cmd)

#Query device timings to ensure HDMI port is receiving signal
time.sleep(0.5)
cmd = "v4l2-ctl -d /dev/v4l-subdev2 --query-dv-timings"
cmd_output = runCommand(cmd)

correct_x_res = False
correct_y_res = False

for line in cmd_output.split("\n"):
	if (re.search("failed: Link has been severed", line)):
		break
	if (re.search(".*Active.*width.*1920", line)):
		correct_x_res = True
	if (re.search(".*Active.*height.*1080", line)):
		correct_y_res = True

	if (correct_x_res and correct_y_res):
		break

if not (correct_x_res and correct_y_res):
	raise Exception("No HDMI input detected. Make sure the HDMI cable is connected to both the RaspberryPi and the target computer")

#Apply the screen timing to the capture setup
cmd = "v4l2-ctl -d /dev/v4l-subdev2 --set-dv-bt-timings query"
cmd_output = runCommand(cmd)

#Initialize/reset media
cmd = "media-ctl -d {} -r".format(media_name)
cmd_output = runCommand(cmd)

#Connect CSI2's pad4 to rp1-cfe-csi2_ch0's pad0
cmd = "media-ctl -d {} ''\\''csi2'\\'':4 -> '\\''rp1-cfe-csi2_ch0'\\'':0 [1]'".format(media_name)
cmd_output = runCommand(cmd)

#Configure the media node
cmd = "media-ctl -d {} -V ''\\''csi2'\\'':0 [fmt:RGB888_1X24/1920x1080 field:none colorspace:srgb]'".format(media_name)
cmd_output = runCommand(cmd)

cmd = "media-ctl -d {} -V ''\\''csi2'\\'':4 [fmt:RGB888_1X24/1920x1080 field:none colorspace:srgb]'".format(media_name)
cmd_output = runCommand(cmd)

cmd = "media-ctl -d {} -V ''\\''tc358743 4-000f'\\'':0 [fmt:RGB888_1X24/1920x1080 field:none colorspace:srgb]'".format(media_name)
cmd_output = runCommand(cmd)

#Set output formats
cmd = "v4l2-ctl -v width=1920,height=1080,pixelformat=RGB3"
cmd_output = runCommand(cmd)

