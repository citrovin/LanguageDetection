# This file is used to create a file from recordings, using the board

#!/usr/bin/env python3

import serial
from serial.serialutil import SerialException
import sys
import time
import os

def main(dev: str='/dev/cu.usbmodem1103', baudrate: int=921600, timeout: int=10):
	with open('out.pcm', 'wb') as f:
		print(f'Waiting on {dev}…')
		ser = None
		while ser is None:
			if os.path.exists(dev):
				try:
					print('Connecting...')
					ser = serial.Serial(dev, baudrate, timeout=timeout)
					print('Serial communication established!')
				except:
					pass
			time.sleep(0.1)
		print('Recording... | Stop recording with CRTL+C')
		while True:
			try:
				r = ser.readline() # Read one line of data
				f.write(r[:-2]) # Write raw data, discarding \r\n
				# print(r)
				
				# r = ser.read(96) # Read one line of data
				# r = ser.read(64) # Read one line of data
				# f.write(r) # Write raw data, discarding \r\n
				
			except KeyboardInterrupt:
				print('\nRecording data interrupted with KeyboardInterrupt: closing and saving file.')
				ser.close()
				f.close()
				break

			except SerialException: # Handle serial port disconnect, try to reconnect
				ser.close()
				ser = None

				print(f'Waiting on {dev}…')
				while ser is None:
					if os.path.exists(dev):
						try:
							ser = serial.Serial(dev, baudrate, timeout=timeout)
						except:
							pass
					time.sleep(0.1)
					
            

if __name__ == '__main__':
	main(*sys.argv[1:])