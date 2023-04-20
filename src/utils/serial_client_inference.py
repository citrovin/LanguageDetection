# This file is used to create serial connection and display the messages of the device

#!/usr/bin/env python3

import serial
from serial.serialutil import SerialException
import sys
import time
import os

def main(dev: str='/dev/cu.usbmodem1103', baudrate: int=921600, timeout: int=10):
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
	while True:
		try:
			r = ser.readline() # Read one line of data
			print(r)
			
		except KeyboardInterrupt:
			print('\Serial communication interrupted with KeyboardInterrupt.')
			ser.close()
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