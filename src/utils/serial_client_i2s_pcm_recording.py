# This file is used to create a file from recordings, using the board

#!/usr/bin/env python3

import serial
from serial.serialutil import SerialException
import sys
import time
import os

def main(dev: str='/dev/cu.usbmodem1103', baudrate: int=921600, timeout: int=10, data_path: str='./data/'):

	file_path = get_path(data_path=data_path)
	print(f'File created in: {file_path}')

	with open(file_path, 'wb') as f:
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
				
			except KeyboardInterrupt:
				print('\nRecording data interrupted with KeyboardInterrupt: closing and saving file.')
				f.close()
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
					

def get_path(data_path: str='./data/'):
	
	choice=int(input('Choose language: \n'
       + '1. German\n' 
	   + '2. English\n'  
       + '3. French\n'
       + '4. Italian\n'
       + '5. Test (e.g. tuning gain of mic)\n'
       
	   )
	)

	file_path = data_path
	
	if(choice==1):
		file_path += 'german/'
	elif(choice==2):
		file_path += 'engilsh/'
	elif(choice==3):
		file_path += 'french/'
	elif(choice==4):
		file_path += 'italian/'
	elif(choice==5):
		file_path += 'test/'
	else:
		print('LanguageError: language does not exist in the data directory.')

	
	specs = input('Name of recording: ')
	specs += '.pcm'
	file_path += specs


	return file_path



if __name__ == '__main__':
	main(*sys.argv[1:])

	# file_path = get_path()
	# print(file_path)
	