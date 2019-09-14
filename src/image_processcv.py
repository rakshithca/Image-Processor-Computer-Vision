import argparse
import cv2,time
import numpy as np
import sys
from scipy import misc



if __name__=="__main__":
	
	parser=argparse.ArgumentParser()

	parser.add_argument("--path",help="path")
	
	args=parser.parse_args()

	def input_function():
		global image_cv
		global kernel_cv
		if(len(sys.argv)< 2):
			image=cv2.VideoCapture(0)
			check,image_cv=image.read()
			
			cv2.imshow('Capturing',image_cv)		
			kernel_cv=cv2.waitKey(0)
			image.release()
		else:

			image_cv=cv2.imread("%s"%(args.path),1)				
			
			cv2.imshow('Image',image_cv)
			kernel_cv=cv2.waitKey(0) 
			
		
	input_function()	

	blue_image_cv=np.copy(image_cv)
	green_image_cv=np.copy(image_cv)
	red_image_cv=np.copy(image_cv)

	blue_image_cv[:,:,1:]=0
	
	green_image_cv[:,:,(0,2)]=0
	
	red_image_cv[:,:,:2]=0
	
	
	x,y,z=image_cv.shape 
	
	
	

	def sliderHandler(n):
		global image_cv
		global distance
		kernel=np.ones((n,n),np.float32)/(n*n)
		distance=cv2.filter2D(image_cv,-1,kernel)
		cv2.imshow('processed',distance)
	
	def sliderHandler1(n):
		global image_cv
		global distance
		angle = 45*np.pi/180
		rows = image_cv.shape[0]
		cols = image_cv.shape[1]
		M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
		distance = cv2.warpAffine(image_cv,M,(cols,rows))
		cv2.imshow('Rotation',distance)
	
			
		
	
	def save(x):
		print("Image is saved")
		cv2.imwrite("out.jpg",x)
		print("Press 'i' to reload image or press enter to quit!")
		
	def reload():
		cv2.destroyAllWindows()
		input_function()
		
		
		
	if kernel_cv==ord('g'):
		image_gray=cv2.cvtColor(image_cv,cv2.COLOR_BGR2GRAY)	
		cv2.imshow("Image_grayscale",image_gray)
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(image_gray)
			m=cv2.waitKey(0)
 
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
			
	if(kernel_cv==ord('i')):
			reload()
			
	if(kernel_cv==ord('G')):	
		
		gray_image=np.copy(image_cv)
		gray_image[:] = image_cv.mean(axis=-1,keepdims=1) 

		cv2.imshow('User_implemented_Grayscale',gray_image)
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(gray_image)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
		
		

	if(kernel_cv==ord('c')):		
		cv2.imshow('Blue Channel',blue_image_cv)		
		r=cv2.waitKey(0)		
		if(r==ord('c')):		
			cv2.imshow('Green Channel',green_image_cv)
			g=cv2.waitKey(0)
			if(g==ord('c')):		
				cv2.imshow('Red Channel',red_image_cv)
				g=cv2.waitKey(0)
				cv2.destroyAllWindows()
				if g==ord('w'):
					save(blue_image_cv)
					m=cv2.waitKey(0)
					if(m==ord('i')):
						reload()
				if(g==ord('i')):
					reload()
					
				
		
	
	if(kernel_cv==ord('s')):	
		image_cv=cv2.cvtColor(image_cv,cv2.COLOR_BGR2GRAY)
		cv2.imshow('blur',image_cv)
		
		cv2.createTrackbar('s','blur',0,10,sliderHandler)
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(distance)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
		
	
		
	
	if (kernel_cv==ord('d')):
		print(image_cv.shape)
		lower_reso = cv2.resize(image_cv, (int(image_cv.shape[1]/2),int(image_cv.shape[0]/2)))
		print(lower_reso.shape)
		cv2.imshow('Modified_Image',lower_reso)
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(lower_reso)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
	
	if (kernel_cv==ord('D')):
		print(image_cv.shape)
		lower_reso = cv2.pyrDown(image_cv)
		print(lower_reso.shape)
		cv2.imshow('Modified_Image',lower_reso)
		g=cv2.waitkernel_cvey(0)
		if g==ord('w'):
			save(lower_reso)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
				
	if (kernel_cv==ord('x')):
		image_gray=cv2.cvtColor(image_cv,cv2.COLOR_BGR2GRAY)
		sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
		sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
		gx = cv2.filter2D(image_gray, -1, sobelx)
		gy = cv2.filter2D(image_gray, -1, sobely)
		g = np.sqrt(gx * gx + gy * gy)
		
		g *= 255.0 / np.max(g)
		cv2.imshow('x_derivative',gx)		
		g=cv2.waitKey(0)
		cv2.destroyAllWindows()
		if g==ord('w'):
			save(gx)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
		
		
	if (kernel_cv==ord('y')):
		image_gray=cv2.cvtColor(image_cv,cv2.COLOR_BGR2GRAY)
		sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
		sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
		gx = cv2.filter2D(image_gray, -1, sobelx)
		gy = cv2.filter2D(image_gray, -1, sobely)
		g = np.sqrt(gx * gx + gy * gy)

		g *= 255.0 / np.max(g)
		cv2.imshow('y_derivative',gy)		
		g=cv2.waitKey(0)
		cv2.destroyAllWindows()
		if g==ord('w'):
			save(gy)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
			
	if (kernel_cv==ord('m')):
		image_gray=cv2.cvtColor(image_cv,cv2.COLOR_BGR2GRAY)
		sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
		sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
		gx = cv2.filter2D(image_gray, -1, sobelx)
		gy = cv2.filter2D(image_gray, -1, sobely)
		
		g = np.sqrt(gx * gx + gy * gy)
		
		
		g *= 255.0 / np.max(g)
		print(g)
		
	if (kernel_cv==ord('r')):
		image_cv=cv2.cvtColor(image_cv,cv2.COLOR_BGR2GRAY)
		cv2.imshow('rotation',image_cv)

		cv2.createTrackbar('s','rotation',0,10,sliderHandler1)
		
		g=cv2.waitKey(0)
		if g==ord('w'):
			save(distance)
			m=cv2.waitKey(0)
			if(m==ord('i')):
				reload()
		if(g==ord('i')):
			reload()
		
	
	
	
	
	
	
	
	if (kernel_cv==ord('h')):
		print("--------------------------------------Description---------------------------------------------------------")
		print("The program loads an image by either reading from a file or capturing directly from camera. If a file name is specified in the command line then image should be read from it or else it captures image from camera. When capturing an image from the camera it continues to capture and process image continuously as a 3 channel color image and it works for any size.")
		print("----------------------------------Command Line Arguments---------------------------------------------------")
		print("We need to go to command prompt and give cmd as \"python\", followed by the relative filename and optional \"--path\" cmd argument followed by relative path name of the image. If path of image is not given then program will capture image from the web-cam.")
		print("------------------------------Example---------------------")
		#print("python C:\Users\adityayaji\Desktop\Assignments\CV\image_process_v4.py --path C:\Users\adityayaji\Desktop\Assignments\CV\image_5.png")
		print("--------------------------------------Shortcut-keys---------------------------------------------------------")
		print("'i'- reload the original image.")
		print("'w'- save the current image into the file 'out.jpg'")
		print("'g'- convert the image to grayscale using the openCV conversion function.")
		print("'G'- convert the image to grayscale using user defined implementation function.")
		print("'c'- cycle through the color channels of the image showing a different channel every time the key is pressed.")
		print("'s'- convert the image to grayscale and smoothing it using openCV function.")
		print("'S'- convert the image to grayscale and smoothing it using user defined function.")
		print("'d'- downsample the image by a factor of 2 without smoothing.")
		print("'D'- downsample the image by a factor of 2 with smoothing.")
		print("'x'- convert the image to grayscale and perform convolution with an x derivative filter.")
		print("'y'- convert the image to grayscale and perform convolution with an y derivative filter.")
		print("'m'- show the magnitude of the gradient normalized to the range [0,255].")
		print("'p'- convert the image to grayscale and plot the gradient vectors of the image every N pixels.")
		print("'r'- convert the image to grayscale and perform convolution with an y derivative filter.")
		print("'h'- displays short description of the program,its command line arguments, and the keys it supports.")
		
		
		
	
	
	
	