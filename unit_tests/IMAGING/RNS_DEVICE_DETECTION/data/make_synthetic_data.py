import math
import numpy as np
import pylab as PLT

class generate_device:

    def __init__(self):
        pass

    def find_device_coords(self):

        # Define the fill-value
        fill_value = self.ellipse_array.max()

        # Find a candidate starting position
        maxind              = self.ellipse_array.argmax()
        self.coordinate_one = np.unravel_index(maxind,self.ellipse_array.shape)

        # Find the second point for the base of the rectangle
        dy   = 50
        tmp  = 1-(((self.coordinate_one[0]+dy)-self.y0)**2)/(self.b**2)
        tmp *= self.a**2
        tmp  = -1*np.sqrt(tmp)
        tmp += self.x0
        self.coordinate_two = (self.coordinate_one[0]+dy,int(tmp))

        # Find the angle on the skull, and the angle that defines the rectangular device
        angle      = np.arctan((self.coordinate_two[0]-self.coordinate_one[0])/(self.coordinate_two[1]-self.coordinate_one[1]))
        self.theta = (np.pi/2.)-angle

        # Find the length of the device
        self.device_length = int(np.sqrt((self.coordinate_two[0]-self.coordinate_one[0])**2+(self.coordinate_two[1]-self.coordinate_one[1])**2))

    def make_device(self,device_height=20):

        # Make the device array
        self.device_array = np.zeros(self.ellipse_array.shape)
        
        # Make the coordinate map
        x_raw  = np.arange(self.device_length)
        y_raw  = np.arange(device_height)
        xv, yv = np.meshgrid(x_raw,y_raw)
        xv     = xv.flatten().reshape((-1,1))
        yv     = yv.flatten().reshape((-1,1))

        # Rotate the coordinate map
        xv_rot = (xv*np.cos(-1*self.theta)-yv*np.sin(-1*self.theta)).astype('int')
        yv_rot = (xv*np.sin(-1*self.theta)+yv*np.cos(-1*self.theta)).astype('int')

        for idx in range(xv.size):
            self.device_array[yv_rot[idx],xv_rot[idx]] = 1

        PLT.imshow(self.device_array,origin='lower')
        PLT.show() 

class generate_skull(generate_device):

    def __init__(self,in_array):

        self.in_array = in_array

    def find_center(self):
        self.x0 = self.in_array.shape[1]/2
        self.y0 = self.in_array.shape[0]/2

    def make_axes(self):
        self.a = self.in_array.shape[1]/4.
        self.b = self.in_array.shape[0]/3.

    def check_ellipse(self,ix,iy,tol=1e-2):
        isum = ((ix-self.x0)**2/self.a**2)+((iy-self.y0)**2/self.b**2)
        if np.fabs(isum-1)<tol:
            return isum
        else:
            return 0

    def make_ellipse(self):
        self.ellipse_array = np.zeros(self.in_array.shape)
        for ix in range(self.in_array.shape[1]):
            for iy in range(self.in_array.shape[0]):
                self.ellipse_array[iy,ix] = self.check_ellipse(ix,iy)
        
    def generate_image(self):
        self.find_center()
        self.make_axes()
        self.make_ellipse()

    def generate_device_image(self):
        generate_device.find_device_coords(self)
        generate_device.make_device(self)

        #PLT.imshow(self.ellipse_array)
        #PLT.show()

if __name__ == '__main__':

    # Define the blank array for creating the image
    np.random.seed(42)
    image_array = np.zeros((513,513))

    # Make the skull
    GS = generate_skull(image_array)
    GS.generate_image()

    # Make the device
    GS.generate_device_image()