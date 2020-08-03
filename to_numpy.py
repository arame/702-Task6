import numpy as np

# Function to prevent warning caused by PILImage- no. writabletensor. - From pytorch forums

class ToNumpy(object):
    def __call__(self, img):
        return np.array(img)