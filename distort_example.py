import numpy as np
import cv2
import time

image=0.1+np.random.randint(256,size=[300,300])

camera_matrix=np.array(
    [[1,0,200],[0,1,150],[0,0,1]])
distortion_coeff=np.array(
    [1e-3,1e-8,0,0])

tt=[time.time()]

for ii in range(100):
    _ = cv2.undistort(image, camera_matrix, distortion_coeff)
    tt.append(time.time())

print('mean time over 100 ops:', np.mean(np.diff(tt)))

