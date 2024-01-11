# Equirectangular Projection from Perspective Image

## Overview

This repo involves the development of a script that reverses the process of a perspective transformation. Given a perspective image, the code projects it back onto a 360° equirectangular canvas. This is particularly useful in the field of computer graphics and image processing where transformations between different views are required.

## Background

The perspective transformation is a common technique in computer vision that simulates the way images are captured by a camera. It accounts for the field of view and camera orientation. The reverse process of mapping a 2D perspective image onto a 3D equirectangular panorama can be complex due to the need to accurately map pixels in 3D space and account for the camera's intrinsic properties.

## Methodology

### Initial Perspective Transformation

The original process involved converting 3D points from camera-centric coordinates to longitude and latitude, and then to 2D image coordinates, simulating the extraction of a perspective view from a 360° equirectangular image.

### Reverse Transformation Process

The reverse process requires a series of steps to map 2D image coordinates back to 3D world coordinates:

1. **Understanding the Original Transformation**:
   - Analyze the existing code that performs the perspective transformation.
   
2. **Reversing the Mathematical Transformations**:
   - Utilize the inverse of the functions that convert from 3D to 2D coordinates.

3. **Implementing the `Perspective` Class**:
   - Initialize with the image, FOV, THETA, and PHI values.
   - Calculate the intrinsic camera matrix and the rotation matrices.
   - Implement the `GetEquirec` method that projects the perspective image back onto the equirectangular canvas.

4. **Refining Masking and Remapping**:
   - Ensure proper masking to keep regions outside the camera's FOV black.
   - Correctly remap pixels within the FOV onto the equirectangular canvas.

## Usage

To use this code, instantiate the `Perspective` class with your image and desired parameters, then call the `GetEquirec` method to obtain the equirectangular projection.

```python
persp = Perspective('path/to/perspective.jpg', 60, 0, 0)
img = persp.GetEquirec(3328, 6656)
cv2.imwrite('path/to/equirectangular_output.jpg', img)
```


## Requirements
```
Python 3.11
OpenCV (cv2) library
NumPy library
```
