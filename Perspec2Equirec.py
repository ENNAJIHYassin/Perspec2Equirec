import cv2
import numpy as np

def XY2lonlat(XY, shape):
    # Normalize X and Y to [-1, 1]
    X = XY[..., 0:1]
    Y = XY[..., 1:2]  # Corrected slicing

    # Normalize based on the shape of the equirectangular image
    X_normalized = (X / (shape[1] - 1) - 0.5) * 2
    Y_normalized = (Y / (shape[0] - 1) - 0.5) * 2

    # Convert back to spherical coordinates
    lon = X_normalized * np.pi
    lat = Y_normalized * np.pi / 2

    # Combine into a single array
    lonlat = np.concatenate([lon, lat], axis=-1)

    return lonlat

def lonlat2xyz(lonlat):
    lon = lonlat[..., 0:1]
    lat = lonlat[..., 1:]

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    xyz = np.concatenate([x, y, z], axis=-1)
    return xyz


class Perspective:
    def __init__(self, img_name, FOV, THETA, PHI):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        self.FOV = FOV
        self.THETA = THETA
        self.PHI = PHI

        # Camera matrix
        f = 0.5 * self._width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (self._width - 1) / 2.0
        cy = (self._height - 1) / 2.0
        self.K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1],
        ], np.float32)
        self.K_inv = np.linalg.inv(self.K)

        # Rotation matrices
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        self.R = R1 @ R2
        self.R_inv = np.linalg.inv(self.R)

    def GetEquirec(self, height, width):
        # Create an equirectangular canvas
        equirec = np.zeros((height, width, 3), dtype=np.uint8)

        # Generate grid
        x = np.linspace(-np.pi, np.pi, width)
        y = np.linspace(-np.pi/2, np.pi/2, height)
        x, y = np.meshgrid(x, y)

        # Convert to 3D Cartesian coordinates
        xyz = np.stack([np.cos(y) * np.sin(x), np.sin(y), np.cos(y) * np.cos(x)], axis=-1)

        # Apply the inverse rotation
        xyz = xyz @ self.R.T

        # Project onto the image plane and normalize
        uv = xyz[..., :2] / xyz[..., 2:3]

        # Apply the intrinsic matrix to get image coordinates
        uv = uv @ self.K[:2, :2].T + self.K[:2, 2]

        # Ensure that we only consider points where z > 0 to avoid wrap-around issues
        mask = xyz[..., 2] > 0

        # Remap the pixels
        for i in range(height):
            for j in range(width):
                if mask[i, j]:
                    u, v = uv[i, j]
                    if 0 <= u < self._width and 0 <= v < self._height:
                        equirec[i, j] = self._img[int(v), int(u)]

        return equirec



if __name__ == '__main__':
    # Load perspective image into class
    persp = Perspective('data/sample/perspective.jpg', 60, 0, 0)    # Specify parameters(FOV, theta, phi

    #
    # FOV unit is degree
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension
    #
    img = persp.GetEquirec(3328, 6656)
    cv2.imwrite('data/result/equirec_out.jpg', img)