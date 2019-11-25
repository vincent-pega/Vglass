import cv2
import numpy as np
from .rotm2euler import rotationMatrixToEulerAngles, fixAngle, eulerAnglesToRotationMatrix

class headPose:
    def __init__(self, model_points=None, camera_matrix=None, dist_coeffs=None):
        self.init()
        self.updateBoxSize()
        self.updateParameter(model_points=model_points, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    def detail(self):
        print('model_points = ', model_points)
        print('camera_matrix = ', camera_matrix)
        print('dist_coeffs = ', dist_coeffs)
    def init(self):
        self.__model_points = np.array([
                                            (0.0, 0.0, 0.0),             # Nose tip
                                            (0.0, -330.0, -65.0),        # Chin
                                            (-225.0, 170.0, -135.0),     # Left eye left corner
                                            (225.0, 170.0, -135.0),      # Right eye right corne
                                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                                            (150.0, -150.0, -125.0)      # Right mouth corner
                                        ])
        self.__camera_matrix = None
        self.__dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    def updateParameter(self, model_points=None, camera_matrix=None, dist_coeffs=None, point_3d=None):
        if model_points is not None:
            self.__model_points = model_points
        if camera_matrix is not None:
            self.__camera_matrix = camera_matrix
        if dist_coeffs is not None:
            self.__dist_coeffs = dist_coeffs
        if point_3d is not None:
            self.__point_3d = point_3d
    def updateCameraMatrixWithSize(self, size):
        center = (size[1]/2, size[0]/2)
        self.__camera_matrix = np.array(
                                             [[size[1], 0, center[0]],
                                             [0, size[1], center[1]],
                                             [0, 0, 1]], dtype = "double"
                                         )
    def updateBoxSize(self, size=350, withFace=False):
        '''
        size : the box size
        
        note that need to match point_3d[(0, 3), (2, 5), (1, 6)]
        '''
        if withFace:
            self.__point_3d = self.__model_points
        else:
            rear_depth = 0
            front_depth = size
            rear = int(size * 0.75)
            front = int(size)
            point_3d = []
            point_3d.append((-rear, -rear, rear_depth))
            point_3d.append((-rear, rear, rear_depth))
            point_3d.append((rear, rear, rear_depth))
            point_3d.append((rear, -rear, rear_depth))
            point_3d.append((front, -front, front_depth))
            point_3d.append((front, front, front_depth))
            point_3d.append((-front, front, front_depth))
            point_3d.append((-front, -front, front_depth))
            point_3d = np.asarray(point_3d, dtype=np.float).reshape(-1, 3)
            self.__point_3d = point_3d
    def fixRotation(self, Rotation):
        angle = rotationMatrixToEulerAngles(Rotation)
        new_angle = fixAngle(angle)
        if len(np.where(angle==new_angle)[0]) != 0:
            Rotation = eulerAnglesToRotationMatrix(new_angle)
        return Rotation
    def run(self, img, image_points, updateCamM=False, color=(0, 255, 255), withNp=False):
        '''
        If you update your own point_3d, please let withNp be True, and draw the box by using output
        '''
        if updateCamM:
            size = img.shape
            self.updateCameraMatrixWithSize(size)
            print(size)
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.__model_points, image_points, self.__camera_matrix, self.__dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        Rotation, jacobian = cv2.Rodrigues(rotation_vector)
        if withNp:
            Rotation = self.fixRotation(Rotation)
            point_3d = np.append(self.__point_3d, np.ones((len(self.__point_3d), 1)), axis=1).T
            Matrix = np.append(Rotation, translation_vector, axis=1)
            ## This blog is to re-calculate the point_2d.
            end = np.dot(Matrix, point_3d)
            end = np.dot(self.__camera_matrix, end)
            end = (end / end[2]).T
            end = end[:, 0:2]
            # end = np.squeeze(end, axis=1)
            return rotationMatrixToEulerAngles(Rotation), self.__camera_matrix, Matrix, end
        else:
            (point_2d, jacobian) = cv2.projectPoints(self.__point_3d, rotation_vector, translation_vector, self.__camera_matrix, self.__dist_coeffs)
            # Draw all lines
            line_width = 5
            point_2d = np.squeeze(np.asarray(point_2d, dtype=np.int), axiz=1)
            cv2.polylines(img, [point_2d[0:4]], True, color, line_width, cv2.LINE_AA)
            cv2.line(img, point_2d[3], point_2d[4], color, line_width, cv2.LINE_AA)
            cv2.line(img, point_2d[2], point_2d[5], color, line_width, cv2.LINE_AA)
            cv2.line(img, point_2d[1], point_2d[6], color, line_width, cv2.LINE_AA)
            cv2.line(img, point_2d[0], point_2d[7], color, line_width, cv2.LINE_AA)
            cv2.polylines(img, [point_2d[4:8]], True, (0, 0, 255), line_width, cv2.LINE_AA)
            return img