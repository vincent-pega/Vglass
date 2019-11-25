import numpy as np
import math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def fixAngle(angle):
    index = np.where(angle > np.pi/2)
    angle[index] -= np.pi
    index = np.where(angle < -np.pi/2)
    angle[index] += np.pi
    return angle

# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    
    x_temp = np.repeat(np.array([[R[2, 1], R[2, 2]]]), 2, axis=0)
    y_temp = np.arcsin(-R[2, 0])
    z_temp = np.repeat(np.array([[R[1, 0], R[0, 0]]]), 2, axis=0)

    y = np.array([[y_temp, np.pi - y_temp]])
    cos_y = np.cos(y).T

    x_temp /= cos_y
    z_temp /= cos_y
    x_temp = np.split(x_temp.T, 2, axis=0)
    z_temp = np.split(z_temp.T, 2, axis=0)

    x = np.arctan2(x_temp[0], x_temp[1])
    z = np.arctan2(z_temp[0], z_temp[1])

    angle = np.append(np.append(x, y, axis=0), z, axis=0)
    # return fixAngle(angle.T[0])
    return angle.T[0]

# Calculates Rotation Matrix given euler angles [x, y, z].
def eulerAnglesToRotationMatrix(theta) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def angleInTwoVector(v1, v2):
    cos_theta = v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return np.arccos(cos_theta)

def rotationMatrixToXYAngle(Rotation):
    ori_v = np.array([0, 0, 1], np.float)
    rota_v = Rotation.dot(ori_v)
    # print(rota_v)
    x_angle = angleInTwoVector(ori_v[[0, 2]], rota_v[[0, 2]])
    y_angle = angleInTwoVector(ori_v[[1, 2]], rota_v[[1, 2]])
    final = np.array([x_angle, y_angle, 0])
    final = np.where(rota_v < 0, -final, final)[:-1]
    return final