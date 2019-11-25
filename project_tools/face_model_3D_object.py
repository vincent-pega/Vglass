import cv2
import numpy as np
import json

class FaceModel3D:
    # constant
    MODE_AXIS = 'axis'
    MODE_BOX = 'box'
    MODE_GLASSES = 'glasses'
    MODE_LANDMARK = 'landmark'
    COUNT_AXIS = 4
    COUNT_GLASSES = 12
    COUNT_BOX = 8
    COUNT_LANDMARK = 52
    # local parameter
    __face_count = 0
    __ID_list = [MODE_AXIS, MODE_BOX, MODE_GLASSES, MODE_LANDMARK]
    __modelIndexes = {}
    __ID = {}
    #glasses model
    __glasses_info = None
    __glasses = None
    __glasses_temple_left = None
    __glasses_temple_right = None
    __glasses_img_flag = False
    __glasses_temple_flag = False
    __glasses_offset = {'top':3, 'buttom':3, 'right':2, 'left':2, 'depth':30, 'temple_offset':4}
    # init
    def __init__(self, facemodel='./project_tools/3d_model.npy', k_means=True):
        self.__getLandmarkList(landmark_68=facemodel, k_means=k_means)
        for idx in self.__ID_list:
            self.__ID[idx] = True
        self.__updateModelIndexes()
        
    # get head 3D model
    def __getLandmarkList(self, landmark_68='./project_tools/3d_model.npy', k_means=False):
        self.__faces = np.load(landmark_68)
        self.__face_count = self.__faces.shape[0]
    # get value of face
    def getFace(self, indexes):
        return self.__faces[indexes]
    def getFaceCount(self):
        return self.__face_count
    
    # show ID selection
    def showID(self):
        print(self.__ID)
    
    # get tuple
    def __getTuple(self, local):
        return (local[0], local[1])
    
    # Interface
    __headIndex = None
    __beChanged = True
    
    # model indexes
    def getModelIndex(self, model='landmark'):
        if model in self.__ID_list:
            return self.__modelIndexes[model]
    def __updateModelIndexes(self):
        index = 0
        for idx in self.__ID_list:
            if idx == self.MODE_AXIS:
                self.__modelIndexes[self.MODE_AXIS] = range(index, index+self.COUNT_AXIS)
                index += self.COUNT_AXIS
            elif idx == self.MODE_BOX:
                self.__modelIndexes[self.MODE_BOX] = range(index, index+self.COUNT_BOX)
                index += self.COUNT_BOX
            elif idx == self.MODE_GLASSES:
                self.__modelIndexes[self.MODE_GLASSES] = range(index, index+self.COUNT_GLASSES)
                index += self.COUNT_GLASSES
            elif idx == self.MODE_LANDMARK:
                self.__modelIndexes[self.MODE_LANDMARK] = range(index, index+self.COUNT_LANDMARK)
                index += self.COUNT_LANDMARK
    
    def setParameter(self, settingDict=None):
        if settingDict is None:
            settingDict = self.__ID
        for key in settingDict:
            if key in self.__ID_list:
                if self.__ID[key] != settingDict[key]:
                    self.__ID[key] = settingDict[key]                
                    self.__beChanged = True
        if self.__beChanged:
            self.showID()
    
    def get(self, headIndex, settingDict=None):
        if settingDict is not None:
            self.setParameter(settingDict)
        if headIndex < 0 or headIndex >= self.__face_count:
            print('Input error : set headIndex value between 0 to {self.__face_count-1}')
            return None
        if self.__headIndex == headIndex:
            return self.__object3D
        object3D = np.empty((0, 3))
        face = self.__faces[headIndex]
        for idx in self.__ID_list:
            if idx == self.MODE_AXIS:
                object3D = np.append(object3D, self.__createAxis(), axis=0)
            elif idx == self.MODE_BOX:
                object3D = np.append(object3D, self.__createBox(face), axis=0)
            elif idx == self.MODE_GLASSES:
                object3D = np.append(object3D, self.__createGlasses(face), axis=0)
            elif idx == self.MODE_LANDMARK:
                object3D = np.append(object3D, face, axis=0)
        self.__object3D = np.asarray(object3D)
        self.__beChanged = False
        return self.__object3D
    
    def draw(self, img, end2D):
        end2D = np.asarray(end2D, dtype=np.int)
        self.__img_size = img.shape
        self.__line_width = int(max(round(min(self.__img_size[0], self.__img_size[1]) / 200), 1))
        for idx in self.__ID_list:
            if idx == self.MODE_AXIS:
                if self.__ID[idx]:
                    img = self.__drawAxis(img, end2D[self.__modelIndexes[self.MODE_AXIS]])
            elif idx == self.MODE_BOX:
                if self.__ID[idx]:
                    img = self.__drawBox(img, end2D[self.__modelIndexes[self.MODE_BOX]])
            elif idx == self.MODE_GLASSES:
                fix_id = self.__fixGlasses(end2D[self.__modelIndexes[self.MODE_GLASSES]], 
                                           end2D[self.__modelIndexes[self.MODE_LANDMARK]])
                if self.__ID[idx]:
                    img = self.__drawGlasses(img, end2D[self.__modelIndexes[self.MODE_GLASSES]], fix_id)
                if self.__glasses_img_flag:
                    img = self.__projTheGlasses(img, end2D[self.__modelIndexes[self.MODE_GLASSES]], fix_id)
            elif idx == self.MODE_LANDMARK:
                if self.__ID[idx]:
                    img = self.__drawLandmark(img, end2D[self.__modelIndexes[self.MODE_LANDMARK]])
        return img
    
    # Axis
    def __createAxis(self):
        return np.array([[0, 0, 0], [35, 0, 0], [0, 35, 0], [0, 0, 35]])
    def __drawAxis(self, img, axis):
        cv2.line(img, self.__getTuple(axis[0]), self.__getTuple(axis[1]), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, self.__getTuple(axis[0]), self.__getTuple(axis[2]), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(img, self.__getTuple(axis[0]), self.__getTuple(axis[3]), (0, 255, 0), 2, cv2.LINE_AA)
        return img
    
    # Glasses
    # Glasses location point
    def __createGlasses(self, face):
        eye_mid = np.mean(face[28:34], axis=0)
        eye_long = np.linalg.norm(face[31] - face[28]) / 2
        eye_short = np.linalg.norm(face[30] - face[32]) / 2
        z_h = face[19][2]
        ear = face[0]
        depth = self.__glasses_offset['depth']
        # create glasses
        glasses = []
        glasses.append(np.array([eye_mid[0]+self.__glasses_offset['left'], 
                                 eye_mid[1]+self.__glasses_offset['top'], z_h]))
        glasses.append(np.array([eye_mid[0]-self.__glasses_offset['right'], 
                                 eye_mid[1]+self.__glasses_offset['top'], z_h]))
        glasses.append(np.array([eye_mid[0]-self.__glasses_offset['right'], 
                                 eye_mid[1]-self.__glasses_offset['buttom'], z_h]))
        glasses.append(np.array([eye_mid[0]+self.__glasses_offset['left'], 
                                 eye_mid[1]-self.__glasses_offset['buttom'], z_h]))
        glasses.append(np.array([eye_mid[0]+self.__glasses_offset['left'], 
                           eye_mid[1]+self.__glasses_offset['top']+self.__glasses_offset['temple_offset'], 
                           depth]))
        glasses.append(np.array([eye_mid[0]+self.__glasses_offset['left'], 
                           eye_mid[1]-self.__glasses_offset['buttom']+self.__glasses_offset['temple_offset'], 
                           depth]))
        # mirror
        glasses_left = np.asarray(glasses)
        glasses_right = glasses_left.copy()
        glasses_right[:, 0] = -glasses_right[:, 0]
        return np.append(glasses_left, glasses_right, axis=0)
    def __drawGlasses(self, img, plot, fix_id, color=(255,0,0)):
        if fix_id[0] is not None:
            plot[4:6] = fix_id[0]
        if fix_id[1] is not None:
            plot[10:12] = fix_id[1]
        cv2.polylines(img, [plot[0:4]], True, color, self.__line_width, cv2.LINE_AA)
        cv2.polylines(img, [plot[6:10]], True, color, self.__line_width, cv2.LINE_AA)
        cv2.line(img, self.__getTuple(np.mean(plot[[4, 5]], axis=0, dtype=np.int)), 
                 self.__getTuple(np.mean(plot[[0, 3]], axis=0, dtype=np.int)), color, self.__line_width, cv2.LINE_AA)
        cv2.line(img, self.__getTuple(np.mean(plot[[6, 9]], axis=0, dtype=np.int)), 
                 self.__getTuple(np.mean(plot[[10, 11]], axis=0, dtype=np.int)), color, self.__line_width, cv2.LINE_AA)
        cv2.line(img, self.__getTuple(np.mean(plot[[1, 2]], axis=0, dtype=np.int)), 
                 self.__getTuple(np.mean(plot[[7, 8]], axis=0, dtype=np.int)), color, self.__line_width, cv2.LINE_AA)
        return img
    '''
    ┌┬┐    ┌┬┐          ┌┬┐   ┌┬┐          ┌┬┐    ┌┬┬┐
    └4┘++++└0┘++++++++++└1┘   └7┘++++++++++└6┘++++└10┘
            +            +     +            +
            +            +++++++            +
            +            +     +            +
    ┌5┐++++┌3┐++++++++++┌2┐   ┌8┐++++++++++┌9┐++++┌11┐
    └┴┘    └┴┘          └┴┘   └┴┘          └┴┘    └┴┴┘
    '''
    def __fixGlasses(self, glasses, face):
        left_glasses = np.mean(glasses[[0, 3]], axis=0)
        right_glasses = np.mean(glasses[[6, 9]], axis=0)
        LEFT_FACE = np.argmin(face[:, 0])
        RIGHT_FACE = np.argmax(face[:, 0])
        fix_id = []
        # print(LEFT_FACE, RIGHT_FACE)
        if face[LEFT_FACE][0] > left_glasses[0] and glasses[4][0] > face[LEFT_FACE][0]:
            # do fix
            s_1 = glasses[4][0] - face[LEFT_FACE][0]
            s_2 = face[LEFT_FACE][0] - left_glasses[0]
            fix_id.append((glasses[4:6] * s_2 + left_glasses * s_1) / (s_1 + s_2))
        else :
            fix_id.append(None)
        if right_glasses[0] > face[RIGHT_FACE][0] and glasses[10][0] < face[RIGHT_FACE][0]:
            # do fix
            s_1 = right_glasses[0] - face[RIGHT_FACE][0]
            s_2 = face[RIGHT_FACE][0] - glasses[10][0]
            fix_id.append((right_glasses * s_2 + glasses[10:12] * s_1) / (s_1 + s_2))
        else:
            fix_id.append(None)
        return fix_id
    # Glasses image projection
    def updateGlassesParameter(self, dict_paramter):
        try:
            for key in dict_paramter:
                self.__glasses_offset[key] += dict_paramter[key]
        except:
            pass
    def getGlassesImage(self):
        return self.__glasses_temple_left, self.__glasses, self.__glasses_temple_right
    def setGlassesModel(self, path=None, setup=True):
        if not setup:
            self.__glasses_img_flag = False
            return
        if path is not None:
            with open(path+'description.json') as fp:
                self.__glasses_info = json.load(fp)
            self.__glasses_offset = self.__glasses_info['parameter']
            temp = cv2.imread(path + self.__glasses_info['path'], cv2.IMREAD_UNCHANGED)
            self.__glasses = self.__modelResize(temp[:, self.__glasses_info['location']['left']:self.__glasses_info['location']['right'], :])
            self.__glasses_temple_left = self.__modelResize(temp[:, :self.__glasses_info['location']['left'], :])
            self.__glasses_temple_right = self.__modelResize(temp[:, self.__glasses_info['location']['right']:, :])
        if self.__glasses_temple_left is not None and self.__glasses_temple_right is not None:
            self.__glasses_temple_flag = True
        else:
            self.__glasses_temple_flag = False
        if self.__glasses is None:
            self.__glasses_img_flag = False
        else:
            self.__glasses_img_flag = setup
    def __projTheGlasses(self, img, glasses_proj, fix_id):
        if self.__glasses_temple_flag:
            img = self.__perspectiveTransform(img, self.__glasses_temple_left, 
                                              np.float32([glasses_proj[4], glasses_proj[5], glasses_proj[0], glasses_proj[3]]), 
                                              [fix_id[0], None])
            img = self.__perspectiveTransform(img, self.__glasses_temple_right, 
                                              np.float32([glasses_proj[6], glasses_proj[9], glasses_proj[10], glasses_proj[11]]), 
                                              [None, fix_id[1]])
        img = self.__perspectiveTransform(img, self.__glasses, 
                                          np.float32([glasses_proj[0], glasses_proj[3], glasses_proj[6], glasses_proj[9]]), [None, None])
        return img
    def __perspectiveTransform(self, img, object_img, object_proj, fix_id):
        size = object_img.shape
        max_g = np.argmax(object_img, axis=0)
        min_g = np.argmin(object_img, axis=0)
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [0, size[0]], 
                                                    [size[1], 0], [size[1], size[0]]]), 
                                        object_proj)
        object_Trans = cv2.warpPerspective(object_img, M, (self.__img_size[1], self.__img_size[0]))
        max_plot = np.asarray(np.max(object_proj, axis=0), dtype=np.int)
        min_plot = np.asarray(np.min(object_proj, axis=0), dtype=np.int)
        # fix with id
        if fix_id[1] is not None:
            min_plot[0] = fix_id[1][0][0]
        if fix_id[0] is not None:
            max_plot[0] = fix_id[0][0][0]
        # calculate
        Alpha = object_Trans[min_plot[1]:max_plot[1], min_plot[0]:max_plot[0], 3:4] / 255
        img[min_plot[1]:max_plot[1], min_plot[0]:max_plot[0], :] = np.asarray(object_Trans[min_plot[1]:max_plot[1], min_plot[0]:max_plot[0], :3] * Alpha + img[min_plot[1]:max_plot[1], min_plot[0]:max_plot[0], :] * (1 - Alpha), np.uint8)
        return img
    def __modelResize(self, model):
        size_y, size_x = model.shape[:2]
        model = cv2.resize(model, (int(size_x / 5), int(size_y / 5)), interpolation=cv2.INTER_AREA)
        return model
    
    # Box
    def __createBox(self, face):
        top = face[4][1] - 2 * (face[4][1] - (face[8][1] + face[0][1]) / 2)
        depth = face[25][2] - 3 * (face[25][2] - (face[0][2] + face[8][2]) / 2)
        point_3d = []
        point_3d.append((face[0][0], face[4][1], face[25][2]))
        point_3d.append((face[8][0], face[4][1], face[25][2]))
        point_3d.append((face[8][0], top, face[25][2]))
        point_3d.append((face[0][0], top, face[25][2]))
        point_3d.append((face[0][0], face[4][1], depth))
        point_3d.append((face[8][0], face[4][1], depth))
        point_3d.append((face[8][0], top, depth))
        point_3d.append((face[0][0], top, depth))
        point_3d = np.asarray(point_3d, dtype=np.float).reshape(-1, 3)
        return point_3d
    def __drawBox(self, img, plot, color=(0,255,0)):
        cv2.polylines(img, [plot[0:4]], True, (255, 0, 0), self.__line_width, cv2.LINE_AA)
        cv2.polylines(img, [plot[4:8]], True, color, self.__line_width, cv2.LINE_AA)
        cv2.line(img, self.__getTuple(plot[0]), self.__getTuple(plot[4]), color, self.__line_width, cv2.LINE_AA)
        cv2.line(img, self.__getTuple(plot[1]), self.__getTuple(plot[5]), color, self.__line_width, cv2.LINE_AA)
        cv2.line(img, self.__getTuple(plot[2]), self.__getTuple(plot[6]), color, self.__line_width, cv2.LINE_AA)
        cv2.line(img, self.__getTuple(plot[3]), self.__getTuple(plot[7]), color, self.__line_width, cv2.LINE_AA)
        return img
    
    # Landmark
    def __drawLandmark(self, img, plot, color=(0,0,255)):
        for p in plot:
            cv2.circle(img, self.__getTuple(p), self.__line_width+2, color, -1)
        return img