import numpy as np
import dlib
import cv2
from functools import partial
from os import listdir
from os.path import exists

from project_tools import *
from compare_source import draw as comp_draw

class glassesProject:
    __delta = 0.5
    __canvasShape = (500, 400)
    __galsses_path = './glasses/'
    
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./project_tools/shape_predictor_68_face_landmarks.dat')
        self.drawPose = headPose()
        # init the text canvas
        self.__textCanvas = cv2.imread('./project_tools/empty_img.png')
        
    def __setCanvasResize(self, size):
        y_shape = size[0] / 4.
        x_shape = size[1] / 6.
        canvasWidth = self.__canvasShape[0]
        canvasHeight = self.__canvasShape[1]
        canvasProp = canvasHeight / canvasWidth
        if y_shape / x_shape > canvasProp:
            scale = y_shape / canvasHeight
        else:
            scale = x_shape / canvasWidth
        self.__canvasResize = (int(canvasWidth * scale), int(canvasHeight * scale))
        
    def __drawEulerAngleText(self, img, euler_angle, compare_flag, detection=True):
        canvas = self.__textCanvas.copy()
        line_size = 2
        if not detection:
            cv2.putText(canvas, ' no detect', (0, 225),cv2.FONT_HERSHEY_TRIPLEX, line_size, (0, 0, 255), line_size, cv2.LINE_AA)
        elif compare_flag:
            cv2.putText(canvas, ' only 2D', (0, 225),cv2.FONT_HERSHEY_TRIPLEX, line_size, (255, 0, 0), line_size, cv2.LINE_AA)
        else:
            text1 = ' Pitch:{x[0]:4.0f}'.format(x=euler_angle)
            text2 = '  Yaw:{x[1]:4.0f}'.format(x=euler_angle)
            text3 = '  Roll:{x[2]:4.0f}'.format(x=euler_angle)
            cv2.putText(canvas, ' 3D enable', (0, 100),cv2.FONT_HERSHEY_TRIPLEX, line_size, (50, 200, 50), line_size, cv2.LINE_AA)
            cv2.putText(canvas, text1, (0, 175),cv2.FONT_HERSHEY_TRIPLEX, line_size, (255, 0, 0), line_size, cv2.LINE_AA)
            cv2.putText(canvas, text2, (0, 250),cv2.FONT_HERSHEY_TRIPLEX, line_size, (0, 0, 255), line_size, cv2.LINE_AA)
            cv2.putText(canvas, text3, (0, 325),cv2.FONT_HERSHEY_TRIPLEX, line_size, (50, 200, 50), line_size, cv2.LINE_AA)
        canvas = cv2.resize(canvas, self.__canvasResize, interpolation=cv2.INTER_CUBIC)
        img[-self.__canvasResize[1]:, -self.__canvasResize[0]:, :] = canvas
        return img
        
    def dlibLandmark(self, frame, num=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        result = []
        for i, face in enumerate(faces):
            if i >= num:
                break
            pred = self.predictor(gray, face)
            result.append(getDlibLandmark(pred))
        return result
    
#     # for xy
#     def fixAngle(self, angle):
#         np.where(angle<0, -180*(1+angle/np.pi), 180*(1-angle/np.pi))
#         return angle
    
    def fixAngle(self, angle):
        return angle * 180 / np.pi
    
    def doNothing(self, *args):
        pass
    
    def set_delta(self, delta=0.5):
        self.__delta = delta
        
    def __research_flag(self, proj1, proj2):
        p1_x1 = np.min(proj1[:, 0])
        p1_y1 = np.min(proj1[:, 1])
        p1_x2 = np.max(proj1[:, 0])
        p1_y2 = np.max(proj1[:, 1])
        p2_x1 = np.min(proj2[:, 0])
        p2_y1 = np.min(proj2[:, 1])
        p2_x2 = np.max(proj2[:, 0])
        p2_y2 = np.max(proj2[:, 1])
        area_p1 = (p1_x2 - p1_x1) * (p1_y2 - p1_y1)
        area_p2 = (p2_x2 - p2_x1) * (p2_y2 - p2_y1)
        area_mid = abs((min(p1_x2, p2_x2) - max(p1_x1, p2_x1)) * (min(p1_y2, p2_y2) - max(p1_y1, p2_y1)))
        IoU = area_mid / (area_p1 + area_p2 - area_mid)
        return IoU < 0.5
    
    def __getGlassesPath(self):
        paths = listdir(self.__galsses_path)
        paths_return = []
        for path in paths:
            file_path = self.__galsses_path+path+'/'
            if exists(file_path+'description.json'):
                paths_return.append(file_path)
        return paths_return

    def run(self, path='./testing_video.mp4', mode='video', save=False, save_path='./output.avi'):
        '''
        path : It can be http:// or file
        mode : 'video' or 'image'
        save : If 'True', than save the output image or video
        glasses : The glasses img
        '''
        try:
            cap = cv2.VideoCapture(int(path))
        except:
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return -1
        save_f = self.doNothing
        
        # setting
        size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        self.__setCanvasResize(size)
        line_size = max(round(min(size[0], size[1]) / 200), 1)
        self.drawPose.updateCameraMatrixWithSize(size)
        model_search = True
        if save:
            if mode == 'video':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(size, fps)
                out = cv2.VideoWriter(save_path, fourcc, fps, (size[0],size[1]))
                save_f = out.write
            elif mode == 'image':
                save_f = partial(cv2.imwrite, save_path)
            else:
                pass
        # glasses count
        glasses_paths = self.__getGlassesPath()
        glasses_count = len(glasses_paths)
        
        # create face_model_3D_object class
        FM3D = FaceModel3D(k_means=True)
        FM3D.setGlassesModel(path=None, setup=True)
        # FM3D.setGlassesModel(path=glasses_paths[1], setup=True)
        face_count = FM3D.getFaceCount()
        FM3D_ID = {'axis': True, 'box': False, 'glasses': False, 'landmark': False}
        end_proj = None
        
        # running
        compare_flag = False
        glasses_model_setup = False
        glasses_model_index = 0
        model_last_index = 0
        min_model_count = 0
        min_model = 20
        offset_add = 3
        no_face = 0
        while(1):
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
            if frame is None:
                break
            # waitkey
            get_key = cv2.waitKey(1) & 0xFF
            if get_key == ord('q'):
                break
            elif get_key == ord('1'):
                FM3D_ID['axis'] = not FM3D_ID['axis']
            elif get_key == ord('2'):
                FM3D_ID['box'] = not FM3D_ID['box']
            elif get_key == ord('3'):
                FM3D_ID['glasses'] = not FM3D_ID['glasses']
            elif get_key == ord('4'):
                FM3D_ID['landmark'] = not FM3D_ID['landmark']
            elif get_key == ord('5'):
                glasses_model_setup = not glasses_model_setup
                FM3D.setGlassesModel(path=glasses_paths[glasses_model_index], setup=glasses_model_setup)
            elif get_key == ord('n'):
                glasses_model_index += 1
                if glasses_model_index >= glasses_count:
                    glasses_model_index = 0
                FM3D.setGlassesModel(path=glasses_paths[glasses_model_index], setup=glasses_model_setup)
            elif get_key == ord('6'):
                compare_flag = not compare_flag
            elif get_key == ord('z'):
                FM3D.updateGlassesParameter({'top':offset_add})
            elif get_key == ord('a'):
                FM3D.updateGlassesParameter({'top':-offset_add})
            elif get_key == ord('x'):
                FM3D.updateGlassesParameter({'buttom':offset_add})
            elif get_key == ord('s'):
                FM3D.updateGlassesParameter({'buttom':-offset_add})
            elif get_key == ord('v'):
                FM3D.updateGlassesParameter({'temple_offset':offset_add})
            elif get_key == ord('f'):
                FM3D.updateGlassesParameter({'temple_offset':-offset_add})
            elif get_key == ord('b'):
                FM3D.updateGlassesParameter({'depth':-2*offset_add})
            elif get_key == ord('g'):
                FM3D.updateGlassesParameter({'depth':2*offset_add})
            elif get_key == ord('c'):
                FM3D.updateGlassesParameter({'right':offset_add, 'left':offset_add})
            elif get_key == ord('d'):
                FM3D.updateGlassesParameter({'right':-offset_add, 'left':-offset_add})
            elif get_key == ord('r'):
                model_search = True
            elif get_key == ord('m'):
                min_model = 7
            # compare
            if compare_flag:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dets = self.detector(gray, 0)
                #find face box bounding points
                for d in dets:

                    x = d.left()
                    y = d.top()
                    w = d.right()
                    h = d.bottom()
                dlib_rect = dlib.rectangle(x, y, w, h)
                detected_landmarks = self.predictor(gray, dlib_rect).parts()
                landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
                _, glasses, _ = FM3D.getGlassesImage()
                frame = comp_draw(frame, landmarks, glasses, x, y, w, h)
                frame = self.__drawEulerAngleText(frame, None, compare_flag, detection=len(landmarks)!=0)
                save_f(frame)
                cv2.imshow('frame', frame)
                continue
            else:
                landmark = self.dlibLandmark(frame)
            # make continue
            if len(landmark) == 0:
                if no_face < 20:
                    no_face += 1
                    if end_proj is not None:
                        frame = FM3D.draw(frame, end_proj)
                        frame = self.__drawEulerAngleText(frame, self.fixAngle(min_angle), compare_flag, detection=False)
                else:
                    model_search = True
                save_f(frame)
                cv2.imshow('frame', frame)
                continue
            else:
                no_face = 0
            # create and draw
            for i, face in enumerate(landmark):
                min_dist = float('inf')
                if model_search:
                    try:
                        if min_model_count > 5:
                            model_search = False
                            min_model_count = 0
                        print('model choose : ', min_model)
                        if model_last_index == min_model:
                            min_model_count += 1
                        else:
                            model_last_index = min_model
                            min_model_count = 0
                    except:
                        pass
                    face_count_temp = range(face_count)
                else:
                    face_count_temp =[min_model]
                for j in face_count_temp:
                    self.drawPose.updateParameter(model_points=FM3D.getFace(j), point_3d=FM3D.get(j, settingDict=FM3D_ID))
                    angle, camera_matrix, Matrix, end = self.drawPose.run(frame, face, updateCamM=False, withNp=True)
                    dist = dist_bt_2_face(face, end[FM3D.getModelIndex(model=FM3D.MODE_LANDMARK)])
                    if dist < min_dist:
                        min_dist = dist
                        min_end = end
                        min_model = j
                        min_angle = angle
                if end_proj is None:
                    if min_dist > 10:
                        continue
                    end_proj = min_end
                elif min_dist > 10:
                    end_proj = end_proj
                    model_search = True
                else:
                    end_proj = min_end * self.__delta + end_proj * (1 - self.__delta)
                    if self.__research_flag(end_proj, min_end):
                        model_search = True
                if min_dist < 10:
                    frame = FM3D.draw(frame, end_proj)
            frame = self.__drawEulerAngleText(frame, self.fixAngle(min_angle), compare_flag, detection=True)
            save_f(frame)
            cv2.imshow('frame', frame)
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Multimedia file location.', default='./testing_video.mp4')
    parser.add_argument('-m', '--mode', type=str, help='File type. Must be "video" or "image".', default='video')
    parser.add_argument('-s', action='store_true', help='-s to active the save mode')
    parser.add_argument('--save', type=str, help='The path to save the output.', default='./output.avi')
    args = parser.parse_args()
    project = glassesProject()
    project.run(path=args.path, mode=args.mode, save=args.s, save_path=args.save)