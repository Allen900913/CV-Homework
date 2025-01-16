import os
import cv2
import glob
import time
import copy
import torch
import random
import threading
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

from itertools import cycle
from PIL import Image, ImageTk
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from torchvision import models, transforms
from torchsummary import summary
from tkinter import ttk
from CVDL_UI_MAC import Ui_MainWindow


class MainWindow_controller(QtWidgets.QMainWindow):
    '''signal1 = pyqtSignal() #clare'''

    ImageR = None
    ImageL = None
    FolderPath = None
    ImageOne = None
    ImageTwo = None
    InferenceImg = None

    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        '''self.signal1.connect(q1)'''
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    ## thread pool for manipulate things

    def setup_control(self):
        # TODO
        # Q1
        # qpushbutton doc: https://doc.qt.io/qt-5/qpushbutton.html
        # self.ui.pushButton.setText('Print message!')
        self.clicked_counter = 0
        self.ui.LoadFolder.clicked.connect(self.getFolderPath)
        self.ui.LoadImgL.clicked.connect(self.getImgL)
        self.ui.LoadImgR.clicked.connect(self.getImgR)
        self.ui.FindCorners.clicked.connect(self.Find_corner)
        self.ui.FindIntrinsic.clicked.connect(self.find_intrinsic)
        self.ui.FindExtrinsic.clicked.connect(self.find_extrinsic)
        self.ui.FindDistortion.clicked.connect(self.find_distortion)
        self.ui.ShowResult.clicked.connect(self.show_result)

        # Q2
        self.clicked_counter = 0
        self.ui.ShowWordsOnBoard.clicked.connect(self.on_clicked3)  # SHOW On board
        self.ui.ShowWordsVertically.clicked.connect(self.on_clicked4)

        # Q3
        self.ui.StereoDisparityMap.clicked.connect(self.stereo)  # SHOW On board

        # Q4
        self.ui.LoadImgOne.clicked.connect(self.getImgOne)
        self.ui.LoadImgTwo.clicked.connect(self.getImgTwo)
        self.ui.Keypoints.clicked.connect(self.drawKeypoints)
        self.ui.MatchedKeypoints.clicked.connect(self.matchKeypoints)

    # Load file/directory path
    def getFolderPath(self):
        folderpath = QtWidgets.QFileDialog.getExistingDirectory()
        self.FolderPath = folderpath
        print(str(self.FolderPath))

        self.ui.comboBox.clear()
        files = os.listdir(self.FolderPath)
        for f in range(1, len(files)+1):
            self.ui.comboBox.addItem(f"{f}.bmp")

    def getImgL(self):
        filename = QFileDialog.getOpenFileName()
        self.ImageL = filename[0]

    def getImgR(self):
        filename = QFileDialog.getOpenFileName()
        self.ImageR = str(filename[0])

    def getImgOne(self):
        filename = QFileDialog.getOpenFileName()
        self.ImageOne = filename[0]

    def getImgTwo(self):
        filename = QFileDialog.getOpenFileName()
        self.ImageTwo = filename[0]

    ### Q1 method
    @QtCore.pyqtSlot()  # THREAD!!!!!!!! if not using ,can't exe in real time
    def on_clicked(self):
        threading.Thread(target=self.Find_corner, daemon=True).start()

    @QtCore.pyqtSlot()
    def on_clicked2(self):
        threading.Thread(target=self.show_result, daemon=True).start()

    @QtCore.pyqtSlot()  # THREAD!!!!!!!! if not using ,can't exe in real time
    def on_clicked3(self):
        threading.Thread(target=self.Show_on_board, daemon=True).start()

    @QtCore.pyqtSlot()
    def on_clicked4(self):
        threading.Thread(target=self.Show_vertically, daemon=True).start()

    @QtCore.pyqtSlot()
    def on_clicked5(self):
        threading.Thread(target=self.matchKeypoints, daemon=True).start()


    ### Q1 method start
    def Find_corner(self):
        '''self.signal1.emit()'''
        self.ui.HintLabel.setText("Processing...")
        # Defining the dimensions of checkerboard
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        path = os.path.join(self.FolderPath, "*.bmp")
        images = glob.glob(path, recursive=True)

        output_dir = "./out/Q1/"
        os.makedirs(output_dir, exist_ok=True)

        i = 1
        for image in images:
            img = cv2.imread(image)
            h, w, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

            img = cv2.resize(img, (w // 2, h // 2))
            cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), img)
            i += 1


        frame_path = "./out/Q1/"
        filenames = os.listdir(frame_path)
        img_iter = (cv2.imread(os.path.join(frame_path, x)) for x in filenames)

        cv2.namedWindow('FindCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FindCorners', int(w // 2), int(h // 2))
        for image in img_iter:
            cv2.imshow('FindCorners', image)
            cv2.waitKey(500)

        self.ui.HintLabel.setText("")
        k = cv2.waitKey(0) & 0xFF
        if k == 27 :
            cv2.destroyAllWindows()

    def find_extrinsic(self):
        
        # Defining the dimensions of checkerboard
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        path = os.path.join(self.FolderPath, "*.bmp")
        images = glob.glob(path)
        print(images)
        i = 1

        images.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))

        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        f = self.ui.comboBox.currentText()

        imgPath = os.path.join(self.FolderPath, f)


        image = cv2.imread(imgPath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print(len(rvecs))

        rvec, tvec = rvecs[int(f.split(".")[0])-1], tvecs[int(f.split(".")[0])-1]
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.hstack((rotation_matrix, tvec))

        # extrinsic_matrix2 = np.concatenate((rotation_matrix, tvecs[i]), axis=1)

        print("\n" + imgPath)
        print(" Extrinsic Matrix : \n ")
        print(extrinsic_matrix)

        return
    
    def find_intrinsic(self):

        # Defining the dimensions of checkerboard
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        path = os.path.join(self.FolderPath, "*.bmp")
        images = glob.glob(path)

        i = 1

        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

            # height, width, channel = img.shape
            img = cv2.resize(img, (600, 500))

        # self.show()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("\n Intrinsic Matrix : \n")
        print(mtx)

        return

    def show_result(self):
        print("Processing Result...")
        self.ui.HintLabel.setText("Processing...")
        # Defining the dimensions of checkerboard
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        path = os.path.join(self.FolderPath, "*.bmp")
        images = glob.glob(path)

        h, w, _ = cv2.imread(images[0]).shape

        frame_distorted_path = "./out/Q1-distorted/"
        frame_undistorted_path = "./out/Q1-undistorted/"
        os.makedirs(frame_distorted_path, exist_ok=True)
        os.makedirs(frame_undistorted_path, exist_ok=True)

        i = 1
        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

            img = cv2.resize(img, (w // 2, h // 2))
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            dst = cv2.undistort(img, mtx, dist, None, mtx)

            cv2.imwrite(os.path.join(frame_distorted_path, f'Noncalibresult{i}.png'), img)
            cv2.imwrite(os.path.join(frame_undistorted_path, f'calibresult{i}.png'), dst)
            i += 1


        filenames_distorted = os.listdir(frame_distorted_path)
        filenames_undistorted = os.listdir(frame_undistorted_path)

        cv2.namedWindow('Distorted', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Distorted', int(w // 2), int(h // 2))

        cv2.namedWindow('Undistorted', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Undistorted', int(w // 2), int(h // 2))

        for i in range(1, len(filenames_distorted)+1):
            cv2.imshow('Distorted', cv2.imread(os.path.join(frame_distorted_path, f"Noncalibresult{i}.png")))
            cv2.imshow('Undistorted', cv2.imread(os.path.join(frame_undistorted_path, f"calibresult{i}.png")))
            cv2.waitKey(800)

        self.ui.HintLabel.setText("")
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()

    def find_distortion(self):

        # Defining the dimensions of checkerboard
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        path = os.path.join(self.FolderPath, "*.bmp")
        images = glob.glob(path)

        i = 1

        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

            # height, width, channel = img.shape
            img = cv2.resize(img, (600, 500))

        # self.show()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("\n Distortion : \n")
        print(dist)
        return    
    ### Q1 method end

    ### Q2 method start
    def Show_on_board(self):

        msg = self.ui.InputText.toPlainText()  # msg為字串
        str_length = len(msg)

        # Defining the dimensions of checkerboard
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        matrix_list = []

        for idx, c in enumerate(msg):
            matrix = self.Get_Alphabet_and_shift(c, idx + 1, 0)
            matrix_list.append(matrix)

        obj = matrix_list[0]
        for i in range(1, len(matrix_list)):
            obj = copy.deepcopy(np.append(obj, matrix_list[i], axis=0))

        alphabet_set = copy.deepcopy(obj)

        dim0 = alphabet_set.shape[0]
        dim1 = alphabet_set.shape[1]

        alphabet_set = np.reshape(alphabet_set, (dim0 * dim1, 3))

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        # Extracting path of individual image stored in a given directory
        # images = glob.glob('./Q2_Image/*.bmp')

        path = os.path.join(self.FolderPath, "*.bmp")
        images = glob.glob(path, recursive=True)

        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # -------------------------------------------------------------------------------------
        # images = glob.glob('./Q2_Image/*.bmp')
        path = os.path.join(self.FolderPath, "*.bmp")
        images = glob.glob(path, recursive=True)

        for image in images:  # when coefs get, draw.
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

                imgpts, jac = cv2.projectPoints(alphabet_set, rvecs, tvecs, mtx, dist)

                imgpts = np.reshape(imgpts, (dim0, dim1, 2))

                img = self.draw(img, corners2, imgpts)

            img = cv2.resize(img, (250, 250))

            cv2.imwrite("./out/SAVE.bmp".format(i), img)

            pixmap = QPixmap("./out/SAVE.bmp")
            self.ui.GraphLabel.setPixmap(pixmap)
            time.sleep(1)

    def Show_vertically(self):

        msg = self.ui.InputText.toPlainText()  # msg為字串
        str_length = len(msg)

        # Defining the dimensions of checkerboard
        CHECKERBOARD = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        str_length = len(msg)
        matrix_list = []

        for idx, c in enumerate(msg):
            matrix = self.Get_Alphabet_and_shift(c, idx + 1, 1)
            matrix_list.append(matrix)

        obj = matrix_list[0]
        for i in range(1, len(matrix_list)):
            obj = copy.deepcopy(np.append(obj, matrix_list[i], axis=0))

        alphabet_set = copy.deepcopy(obj)

        dim0 = alphabet_set.shape[0]
        dim1 = alphabet_set.shape[1]

        alphabet_set = np.reshape(alphabet_set, (dim0 * dim1, 3))

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        # Extracting path of individual image stored in a given directory
        path = os.path.join(self.FolderPath, "*.bmp")
        images = glob.glob(path, recursive=True)
        # images = glob.glob('./Q2_Image/*.bmp')

        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # -------------------------------------------------------------------------------------

        # images = glob.glob('./Q2_Image/*.bmp')
        path = os.path.join(self.FolderPath, "*.bmp")
        images = glob.glob(path, recursive=True)

        for image in images:  # when coefs get, draw.
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

                imgpts, jac = cv2.projectPoints(alphabet_set, rvecs, tvecs, mtx, dist)

                imgpts = np.reshape(imgpts, (dim0, dim1, 2))

                img = self.draw(img, corners2, imgpts)

            img = cv2.resize(img, (250, 250))

            cv2.imwrite("./out/SAVE.bmp".format(i), img)

            pixmap = QPixmap("./out/SAVE.bmp")
            self.ui.GraphLabel.setPixmap(pixmap)
            time.sleep(0.8)

    def draw(self, img, corners, imgpts):
            '''   image = cv2.line(image, start_point, end_point, color, thickness)  '''
            imgpts = np.float32(imgpts).reshape(-1, 2)
            imgpts = imgpts.astype(np.int64)

            w = imgpts.shape[0]

            for i in range(0, w, 2):
                img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i + 1]), (0, 0, 125), 50)

            return img
    
    def Get_Alphabet_and_shift(self, alphabet, ith, func_number):

        libPath0 = self.FolderPath + "/Q2_db/alphabet_db_onboard.txt"
        libPath1 = self.FolderPath + "/Q2_db/alphabet_db_vertical.txt"

        if (func_number == 0):

            fs = cv2.FileStorage(libPath0, cv2.FILE_STORAGE_READ)

        if (func_number == 1):

            fs = cv2.FileStorage(libPath1, cv2.FILE_STORAGE_READ)

        alphabet_matrix = fs.getNode('{}'.format(alphabet)).mat()  # get the lines of 'K'

        alphabet_matrix = alphabet_matrix.astype(np.float32)

        if (ith <= 3):
            y = 5
        else:
            y = 2

        if (ith % 3 == 1):
            x = 7
        elif (ith % 3 == 2):
            x = 4
        else:
            x = 1

        # y平移
        for i in range(alphabet_matrix.shape[0]):
            alphabet_matrix[i, 0, 1] += y
            alphabet_matrix[i, 1, 1] += y
        # x平移
        for i in range(alphabet_matrix.shape[0]):
            alphabet_matrix[i, 0, 0] += x
            alphabet_matrix[i, 1, 0] += x

        return alphabet_matrix
    ### Q2 method end

    ### Q3 method start
    def stereo(self):

        if (not self.ImageR or not self.ImageL):
            print("Image are not properly loaded, Please Check image resource!")
            return

        def draw_circle(event, x, y, flags, param):
            global mouseX, mouseY

            if event == cv2.EVENT_LBUTTONDOWN:

                img = cv2.cvtColor(np.copy(disparity), cv2.COLOR_GRAY2BGR)
                img_dot = cv2.cvtColor(np.copy(disparity), cv2.COLOR_GRAY2BGR)
                cv2.circle(img_dot, (x, y), 10, (255, 0, 0), -1)

                mouseX, mouseY = x, y
                depth = baseline * focal_length / (img[y][x][0] + doffs)
                print(f"(x, y) = ({x}, {y}) , dis = {depth}")
                self.ui.HintLabel.setText(f"({x}, {y})\ndis = {depth:.3f}")

                # maybe bug here
                imgR_dot = cv2.imread(self.ImageR)
                z = img[y][x][0]

                if img[y][x][0] != 0:
                    cv2.circle(imgR_dot, (x - z, y), 25, (0, 255, 0), -1)

                cv2.namedWindow('imgR', cv2.WINDOW_NORMAL)
                cv2.resizeWindow("imgR", int(imgR_dot.shape[1] / 4), int(imgR_dot.shape[0] / 4))
                cv2.imshow('imgR', imgR_dot)

                k = cv2.waitKey(0) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()


        imgL = cv2.imread(self.ImageL, 0)
        imgR = cv2.imread(self.ImageR, 0)

        baseline = 342.789  # mm
        doffs = 279.184  # pixel
        focal_length = 4019.284  # pixel

        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL, imgR)
        disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        w, h = int(disparity.shape[1]), int(disparity.shape[0])
        w, h = w//8 , h//8

        imgR = cv2.imread(self.ImageR)
        cv2.namedWindow("imgR", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("imgR", w, h)
        # cv2.setMouseCallback('imgR', draw_circle)
        cv2.imshow('imgR', imgR)

        imgL = cv2.imread(self.ImageL)
        cv2.namedWindow("imgL", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("imgL", w, h)
        cv2.setMouseCallback('imgL', draw_circle)
        cv2.imshow('imgL', imgL)

        cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("disparity", w, h)
        cv2.imshow('disparity', disparity)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
    ### Q3 method end

    ### Q4 method start
    def drawKeypoints(self):
        self.ui.HintLabel.setText("Processing...")
        img = cv2.imread(self.ImageOne)

        # Converting image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Applying SIFT detector
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)

        # Marking the keypoint on the image using circles
        img = cv2.drawKeypoints(gray,
                                kp,
                                img,
                                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
                                color=(0, 255, 0))

        cv2.namedWindow('image-with-keypoints', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image-with-keypoints', int(img.shape[1] / 4), int(img.shape[0] / 4))
        cv2.imshow('image-with-keypoints', img)
        self.ui.HintLabel.setText("")
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()

    def matchKeypoints(self):
        # load the images
        self.ui.HintLabel.setText("Processing...")
        image1 = cv2.imread(self.ImageOne)
        image2 = cv2.imread(self.ImageTwo, 0)

        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        keypoint1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoint2, descriptors2 = sift.detectAndCompute(image2, None)

        # Initialize the BFMatcher for matching
        BFMatch = cv2.BFMatcher()
        Matches = BFMatch.knnMatch(descriptors1, descriptors2, k=2)

        # Need to draw only good matches, so create a mask
        good_matches = [[0, 0] for i in range(len(Matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(Matches):
            if m.distance < 0.75 * n.distance:
                good_matches[i] = [1, 0]

                # Draw the matches using drawMatchesKnn()
        Matched = cv2.drawMatchesKnn(image1,
                                     keypoint1,
                                     image2,
                                     keypoint2,
                                     Matches,
                                     outImg=None,
                                     matchesMask=good_matches,
                                     flags=2
                                     )
        # Save the image
        cv2.namedWindow('BFMatchs', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('BFMatchs', int(Matched.shape[1] / 4), int(Matched.shape[0] / 4))
        cv2.imshow('BFMatchs', Matched)
        self.ui.HintLabel.setText("")
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()

        return
    ### Q4 method end