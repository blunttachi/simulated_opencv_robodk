from robodk import robolink  # RoboDK API
from robodk import robomath  # Robot toolbox
from robodk import robodialogs  # Dialogs
from robodk.robomath import *


import cv2 as cv
from cv2 import aruco
import numpy as np
from math import atan2, cos, sin, sqrt, pi
from enum import Enum
from matplotlib import pyplot as plt



class FeatureDesc(Enum):
    ORB = 0
    SIFT = 1

class CalibType(Enum):
    SIMPLE = 0
    HANDEYE = 1

CALIB_TYPE = CalibType.HANDEYE
FEATURES = FeatureDesc.SIFT

items = [{"item" : "Zdenka", "pic" : "test3\item3.jpg", "model" : "model1"}, 
        {"item" : "Nescafe", "pic" : "test3\item1.jpg", "model" : "model2"},
        {"item" : "Metar", "pic" : "test3\item4.jpg", "model" : "model3"},
        {"item" : "Crta", "pic" : "test3\item2.jpg", "model" : "model4"}]

cam_mtx4 = np.array([[8878.5308, 0, 1565.2929],
                     [0, 8871.6445, 1053.1138],
                     [0, 0, 1]])

dist4 = np.array([[2.35942, -73.06175, -0.0373206, 0.00206349, -44.78099]])

cam_mtx3 = np.array([[1424.90038, 0, 1509.85319],
 [0, 1429.04338, 1096.18839],
 [0, 0, 1]
])

dist3 = np.array([[0.07364505, -0.10243549, -0.00359098, -0.01113669, 0.05093577]])


cam_mtx5 = np.array([[1375.3645, 0, 1579.8830],
 [0, 1377.7314, 1126.9327],
 [0, 0, 1]
])

dist5 = np.array([[0.0430375, -0.07167579, 0.00187173, 0.00160114, 0.01481508]])




clahe = cv.createCLAHE(clipLimit=1.2, tileGridSize=(5,5))

result = None

def getCenter(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    cv.circle(img, cntr, 15, (255, 0, 255), 4)

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return cntr, angle


def get_scene():
    cam = cv.VideoCapture(1, cv.CAP_DSHOW)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 3072)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 2048)
    success, img = cam.read()
    # cv.imshow("VIew", img)
    # cv.waitKey()
    return img


def describe_kp(img):
    if FEATURES == FeatureDesc.ORB:
        detector = cv.ORB_create(10000)

        # index_params = dict(algorithm=6,
        #                     table_number=6,
        #                     key_size=12,
        #                     multi_probe_level=2)
        # search_params = {}
        # matcher = cv.FlannBasedMatcher(index_params, search_params)


    elif FEATURES == FeatureDesc.SIFT:
        detector = cv.SIFT_create()
        # matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
  

    keypoints, descriptors = detector.detectAndCompute(img, None)

    return keypoints, descriptors


def find_object(scene, item, kp_scene, desc_scene):

    img_object = cv.imread(item, cv.IMREAD_GRAYSCALE)
    img_scene = scene

    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)

    # img_object_stock = img_object.copy()
    # img_scene_stock = img_scene.copy()


    # clahe = cv.createCLAHE(clipLimit=1.2, tileGridSize=(5,5))
    
    img_object = clahe.apply(img_object)

    '''
    #downsize for speed improvements
    # img_scene = cv.resize(img_scene, None, fx=0.8, fy=0.8, interpolation=cv.INTER_LINEAR)
    # img_object = cv.resize(img_object, None, fx=0.8, fy=0.8, interpolation=cv.INTER_LINEAR)


    #cleaning for contours
    #USING NON CLAHE IMAGE!!
    # blur_item = cv.GaussianBlur(img_object, (13,13),0)
    # # blur_scene = cv.GaussianBlur(img_scene, (13,13),0)
    # th_item = cv.adaptiveThreshold(blur_item,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,7,2)
    # # th_scene = cv.adaptiveThreshold(blur_scene,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    # # plt.imshow(th_item), plt.show()
    # # plt.imshow(th_scene), plt.show()

    # blur_item = cv.medianBlur(th_item,3)
    # # blur_scene = cv.medianBlur(th_scene,7)

    # kernel = np.ones((3,3), np.uint8)
    # # erode_item = cv.erode(blur_item, kernel, iterations=2)
    # # kernel = np.ones((3,3), np.uint8)
    # # erode_scene = cv.erode(blur_scene, kernel, iterations=2)

    # kernel = np.ones((3,3), np.uint8)
    # mask_item = cv.morphologyEx(blur_item, cv.MORPH_OPEN, kernel)
    # # kernel = np.ones((7,7), np.uint8)
    # # mask_scene = cv.morphologyEx(erode_scene, cv.MORPH_OPEN, kernel)


    # invert_item = cv.bitwise_not(mask_item)
    # invert_scene = cv.bitwise_not(mask_scene)
    # plt.imshow(invert_item), plt.show()
    # plt.imshow(invert_scene), plt.show()


    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    '''

    if FEATURES == FeatureDesc.ORB:
        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        search_params = {}
        matcher = cv.FlannBasedMatcher(index_params, search_params)


    elif FEATURES == FeatureDesc.SIFT:
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
  

    keypoints_obj, descriptors_obj = describe_kp(img_object)
    keypoints_scene = kp_scene
    descriptors_scene = desc_scene

    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)


    ratio_thresh = 0.74
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    #-- Localize the object

    obj = np.float32([keypoints_obj[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    scene = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # print("Number of good matches: ", {len(good_matches)})
    if len(good_matches) < 10:
        return None, None

    H, mask =  cv.findHomography(obj, scene, cv.RANSAC,4)

    ## derive rotation angle from homography
    theta = -atan2(H[1,0], H[0,0])

    h,w = img_object.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,H)

    cntr, _ = getCenter(dst, img_scene)


    px2mm = 2.05 #30mm je široko 60 px
    px2mmY = 1.8545
    
    if CALIB_TYPE == CalibType.SIMPLE:
        p0 = [1204 ,1846] #coordinate system start point if simple calib used
        p1 = cntr

        p = [p1[0] - p0[0], p1[1] - p0[1]]
        dx= p[0]
        dy = p[1]
        
        uv = np.array([[1,0,0,dx/2.05],
                        [0,1,0,dy/2.05],
                        [0,0,1,0],
                        [0,0,0,1]], dtype=np.float32)
        
        Rt = np.array([[0.717, 0.697, 0, -470],
                       [0.697, -0.717, 0, -724],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.float32)
        
        rotAngle = -atan2(Rt[1,0], Rt[0,0])
        item_loc = np.matmul(Rt, uv)
        xyz = item_loc

    elif CALIB_TYPE == CalibType.HANDEYE:
        dx = cam_mtx5[0,2] - cntr[0]
        dy = cam_mtx5[1,2] - cntr[1]

        Rt = np.array([[0.7173754810141859, 0.6966179234465155, 0.009792240434646583, -581.95],
                       [0.6965090352371797, -0.7168042046648919, -0.032663374101656324, -325.90951685],
                       [-0.01573477272280828, 0.030252287645668206, -0.999418438903128, 672.4878897],
                       [0,0,0,1]], dtype=np.float32)

        Xc = ((dx*(Rt[2,3])/cam_mtx5[0,0]))
        Yc = ((dy*(Rt[2,3])/cam_mtx5[1,1]))

        loc = np.array([[1,0,0,-Xc],
                       [0,1,0,-Yc],
                       [0,0,1,0],
                       [0,0,0,1]], dtype=np.float32)

        rotAngle = -atan2(Rt[1,0], Rt[0,0])
        item_loc = np.matmul(Rt, loc)
        xyz = item_loc

    img2 = cv.polylines(img_scene,[np.int32(dst)],True,255,3, cv.LINE_AA)

    global result
    result = img2
    return xyz, theta + rotAngle



def runmain():
    RDK = robolink.Robolink()
    robot = RDK.Item("UR5")

    scene = get_scene()

    img_scene = scene
    img_scene = cv.cvtColor(img_scene, cv.COLOR_RGB2GRAY)

    newcameramatrix, _ = cv.getOptimalNewCameraMatrix(cam_mtx5, dist5, (3072, 2048), 1, (3072, 2048))


    #undistort the scene
    img_scene = cv.undistort(img_scene, cam_mtx5, dist5, None, newcameramatrix)

    img_scene = clahe.apply(img_scene)
    keypoints_scene, descriptors_scene = describe_kp(img_scene)
    
    home = RDK.Item("Home", 6)
    basket = RDK.Item("Basket",6)
    center = RDK.Item("Center",6)
    
    targets = RDK.ItemList(filter=robolink.ITEM_TYPE_TARGET)
    for target in targets:
        if target.equals(home) or target.equals(basket) or target.equals(center):
            continue
        RDK.Delete(target)
        

    for item in items:
        RDK.Item(item["item"]).setVisible(False)
    
    for item in items:
        xyz, angle = find_object(img_scene, item["pic"], keypoints_scene, descriptors_scene)
        if xyz is None:
            continue
        pose = TxyzRxyz_2_Pose([xyz[0,3],xyz[1,3],0,0,0,angle])
        target_pose = TxyzRxyz_2_Pose([xyz[0,3],xyz[1,3],200, 
                                  3.1415,0,0])
        RDK.Item(item["item"]).setPose(pose)
        RDK.Item(item["item"]).setVisible(True)

        target = RDK.AddTarget(item["item"])
        target.setPose(target_pose)


if __name__ == '__main__':
    runmain()