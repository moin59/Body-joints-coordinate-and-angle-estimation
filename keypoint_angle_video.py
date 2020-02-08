import numpy as np
# import pandas as pd
import time
import cv2
import ast
import math
# import caffe
# caffe.set_mode_gpu()


# load keypoints based on model
#  caffe model can be download from 
# https://drive.google.com/file/d/1-LjIOWlr9kqIrhjjyp8fAcVwePkj61Ck/view?usp=sharing
caffemodel = 'kp_iter_660000.caffemodel'
protoFile = 'kp_deploy_linevec.prototxt'
kpoint = 18

# read the network  layer and weight files
net = cv2.dnn.readNetFromCaffe(protoFile, caffemodel)

inputWidth = 368
inputHeight = 368
threshold = 0.1

# video source for internal (0), for external (1)
source = cv2.VideoCapture(0)
retVal, frame = source.read()

vid_writer = cv2.VideoWriter('SkeletonVideo.avi', cv2.VideoWriter_fourcc(
    *'XVID'), 10, (frame.shape[1], frame.shape[0]))

# recording original video
# original_vid.write(frame)

while cv2.waitKey(1) < 0:
    t = time.time()
    retVal, frame = source.read()
    frameCopy = np.copy(frame)

    if not retVal:
        cv2.waitKey()
        break
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    getBlob = cv2.dnn.blobFromImage(
        frame, 1.0/255, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(getBlob)
    blobOutput = net.forward()

    H = blobOutput.shape[2]
    W = blobOutput.shape[3]

    points = []

    for i in range(kpoint):
        # confidence map of corresponding body's part.
        confMap = blobOutput[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(confMap)

        # Scale the point to fit on the original image
        x_cor = (frameWidth * point[0]) / W
        y_cor = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x_cor), int(y_cor)), 8,
                       (0.0, 255.0, 255.0), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x_cor), int(y_cor)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0.0, 0.0, 255.0), 2, lineType=cv2.LINE_AA)
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x_cor), int(y_cor)))
        else:
            points.append(None)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', gray)
    cv2.imshow('Output-Keypoints', frameCopy)

    vid_writer.write(frame)

    # generated key-points value save into a txt file
    # for classification Standing=0 and Sitting =1, laying=2
    with open("coordinate.txt", 'a') as file:
        conv_ = str(points)
        value_ext = conv_.replace('(', '').replace(')', '').replace('None', '0, 0')
        abs = ast.literal_eval(value_ext)
        coordinate_value = np.array(abs, dtype=np.float32)

        #calculating angle
        """Counterclockwise angle in degrees by turning from a to c around b
            Returns a float between 0.0 and 360.0"""
        def angle3pt(p, q, r):
            if p[0]!=0 and q[0]!=0 and r[0]!=0:
                ang = math.degrees(math.atan2(r[1]-q[1], r[0]-q[0]) - math.atan2(p[1]-q[1], p[0]-q[0]))
                return ang + 360 if ang < 0 else ang

        #Body limbs coordinate values
        #################################
        R_sho_x = coordinate_value[4]
        R_sho_y = coordinate_value[5]
        L_sho_x = coordinate_value[10]
        L_sho_y = coordinate_value[11]
        #################################
        R_heep_x = coordinate_value[16]
        R_heep_y = coordinate_value[17]
        L_heep_x = coordinate_value[22]
        L_heep_y = coordinate_value[23]
        #################################
        R_knee_x = coordinate_value[18]
        R_knee_y = coordinate_value[19]
        L_knee_x = coordinate_value[24]
        L_knee_y = coordinate_value[25]
        #################################
        R_ankle_x = coordinate_value[20]
        R_ankle_y = coordinate_value[21]
        L_ankle_x = coordinate_value[26]
        L_ankle_y = coordinate_value[27]

        # calculate angle at knee(left and right)
        leg_angleR=angle3pt((R_heep_x, R_heep_y), (R_knee_x,R_knee_y), (R_ankle_x, R_ankle_y))
        leg_angleL=angle3pt((L_heep_x, L_heep_y), (L_knee_x,L_knee_y), (L_ankle_x, L_ankle_y))

        # calculate angle at hip(left and right)
        hip_angleR=angle3pt((R_sho_x, R_sho_y), (R_heep_x,R_heep_y), (R_knee_x, R_knee_y))
        hip_angleL=angle3pt((L_sho_x, L_sho_y), (L_heep_x,L_heep_y), (L_knee_x, L_knee_y))

        conv_str = str(points)
        value_extract = conv_str.replace(
            '(', '').replace(')', '').replace('None', '0, 0')
        abstract = ast.literal_eval(value_extract)
        new_list = []
        for item in abstract:
            new_list.append(float(item))
        file.write(str(new_list)+','+str(hip_angleR)+','+str(hip_angleL)+','+str(leg_angleR)+',' + str(leg_angleL) + '\n'+"Sitting,")
        print(new_list, hip_angleR, hip_angleL ,leg_angleR, leg_angleL)

vid_writer.release()
# cv2.destroyAllWindows()
