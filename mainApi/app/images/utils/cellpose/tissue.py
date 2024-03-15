import os
from cellpose import models
import numpy as np
import time, os, sys
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
from urllib.parse import urlparse
from cellpose import models, core
import cv2
import math
import random
import pandas as pd

# DISPLAY RESULTS
from cellpose import plot


MIN_THRESH = 10
BLACK_VALUE = 0
WHITE_VALUE = 255


TISSUE_COLORS = {
    "S" : [128,128,128],
    "R" : [0,0,255],
    "G": [0,255,0],
    "B" : [255,0,0],
    "B+G" : [255,255,0],
    "B+R" : [255,0,255],
    "S+B" : [255,128,128],
    "S+G" : [128,255,128],
    "G+R" : [0,255,255],
    "B+G+R" : [255,255,255],
    "S+G+R" : [128,255,255],
    "S+B+R" : [255,128,255],
    "S+B+G" : [255,255,128],
    "S+B+G+R" : [255,255,255]
}

def getBinaryImage(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary  = cv2.threshold(gray,MIN_THRESH,WHITE_VALUE,BLACK_VALUE)
    return binary


def process_TissueNT_Segmentation(input_path, output_directory, channel):

    print("This is process_TissueNT_Segmentation function in tissue.py file...")

    img = skimage.io.imread(input_path)
    
    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d'%use_GPU)

    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=use_GPU, model_type='cyto')

    channels = []
    if channel == "R":
        channels = [[1,1]]
    if channel == "S":
        channels = [[0,0]]
    if channel == "B":
        channels = [[3,3]]
    if channel == "G":
        channels = [[2,2]]
    masks, flows, styles, diams = model.eval([img], diameter=None, flow_threshold=None, channels=channels)
    
    idx = 0
    maski = masks[idx]
    flowi = flows[idx][0]


    mask_output_filename = "mask_output.jpg"
    flow_output_filename = "flow_output.jpg"


    mask_output_path = os.path.join(output_directory, mask_output_filename)
    flow_output_path = os.path.join(output_directory, flow_output_filename)


    print(mask_output_path)
    print(flow_output_path)

    
    cv2.imwrite(mask_output_path, maski)
    cv2.imwrite(flow_output_path,  flowi)



def process_TissueNT_Test_Segmentation(input_path, output_directory, channel):

    print("This is process_TissueNT_Test_Segmentation function in tissue.py file...")

    img = skimage.io.imread(input_path)
    
    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d'%use_GPU)

    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=use_GPU, model_type='cyto')

    channels = []
    if channel == "R":
        channels = [[1,1]]
    if channel == "S":
        channels = [[0,0]]
    if channel == "B":
        channels = [[3,3]]
    if channel == "G":
        channels = [[2,2]]
    masks, flows, styles, diams = model.eval([img], diameter=None, flow_threshold=None, channels=channels)
    
    idx = 0
    maski = masks[idx]
    flowi = flows[idx][0]


    mask_output_filename = "test_mask_output.jpg"
    flow_output_filename = "test_flow_output.jpg"


    mask_output_path = os.path.join(output_directory, mask_output_filename)
    flow_output_path = os.path.join(output_directory, flow_output_filename)


    print(mask_output_path)
    print(flow_output_path)

    
    cv2.imwrite(mask_output_path, maski)
    cv2.imwrite(flow_output_path,  flowi)




def getMergedImageByWholeChannels(R,G,B,S):

    
    print("This is getMergedImageByWholeChannels function in tissue.py file...")


    image = cv2.merge([S,S,S])

    (height, width, channel) = image.shape

    s_merged_image = np.zeros((height, width, channel), dtype="uint8")
    s_merged_image[np.where((image==[255,255,255]).all(axis=2))] = [128,128,128]


    for x in range(width):
        for y in range(height):
            if R[y,x] > 0 or G[y,x] > 0  or B[y,x] > 0:
                s_merged_image[y,x][0] = B[y,x]
                s_merged_image[y,x][1] = G[y,x]
                s_merged_image[y,x][2] = R[y,x]

    return s_merged_image


def getResultImageFromColorOptions(resultImage, color_options, fillColors):

    print("This is getResultImageFromColorOptions function in tissue.py file...")

    res_image = resultImage
    (height, width, channel) = res_image.shape
    
    finalImgae = np.zeros((height,width,3), np.uint8)
    for x in range(width):
        for y in range(height):
            pix_val = res_image[y,x]

            if pix_val[0] < MIN_THRESH and pix_val[1] < MIN_THRESH and pix_val[2] < MIN_THRESH :
                    continue


            for opt in color_options:
                value = TISSUE_COLORS[opt]
                fillColorValue = fillColors[opt]
              
                if abs(pix_val[0] - value[0]) < MIN_THRESH and abs(pix_val[1] - value[1]) < MIN_THRESH and abs(pix_val[2] - value[2]) < MIN_THRESH:
                    finalImgae[y,x] = fillColorValue



    gray = cv2.cvtColor(finalImgae, cv2.COLOR_BGR2GRAY)
    ret, res_binary  = cv2.threshold(gray,MIN_THRESH,WHITE_VALUE,BLACK_VALUE)


    return [finalImgae, res_binary]




def getDotPlotImage(segmented_image, radius):

        (height, width, channel) = segmented_image.shape
        image = np.zeros((height, width, channel), dtype="uint8") 

        step = radius * 3 + 1
        thickness = radius

        for x in range(1,width,step):
            for y in range(1, height,step):
                colors = segmented_image[y,x]
                color = (int(colors[0]), int(colors[1]), int(colors[2]))
                cv2.circle(image, (x,y), radius , color, thickness)


        return image





def getShortLength(w,h):
    return min(w,h)

def getLongLength(w,h):
    return max(w,h)


# measure of flattening at the poles of a planet or other celestial body. 
#The oblateness is measured as the ratio between the polar and equatorial diameter.
def getValueOfOblateness(w, h):
    long_d = max(w,h)
    short_d = min(w,h)
    return int(short_d * 10000 / long_d)


#inverse value of oblateness
def getValueOfInvFlatRate(w, h):
    long_d = max(w,h)
    short_d = min(w,h)
    return int (long_d * 10000 / short_d)


#ratio = 4 * pi * Area / ( Perimeter^2 )
def getValueOfRoundness(area, perimeter):
    return int(10000 * 4 * math.pi * area / perimeter/ perimeter)



def getValueOfEllpsity(area, d):
    return int(10000 * d * d / 4 / math.pi /  area)


def getValueOfFillRatio(area, w, d):
    return int(10 * (w * d - area))

def getValueOfOutPixels(area, w, d):
    return int( w * d - area)


def getMergedImage(originImage, mergeImage,percent):


    print(originImage.shape)
    print(mergeImage.shape)

    h, w, _ = originImage.shape
    startX =  int(int(w / 2) -  int(w * percent / 2 / 100))
    endX =  int(int(w / 2) + int( w * percent / 2 / 100))
    startY = int(int(h / 2) - int(w * percent / 2 / 100))
    endY = int(int(h / 2) + int(w * percent / 2 / 100))

    print(startX)
    print(startY)
    print(endX)
    print(endY)
    
    originImage[startY:endY, startX:endX] = mergeImage

    return originImage

def processMeasureForTissueNT(image, output_path):

    print("This is the test program for measurement item functions")

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    arrNo = []
    arrPixels = []
    arrCentX = []
    arrCentY = []
    arrMinX = []
    arrMaxX = []
    arrMinY = []
    arrMaxY = []
    arrWidth = []
    arrHeight = []

    arrInvFlatRate = []
    arrShortLen = []
    arrLongLen = []
    arrOblateness = []

    arrRoundness = []
    arrEllpsity  = []
    arrRectFillRatio = []
    arrOuterPixels = []
    arrAvgThickness = []
    arrSlenderness = []
    arrWrapCount = []
    arrOuterArea = []

    for i in range(2, numLabels):
        # extract the connected component statistics for the current
        # label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        arrNo.append(i - 1)
        arrPixels.append(area)
        arrCentX.append(int(centroids[i][0]))
        arrCentY.append(int(centroids[i][1]))
        arrMinX.append(x)
        arrMaxX.append(x+w)
        arrMinY.append(y)
        arrMaxY.append(y+h)
        arrWidth.append(w)
        arrHeight.append(h)

        arrInvFlatRate.append(getValueOfInvFlatRate(w,h))
        arrShortLen.append(getShortLength(w,h))
        arrLongLen.append(getLongLength(w,h))
        arrOblateness.append(getValueOfOblateness(w,h))
        arrRoundness.append(getValueOfRoundness(area, (w+h) * 2 + random.randint(1,2)))

        arrEllpsity.append(getValueOfRoundness(area, (w+h) * 2 + random.randint(1,2)))
        arrRectFillRatio.append(getValueOfFillRatio(area, w, h))
        arrOuterPixels.append(getValueOfOutPixels(area, w , h))
        arrAvgThickness.append(int(math.sqrt(w * h)))
        arrSlenderness.append(int(w * h))
        arrWrapCount.append(int(w))
        arrOuterArea.append(getValueOfOutPixels(area,w,h))

    df = pd.DataFrame() 

    print(output_path)

    df["no"] = arrNo
    df["0:pixels"] = arrPixels
    df["1:point-x"] = arrCentX
    df["2:point-y"] = arrCentY
    df["3:0x"] = arrMinX
    df["4:x1"] = arrMaxX
    df["5:y0"] = arrMinY
    df["6:y1"] = arrMaxY
    df["7:width"] = arrWidth
    df["8:height"] = arrHeight
    df["9:inv-oblate"] = arrInvFlatRate
    df["10:S-length"] = arrShortLen
    df["11:L-length1"] = arrLongLen
    df["12:oblate"] = arrOblateness
    df["13:round"] = arrRoundness
    df["14:ellipsity"] = arrEllpsity
    df["15:filling-ratio"] = arrRectFillRatio
    df["16:outer-pixels"] = arrOuterPixels  
    df["17:thickness"] = arrAvgThickness
    df["18:slenderness"] = arrSlenderness
    df["19:wrap-count"] = arrWrapCount
    df["20:outer-area"] = arrOuterArea


    df.to_csv(output_path)





