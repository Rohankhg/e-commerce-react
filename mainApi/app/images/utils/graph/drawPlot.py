from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
from random import random as getValue
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
import FlowCal
#from matplotlib import colormaps
#print(list(colormaps))

INPUT_CSV_PATH = "mainApi/ml_lib/sample.ome_300.csv"
INPUT_IMAGE_PATH="mainApi/ml_lib/High Content Flow Plate Data/Well_A01.fcs"

def saveHeatmap(input_path, output_path,x_index, y_index, cmap = "jet"):
    print(input_path)
    data=FlowCal.io.FCSData(INPUT_IMAGE_PATH)
    print(data.channels)
    data=FlowCal.transform.to_rfi(data)
    FlowCal.plot.density2d(data, channels=['FSC-A','SSC-A'], mode= 'scatter',cmap=cmap)
    plt.savefig(output_path)
    # data = pd.read_csv(INPUT_CSV_PATH)

    # headers = []
    # for col in data.columns:
    #     headers.append(col)

    # data = data.dropna()
    
    # x = data[
    #     headers[x_index]
    # ].values

    # y = data[
    #     headers[y_index]
    # ].values

    # s_max = max(max(x), max(y))

    # x = x / max(x) * s_max * 3
    # y = y / max(y) * s_max * 3

    # s_len = 3000


    # x = x.tolist()
    # y = y.tolist()

    # for i in range(s_len):  
    #     x.append(getValue() * s_max)
    #     y.append(getValue() * s_max)
    
    # xy = np.vstack([x,y])
    # z = gaussian_kde(xy)(xy)

    # fig, ax = plt.subplots()
    # ax.scatter(x, y, c=z, s=6,cmap =cmap)
    # plt.savefig(output_path)

    #plt.show()
def addROIInHeatMapImage(input_path, output_path, roi_area):
    image = cv2.imread(input_path)

    h,w,c = image.shape

    start_point = (int(w * roi_area[0] / 100), int(h * roi_area[2] / 100)) 
    end_point = (int(w * roi_area[1] / 100), int(h * roi_area[3] / 100)) 

    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 2 px 
    thickness = 2
    image_with_rectangle = cv2.rectangle(image.copy(), start_point, end_point, color, thickness)

    # Crop the specified ROI from the original image
    cropped_image = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    bounding_box_thickness = 2
    bounding_box_color = (0, 0, 0)  # Black color
    cropped_image_with_box = cv2.rectangle(
        cropped_image.copy(), (0, 0), (cropped_image.shape[1], cropped_image.shape[0]), bounding_box_color, bounding_box_thickness
    )

    # Display the original image and the cropped ROI with bounding box using matplotlib
    # plt.figure(figure=(12,6))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(cv2.cvtColor(cropped_image_with_box, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title('Cropped ROI with Bounding Box')

    # Save the cropped image with the bounding box to the specified output path
    plt.savefig(output_path)
    # axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axes[0].set_title('Original Image')

    # plt.imshow(cv2.cvtColor(cropped_image_with_box, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # # axes[1].set_title('Cropped ROI with Bounding Box')
    # plt.savefig(output_path)

    # plt.show()
    print(cropped_image)

    # # Display the original image and the cropped ROI
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Image with ROI Rectangle", image_with_rectangle)
    # cv2.imshow("Cropped ROI", cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the image with the rectangle to the specified output path
    # cv2.imwrite(output_path, cropped_image_with_box)

    
    # Using cv2.rectangle() method 
    # Draw a rectangle with blue line borders of thickness of 2 px 
    # image = cv2.rectangle(image, start_point, end_point, color, thickness) 

    # cv2.imwrite(output_path, image)


# saveHeatmap(input_path, "1.png",3,12)
# saveHeatmap(input_path, "2.png",2,4)
# saveHeatmap(input_path, "3.png",2,13,'cividis' )
# saveHeatmap(input_path, "4.png",4,15,'magma' )
# saveHeatmap(input_path, "5.png",8,17,'gist_gray_r' )