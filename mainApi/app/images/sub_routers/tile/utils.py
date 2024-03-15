from PIL import Image, ImageFilter
import os
import numpy as np
import cv2




def edgeBlurImage(img):

    # blur radius and diameter
    radius, diameter = 1,3

    r,g,b = img.split()

    aver_v = int(np.mean(img))


    aver_r = int(np.mean(r))
    aver_g = int(np.mean(g))
    aver_b = int(np.mean(b))


    # Paste image on white background
    background_size = (img.size[0] + diameter, img.size[1] + diameter)
    background = Image.new('RGB', background_size, (aver_r,aver_g , aver_b))
    background.paste(img, (radius, radius))

   

    # create new images with white and black
    mask_size = (img.size[0] + diameter, img.size[1] + diameter)
    mask = Image.new('L', mask_size, aver_v )

    black_size = (img.size[0] - diameter, img.size[1] - diameter)
    black = Image.new('L', black_size, aver_v)

    # create blur mask
    mask.paste(black, (diameter, diameter))

    # Blur image and paste blurred edge according to mask
    #blur = background.filter(ImageFilter.GaussianBlur(radius / 2))
    blur = background.filter(ImageFilter.BLUR)
    #background.paste(blur, mask=mask)
    return background

#align :   "snake" , "raster"
#direction :  "horizontal" , "vertical"
#sortOrder : true,false
#overlapX, overlapY : number for overlapping

FILE_NAME_TEMPLATE = "tile_image_series"


def mergeImageWithOverlap(imageDir,rows,cols,align,direction,sortOrder,overlapX,overlapY, output_path,ext):
    
    print("The directory name we have to merge is ...")
    print(imageDir)
    arrFileName = []
    for root, dir, files in os.walk(imageDir):
        for filename in files:
            if FILE_NAME_TEMPLATE in filename and ext in filename:
                arrFileName.append(os.path.join(imageDir, filename))

    arrFileName = sorted(arrFileName)
    if sortOrder == False:
        arrFileName = arrFileName[::-1]
    
    arrProcessFileNames = []
    for i in range(rows):
        temp = arrFileName[cols*i: (cols) * (i+1)]
        if i % 2 == 1  and align == "snake":
            temp = temp[::-1]
        arrProcessFileNames.append(temp)

    print(arrProcessFileNames)

    horizontalImages = []

    for rowImages in arrProcessFileNames:
        tempImage = Image.open(rowImages[0])
        #tempImage = edgeBlurImage(tempImage)
        tempImage = tempImage.resize((int (tempImage.size[0] / 1), int(tempImage.size[1] / 1) ))
        for imgPath in rowImages:
            if rowImages.index(imgPath) == 0:
                continue
            
            img = Image.open(imgPath)
            # img = edgeBlurImage(img)
            img = img.resize((int(img.size[0] / 1),int( img.size[1] / 1) ))
            tempImage = mergeHorizontal(tempImage, img, overlapX)

        horizontalImages.append(tempImage)

    final = horizontalImages[0]
    for img in horizontalImages:
        if(horizontalImages.index(img) == 0): continue
        final = mergeVertical(final, img, overlapY)
    
    final.save(output_path)
    return 

def mergeHorizontal(im1, im2, overlap):

    w = im1.size[0] + im2.size[0] - overlap
    h = max(im1.size[1], im2.size[1])
    im = Image.new("RGB", (w, h))

    im.paste(im1)
    im.paste(im2, (im1.size[0] - overlap, 0))

    return im

def mergeVertical(im1, im2, overlap):
    w = max(im1.size[0], im2.size[0])
    h = im1.size[1] + im2.size[1] - overlap

    im = Image.new("RGB", (w, h))

    im.paste(im1)
    im.paste(im2, (0, im1.size[1] - overlap))

    return im




def cropImage(orgImg, percent):
    h, w, _ = orgImg.shape
    startX =  int(int(w / 2) -  int(w * percent / 2 / 100))
    endX =  int(int(w / 2) + int( w * percent / 2 / 100))
    startY = int(int(h / 2) - int(w * percent / 2 / 100))
    endY = int(int(h / 2) + int(w * percent / 2 / 100))

    print(startX)
    print(startY)
    print(endX)
    print(endY)

    crop_img = orgImg[startY:endY, startX:endX]
    return crop_img
