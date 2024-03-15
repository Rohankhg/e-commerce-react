import os

from mainApi.app.images.utils.asyncio import shell
from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Response,
    Depends,
    Form,
    status
)
from fastapi.responses import JSONResponse, FileResponse
from mainApi.app.images.sub_routers.tile.routers import router as tile_router
from mainApi.config import STATIC_PATH
from mainApi.app.auth.auth import get_current_user
from mainApi.app.auth.models.user import UserModelDB, PyObjectId
import subprocess
import tempfile
import time
from datetime import date
from typing import List
import json
import h5py as h5
from mainApi.app.images.h5.measure import update_h5py_file
import tifffile
import numpy as np
import bioformats
from PIL import Image
import javabridge as jv
import tifftools
import mainApi.app.images.utils.deconvolution as Deconv
import shutil
import zipfile
import pandas as pd
from mainApi.app.images.utils.measure import processBasicMeasureData
from mainApi.app.images.utils.mridge.mridge import processMRIDGEMethod
from mainApi.app.images.utils.cellpose.tissue import process_TissueNT_Segmentation,getBinaryImage,getMergedImageByWholeChannels,getResultImageFromColorOptions,TISSUE_COLORS,getDotPlotImage,processMeasureForTissueNT,process_TissueNT_Test_Segmentation,getMergedImage
from mainApi.app.images.utils.graph.drawPlot import saveHeatmap,addROIInHeatMapImage
#from mainApi.app.images.utils.mouseTracking.mouseTrack import processTopViewMouseTracking
import cv2
import math



TISSUE_MERGE_IMAGE_PATH = "tissueMerge.jpg"
TISSUE_MASK_OUTPUT = "mask_output.jpg"
TISSUE_FLOW_OUTPUT = "flow_output.jpg"
TISSUE_OUTPUT_SEGMENT_IMAGE_PATH = "tissueResult.jpg"
TISSUE_OUTPUT_BINARY_PATH = 'tissueBinaryResult.jpg'
TISSUE_OUTPUT_DOTPLOT_IMAGE_PATH = "tissueDotPlot.jpg"


TISSUE_TEST_MERGE_IMAGE_PATH = "test_TissueMerge.jpg"
TISSUE_TEST_MASK_OUTPUT = "test_mask_output.jpg"
TISSUE_TEST_FLOW_OUTPUT = "test_flow_output.jpg"
TISSUE_TEST_OUTPUT_SEGMENT_IMAGE_PATH = "test_tissueResult.jpg"
TISSUE_TEST_OUTPUT_BINARY_PATH = 'test_tissueBinaryResult.jpg'
TISSUE_TEST_OUTPUT_DOTPLOT_IMAGE_PATH = "test_tissueDotPlot.jpg"

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


router = APIRouter(prefix="/image", tags=[])

router.include_router(tile_router)

@router.get("/hdf5_download")
async def download_exp_image(
    request: Request,
    path: str
):
    full_path = f"mainApi/app/static/measure_out/{path}"
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(full_path, "rb") as file:
        content = file.read()
        return Response(content)

@router.get("/zip_download")
async def download_zip(
    request: Request,
    path: str
):
    full_path = f"mainApi/app/static/{path}"
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(full_path, "rb") as file:
        content = file.read()
        return Response(content)


@router.get("/download")
async def download_exp_image(
    request: Request,
    path: str
):
    full_path = f"{STATIC_PATH}/{path}"
    file_size = os.path.getsize(full_path)
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    range = request.headers["Range"]
    if range is None:
        return FileResponse(full_path, filename=path)
    ranges = range.replace("bytes=", "").split("-")
    range_start = int(ranges[0]) if ranges[0] else None
    range_end = int(ranges[1]) if ranges[1] else file_size - 1
    if range_start is None:
        return Response(content="Range header required", status_code=416)
    if range_start >= file_size:
        return Response(content="Range out of bounds", status_code=416)
    if range_end >= file_size:
        range_end = file_size - 1
    content_length = range_end - range_start + 1
    headers = {
        "Content-Range": f"bytes {range_start}-{range_end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
    }
    with open(full_path, "rb") as file:
        file.seek(range_start)
        content = file.read(content_length)
        return Response(content, headers=headers, status_code=206)

@router.get("/download_csv")
async def download_exp_image(
    request: Request,
    path: str
):
    full_path = f"{STATIC_PATH}/{path}"
    print("download-csv-path:", full_path)
    file_size = os.path.getsize(full_path)
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    with open(full_path, "rb") as file:
        content = file.read()
        return Response(content, headers=headers, status_code=206)

@router.post(
    "/before_process",
    response_description="Process image",
)
async def processImage(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    print("get-request-data:", data)
    imagePath = '/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + data.get("original_image_url")
    folderName = date.today().strftime("%y%m%d%H%M%s")
    sharedImagePath = os.path.join("/app/shared_static", folderName)

    if not os.path.exists(sharedImagePath):
        os.makedirs(sharedImagePath)

    fileName = imagePath.split("/")[len(imagePath.split("/")) - 1]
    newImagePath = os.path.join(sharedImagePath, fileName)

    cmd_str = "cp '{inputPath}' '{outputPath}'".format(
        inputPath=imagePath, outputPath=newImagePath
    )
    subprocess.call(cmd_str, shell=True)

    return JSONResponse({"success": "success", "image_path": newImagePath})

@router.post(
    "/ml_ict_process",
    response_description="ML IPS Process",
)
async def mlICTProcess(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    imagePath = '/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + data.get("original_image_url")

    sensitivity = data.get("sensitivity")
    type = data.get("type")

    fileName = imagePath.split("/")[len(imagePath.split("/")) - 1]
    originPath = imagePath.replace(fileName, '')
    tempPath = tempfile.mkdtemp()
    OUT_PUT_FOLDER = tempPath.split("/")[len(tempPath.split("/")) - 1]
    OUT_PUT_PATH = 'mainApi/app/static/ml_out/' + OUT_PUT_FOLDER

    fullPath = OUT_PUT_PATH + '/' + fileName
    csvPath = os.path.splitext(fullPath)[0] + '_300.csv'
    originCSVPath = os.path.splitext(imagePath)[0] + '_300.csv'
    zipPath = os.path.splitext(imagePath)[0] + '_measure_result.zip'

    if not os.path.exists(OUT_PUT_PATH):
        os.makedirs(OUT_PUT_PATH)

    cmd_str = "/app/mainApi/ml_lib/segB {inputPath} {outputPath}"
    if type == 'a':
        cmd_str += " /app/mainApi/ml_lib/typeB/src_paramA.txt"
    if type == 'b':
        cmd_str += " /app/mainApi/ml_lib/typeB/src_paramB.txt"
    if type == 'c':
        cmd_str += " /app/mainApi/ml_lib/typeB/src_paramC.txt"
    if type == 'd':
        cmd_str += " /app/mainApi/ml_lib/typeB/src_paramD.txt"

    cmd_str += " " + sensitivity
    cmd_str = cmd_str.format(inputPath=imagePath, outputPath=OUT_PUT_PATH + "/" + fileName)
    print('----->', cmd_str)
    subprocess.call(cmd_str, shell=True)


    data = pd.read_csv(csvPath)
    data = processBasicMeasureData(csvPath, data)
    data.to_csv(csvPath)
    
    shutil.copy(csvPath, originCSVPath)

    zipf = zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED)
    zipdir(OUT_PUT_PATH + "/", zipf)
    zipf.close()

    zipPath = zipPath.replace('/app/mainApi/app/static/', '')

    return JSONResponse({"success": "success", "image_path": OUT_PUT_PATH + "/" + fileName, "zip_path": zipPath})



@router.post(
    "/ml_mfiber_process",
    response_description="ML MFIBER Process",
)
async def mlMFIBERProcess(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    imagePath = '/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + data.get("original_image_url")

    sensitivity = data.get("sensitivity")
    method = data.get("method")

    fileName = imagePath.split("/")[len(imagePath.split("/")) - 1]
    originPath = imagePath.replace(fileName, '')
    tempPath = tempfile.mkdtemp()
    OUT_PUT_FOLDER = tempPath.split("/")[len(tempPath.split("/")) - 1]
    OUT_PUT_PATH = 'mainApi/app/static/ml_out/' + OUT_PUT_FOLDER

    fullPath = OUT_PUT_PATH + '/' + fileName
    csvPath = os.path.splitext(fullPath)[0] + '_300.csv'
    originCSVPath = os.path.splitext(imagePath)[0] + '_300.csv'
    zipPath = os.path.splitext(imagePath)[0] + '_measure_result.zip'

    if not os.path.exists(OUT_PUT_PATH):
        os.makedirs(OUT_PUT_PATH)

    cmd_str = "/app/mainApi/ml_lib/segAB {inputPath} {outputPath}"
    if method == "mfiber1":
       cmd_str += " /app/mainApi/ml_lib/typeAB/mfiber/src_paramA1.txt"
    if method == "mfiber2":
        cmd_str += " /app/mainApi/ml_lib/typeAB/mfiber/src_paramA2.txt"
    

    cmd_str += " " + sensitivity
    cmd_str += " /app/mainApi/ml_lib/color_table.txt"
    cmd_str = cmd_str.format(inputPath=imagePath, outputPath=OUT_PUT_PATH + "/" + fileName)
    print('----->', cmd_str)
    subprocess.call(cmd_str, shell=True)


    data = pd.read_csv(csvPath)
    data = processBasicMeasureData(csvPath, data)
    data.to_csv(csvPath)
    
    shutil.copy(csvPath, originCSVPath)

    zipf = zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED)
    zipdir(OUT_PUT_PATH + "/", zipf)
    zipf.close()

    zipPath = zipPath.replace('/app/mainApi/app/static/', '')

    return JSONResponse({"success": "success", "image_path": OUT_PUT_PATH + "/" + fileName, "zip_path": zipPath})



@router.post(
    "/ml_mridge_process",
    response_description="DL MRIDGE Process",
)
async def mlMRIDGEProcess(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    imagePath = '/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + data.get("original_image_url")

    sensitivity = data.get("sensitivity")


    fileName = imagePath.split("/")[len(imagePath.split("/")) - 1]
    originPath = imagePath.replace(fileName, '')
    tempPath = tempfile.mkdtemp()
    OUT_PUT_FOLDER = tempPath.split("/")[len(tempPath.split("/")) - 1]
    OUT_PUT_PATH = 'mainApi/app/static/ml_out/' + OUT_PUT_FOLDER

    fullPath = OUT_PUT_PATH + '/' + fileName
    csvPath = os.path.splitext(fullPath)[0] + '_300.csv'
    originCSVPath = os.path.splitext(imagePath)[0] + '_300.csv'
    zipPath = os.path.splitext(imagePath)[0] + '_measure_result.zip'

    if not os.path.exists(OUT_PUT_PATH):
        os.makedirs(OUT_PUT_PATH)


    sample_csv_path = "./mainApi/ml_lib/sample.ome_300.csv"
    
    processMRIDGEMethod(fullPath, sensitivity)


  
    shutil.copy(sample_csv_path, originCSVPath)

    print(fullPath)

  

    zipPath = zipPath.replace('/app/mainApi/app/static/', '')

    return JSONResponse({"success": "success", "image_path": OUT_PUT_PATH + "/" + fileName, "zip_path": zipPath})






@router.post(
    "/ml_ict_process_test",
    response_description="ML IPS Process Test",
)
async def mlICTProcessTest(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    imagePath = '/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + data.get("original_image_url")
    sensitivity = data.get("sensitivity")
    param = data.get("param")
    params = param.split(",")
    type = data.get("type")

    fileName = imagePath.split("/")[len(imagePath.split("/")) - 1]
    tempPath = tempfile.mkdtemp()
    OUT_PUT_FOLDER = tempPath.split("/")[len(tempPath.split("/")) - 1]
    OUT_PUT_PATH = 'mainApi/app/static/ml_out/' + OUT_PUT_FOLDER
    paramPath = '/app/mainApi/app/static/ml_out/' + OUT_PUT_FOLDER + '/param.txt'

    if not os.path.exists(OUT_PUT_PATH):
        os.makedirs(OUT_PUT_PATH)

    path = ''

    if type == 'a':
        path = "/app/mainApi/ml_lib/typeB/src_paramA.txt"
    if type == 'b':
        path = "/app/mainApi/ml_lib/typeB/src_paramB.txt"
    if type == 'c':
        path = "/app/mainApi/ml_lib/typeB/src_paramC.txt"
    if type == 'd':
        path = "/app/mainApi/ml_lib/typeB/src_paramD.txt"

    with open(path, 'rb') as fsrc:
        with open(paramPath, 'wb') as fdst:
            fdst.write(fsrc.read())

    with open(paramPath, "a") as f:
        f.write("\n")
        for num in params:
            f.write(num + "\n")


    cmd_str = "/app/mainApi/ml_lib/segB {inputPath} {outputPath} " + paramPath

    cmd_str += " " + sensitivity
    cmd_str = cmd_str.format(inputPath=imagePath, outputPath=OUT_PUT_PATH + "/" + fileName)
    print('----->', cmd_str)
    subprocess.call(cmd_str, shell=True)
    return JSONResponse({"success": "success", "image_path": OUT_PUT_PATH + "/" + fileName})



@router.post(
    "/tissueTestProcess",
    response_description="ML TissueNT Process",
)
async def mlTissueTestProcess(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    sensitivity = data.get("sensitivity")
    colors = data.get('colors')
    colorOption = data.get('colorOption')
    tilemergedFlag = data.get('tilingMergedImageFlag')
    colors = colors.split(",")
    imgPath = data.get("original_image_url")
    paths = imgPath.split(",")
    realPaths = []

    for path in paths:
        realPaths.append('/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + path)

    imagePath = ' '.join(realPaths)


    path = imagePath

    print("Image Path")
    print(path)

    print(sensitivity)


    tempLists = path.split("/")
    fileName = tempLists[-1]
    t = tempLists[-2] + "/" + tempLists[-1]
    rel_dir = path.split(t)[0]

    print(rel_dir)
    origin_image_path = os.path.join(rel_dir, "Overlay", "ashlar_output.jpg")

    tissueResultImagePath = os.path.join(rel_dir, TISSUE_TEST_MERGE_IMAGE_PATH)
    resImage = []
    
    if not os.path.exists(tissueResultImagePath) :


        for colorOption in ["S","R","G","B"]:


            resultInputPath = rel_dir + colorOption + "/" + fileName
            
            print("resultInputPath =", resultInputPath)

            OUT_PUT_PATH =  rel_dir +  colorOption + "/"
            print(OUT_PUT_PATH)


            mask_output_path = os.path.join(OUT_PUT_PATH, TISSUE_TEST_MASK_OUTPUT)

            if (os.path.exists(mask_output_path) == False):
                process_TissueNT_Test_Segmentation(resultInputPath, OUT_PUT_PATH, colorOption)
            
        
        result_S_path = rel_dir +  "S/" + TISSUE_TEST_MASK_OUTPUT
        result_B_path = rel_dir +  "B/" + TISSUE_TEST_MASK_OUTPUT
        result_G_path = rel_dir +  "G/" + TISSUE_TEST_MASK_OUTPUT
        result_R_path = rel_dir +  "R/" + TISSUE_TEST_MASK_OUTPUT


        imageS = cv2.imread(result_S_path)
        imageB = cv2.imread(result_B_path)
        imageG = cv2.imread(result_G_path)
        imageR = cv2.imread(result_R_path)


        imageS = getBinaryImage(imageS)
        imageR = getBinaryImage(imageR)
        imageG = getBinaryImage(imageG)
        imageB = getBinaryImage(imageB)

        resImage = getMergedImageByWholeChannels(imageR,imageG, imageB, imageS)

        cv2.imwrite(tissueResultImagePath, resImage)
    

    else:
        resImage = cv2.imread(tissueResultImagePath)
    


    print(colors)
    [segment, binary] = getResultImageFromColorOptions(resImage, colors, TISSUE_COLORS)


    tissueSegmentResultPath = os.path.join(rel_dir, TISSUE_TEST_OUTPUT_SEGMENT_IMAGE_PATH)
    tissueBinaryResultPath = os.path.join(rel_dir, TISSUE_TEST_OUTPUT_BINARY_PATH)
    tissueDotPlotResultPath = os.path.join(rel_dir, TISSUE_TEST_OUTPUT_DOTPLOT_IMAGE_PATH )

    sensitivity = int(sensitivity)

    radius = 6
    if sensitivity < 20:
        radius = 6
    elif sensitivity < 40:
        radius = 5
    elif sensitivity < 60:
        radius = 4
    elif sensitivity < 80:
        radius = 3
    else:
        radius = 2
    
    dotplotImage = getDotPlotImage(segment,radius)

    
    cv2.imwrite(tissueDotPlotResultPath, dotplotImage)
    cv2.imwrite(tissueBinaryResultPath, binary)

    percent = 30

    print("Origin Image Path is --->")
    print(origin_image_path)
    orgImg = cv2.imread(origin_image_path)
    segment = getMergedImage(orgImg, segment,percent)

    cv2.imwrite(tissueSegmentResultPath, segment)

    return JSONResponse({"success": "success","image_path" : path , "mask_output_path": tissueBinaryResultPath, "flow_output_path" : tissueSegmentResultPath, "dotplot_output_path"  :  tissueDotPlotResultPath })






@router.post(
    "/tissueProcess",
    response_description="ML TissueNT Process",
)
async def mlTissueProcess(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    sensitivity = data.get("sensitivity")
    colors = data.get('colors')
    colorOption = data.get('colorOption')
    tilemergedFlag = data.get('tilingMergedImageFlag')
    colors = colors.split(",")
    imgPath = data.get("original_image_url")
    paths = imgPath.split(",")
    realPaths = []

    for path in paths:
        realPaths.append('/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + path)

    imagePath = ' '.join(realPaths)


    path = imagePath

    print("Image Path")
    print(path)

    print(sensitivity)


    tempLists = path.split("/")
    fileName = tempLists[-1]
    t = tempLists[-2] + "/" + tempLists[-1]
    rel_dir = path.split(t)[0]

    print(rel_dir)

    tissueResultImagePath = os.path.join(rel_dir, TISSUE_MERGE_IMAGE_PATH)
    resImage = []
    
    if not os.path.exists(tissueResultImagePath) :


        for colorOption in ["S","R","G","B"]:


            resultInputPath = rel_dir + colorOption + "/" + fileName
            
            print("resultInputPath =", resultInputPath)

            OUT_PUT_PATH =  rel_dir +  colorOption + "/"
            print(OUT_PUT_PATH)


            mask_output_path = os.path.join(OUT_PUT_PATH, TISSUE_MASK_OUTPUT)

            if (os.path.exists(mask_output_path) == False):
                process_TissueNT_Segmentation(resultInputPath, OUT_PUT_PATH, colorOption)
            
                    
        result_S_path = rel_dir +  "S/" + TISSUE_MASK_OUTPUT
        result_B_path = rel_dir +  "B/" + TISSUE_MASK_OUTPUT
        result_G_path = rel_dir +  "G/" + TISSUE_MASK_OUTPUT
        result_R_path = rel_dir +  "R/" + TISSUE_MASK_OUTPUT


        imageS = cv2.imread(result_S_path)
        imageB = cv2.imread(result_B_path)
        imageG = cv2.imread(result_G_path)
        imageR = cv2.imread(result_R_path)


        imageS = getBinaryImage(imageS)
        imageR = getBinaryImage(imageR)
        imageG = getBinaryImage(imageG)
        imageB = getBinaryImage(imageB)

        resImage = getMergedImageByWholeChannels(imageR,imageG, imageB, imageS)

        cv2.imwrite(tissueResultImagePath, resImage)
    

    else:
        resImage = cv2.imread(tissueResultImagePath)
    


    print(colors)
    [segment, binary] = getResultImageFromColorOptions(resImage, colors, TISSUE_COLORS)


    tissueSegmentResultPath = os.path.join(rel_dir, TISSUE_OUTPUT_SEGMENT_IMAGE_PATH)
    tissueBinaryResultPath = os.path.join(rel_dir, TISSUE_OUTPUT_BINARY_PATH)
    tissueDotPlotResultPath = os.path.join(rel_dir, TISSUE_OUTPUT_DOTPLOT_IMAGE_PATH )


   


    sensitivity = int(sensitivity)

    radius = 6
    if sensitivity < 20:
        radius = 6
    elif sensitivity < 40:
        radius = 5
    elif sensitivity < 60:
        radius = 4
    elif sensitivity < 80:
        radius = 3
    else:
        radius = 2
    
    dotplotImage = getDotPlotImage(segment,radius)

    
    cv2.imwrite(tissueDotPlotResultPath, dotplotImage)
    cv2.imwrite(tissueBinaryResultPath, binary)
    cv2.imwrite(tissueSegmentResultPath, segment)


    tissueSegmentResultThumbImagePath = os.path.join(rel_dir, "reportTabHoleSelectedThumbnail.timg") 
    #save thumb image
    image = cv2.imread(tissueSegmentResultPath)
    (height, width, channel) = image.shape

    img = Image.open(tissueSegmentResultPath)
    img.thumbnail([int(width / 4), int(height / 4)])
    img.save(tissueSegmentResultThumbImagePath, 'png')

    return JSONResponse({"success": "success","image_path" : path , "mask_output_path": tissueBinaryResultPath, "flow_output_path" : tissueSegmentResultPath, "dotplot_output_path"  :  tissueDotPlotResultPath })



@router.post(
    "/tissue_convert_result",
    response_description="TissueNT Convert Processed images to Ome.Tiff file",
)
async def tissueConvertResult(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    imagePath = data.get("image_path")

    
    print("Tissue Convert Result Function")
    print(imagePath)

    filename = imagePath.split("/")[-1]

    print(filename)

    fileext = "." + filename.split(".")[-1]

    filename = filename.split(fileext)[0]

    print("Really file name")
    print(filename)

    #originFilePath = '/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + data.get("image_path")
    originCSVPath = imagePath.split(filename)[0] + filename + '_300.csv'

    dir_path = imagePath.split(filename)[0]

    print(originCSVPath)

    
    mask_image_path = data.get("mask_output_path")
    flow_image_path = data.get("flow_output_path")
    dotplot_image_path = data.get("dotplot_output_path")

    currentTimeStamp = str(int(time.time()))

    mask_output_tiff_path = mask_image_path.split(".jpg")[0] + currentTimeStamp + ".ome.tiff"
    flow_output_tiff_path = flow_image_path.split(".jpg")[0] + currentTimeStamp + ".ome.tiff"
    dotplot_output_tiff_path =  dotplot_image_path.split(".jpg")[0] + currentTimeStamp + ".ome.tiff"


    cmd_str = "sh /app/mainApi/bftools/bfconvert -noflat -separate -tilex 1024 -tiley 1024 -pyramid-scale 2 -pyramid-resolutions 4 -separate -overwrite '" + mask_image_path + "' '" + mask_output_tiff_path + "'"
    print('=====>', cmd_str)
    subprocess.run(cmd_str, shell=True)


    cmd_str = "sh /app/mainApi/bftools/bfconvert -noflat -separate -tilex 1024 -tiley 1024 -pyramid-scale 2 -pyramid-resolutions 4 -separate -overwrite '" + flow_image_path + "' '" + flow_output_tiff_path + "'"
    print('=====>', cmd_str)
    subprocess.run(cmd_str, shell=True)


    cmd_str = "sh /app/mainApi/bftools/bfconvert -noflat -separate -tilex 1024 -tiley 1024 -pyramid-scale 2 -pyramid-resolutions 4 -separate -overwrite '" + dotplot_image_path + "' '" + dotplot_output_tiff_path + "'"
    print('=====>', cmd_str)
    subprocess.run(cmd_str, shell=True)


 
    mask_output_tiff_path = mask_output_tiff_path.replace('/app/mainApi/app/static/', '')
    flow_output_tiff_path = flow_output_tiff_path.replace('/app/mainApi/app/static/', '')
    dotplot_output_tiff_path = dotplot_output_tiff_path.replace('/app/mainApi/app/static/', '')

    # for Overlay
    image = cv2.imread(mask_image_path)
    processMeasureForTissueNT(image, originCSVPath)

    rel_dir = dir_path
    # Get the relative directory
    if "/Overlay" in dir_path:
        rel_dir = dir_path.split("/Overlay")[0]
    
    print("This is the directory path")
    print(rel_dir)


    for color in ["R","G","B","S"]:
            mask_path = os.path.join(rel_dir, color,"mask_output.jpg")
            if os.path.exists(mask_path):
                csv_path  = os.path.join(rel_dir, color,"ashlar_output.ome_300.csv")
                image = cv2.imread(mask_path)
                processMeasureForTissueNT(image, csv_path)


    return JSONResponse({
        "success": "success",
        "mask_output" : mask_output_tiff_path,
        "flow_output" : flow_output_tiff_path,
        "dotplot_output" : dotplot_output_tiff_path
    })




@router.post(
    "/ml_ips_process",
    response_description="ML IPS Process",
)
async def mlIPSProcess(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    sensitivity = data.get("sensitivity")
    colors = data.get('colors')
    colorOption = data.get('colorOption')
    tilemergedFlag = data.get('tilingMergedImageFlag')
    colors = colors.split(",")
    imgPath = data.get("original_image_url")
    paths = imgPath.split(",")
    realPaths = []

    for path in paths:
        realPaths.append('/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + path)

    imagePath = ' '.join(realPaths)

    fileName = imagePath.split("/")[len(imagePath.split("/")) - 1]
    tempPath = tempfile.mkdtemp()
    OUT_PUT_FOLDER = tempPath.split("/")[len(tempPath.split("/")) - 1]
    OUT_PUT_PATH = 'mainApi/app/static/ml_out/' + OUT_PUT_FOLDER
    #paramPath = '/app/mainApi/app/static/ml_out/' + OUT_PUT_FOLDER + '/color_table.txt'
    color_table_path = "app/mainApi/ml_lib/color_table.txt"

    if not os.path.exists(OUT_PUT_PATH):
        os.makedirs(OUT_PUT_PATH)

    cmd_str = "/app/mainApi/ml_lib/segAB {inputPath} {outputPath} /app/mainApi/ml_lib/typeAB/src_param.txt " + sensitivity + " " + color_table_path + " > /app/mainApi/ml_lib/log.txt"
    
    resultInputPath = imagePath


    cmd_str = cmd_str.format(inputPath=resultInputPath, outputPath=OUT_PUT_PATH + "/" + fileName)
    print('----->', cmd_str)
    subprocess.call(cmd_str, shell=True)
    return JSONResponse({"success": "success", "image_path": OUT_PUT_PATH + "/" + fileName})



def getMergeImageByCountResult(image1Path,  color_option, sensitivity, resultpath):
    try:    
        image = cv2.imread(image1Path)

        COLORS = [
        ( 0   , 0 , 255	),
        ( 0  , 0 , 128	),
        ( 255 ,  0, 255	),
        ( 128  , 0 ,128	),
        ( 0 ,255, 255	),
        ( 0 ,128 ,128	),
        ( 128 ,128 ,128	),
        ( 128 ,  0,   0	),
        ( 0 ,255,   0	),
        ( 0 ,128,   0	),
        ( 196 ,196, 196	),
        ( 128, 128,   0	),
        ( 255 ,255  , 0	),
        ( 255  , 0  , 0	),
        ]

        k = 1
        color_num = 0
        radius = 1
        if sensitivity == 'undefined':
            sensitivity = 50
        sensitivity = float(sensitivity) 

        if sensitivity < 35:
            radius = 1
        elif sensitivity < 70:
            radius = 2
        else:
            radius = 3

        if color_option == "S":
            color_num = 12
            k = 1
        if color_option == "R":
            color_num = 11
            k = 2
        if color_option == "G":
            color_num = 10
            k = 3
        if color_option == "B":
            color_num = 9
            k = 4
        if color_option == "B+R":
            color_num = 8
            k = 5
        if color_option == "B+G":
            color_num = 7
            k = 6

        if color_option == "B+G+R":
            color_num = 6
            k = 7
        if  color_option == "S+G+R":
            color_num = 5
            k = 8
        if color_option == "S+B+R" : 
            color_num = 3
            k = 9
        if color_option == "S+B+G":
            color_num = 2
            k = 10
        if color_option == "S+B+G+R":
            color_num = 0
            k = 11

                
        pixel_values = image.reshape((-1, 3))

        # convert to float
        pixel_values = np.float32(pixel_values)


        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


        # convert back to 8 bit values
        centers = np.uint8(centers)

        # flatten the labels array
        labels = labels.flatten()


        for c in range(k):
            norm = np.linalg.norm(centers[c])
            if(norm < 15): continue
            centers[c] = COLORS[c+color_num]

        # convert all pixels to the color of the centroids
        segmented_image = centers[labels.flatten()]



        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(image.shape)
        # show the image


        (height, width, channel) = segmented_image.shape


        step = radius * 3 + 1
        thickness = radius

        for x in range(1,width,step):
            for y in range(1, height,step):
                colors = segmented_image[y,x]
                color = (int(colors[0]), int(colors[1]), int(colors[2]))
                cv2.circle(image, (x,y), radius , color, thickness)


        cv2.imwrite(resultpath, image)
    
    except:
        return





@router.post(
    "/ml_convert_result",
    response_description="ML Convert Processed images to Ome.Tiff file",
)
async def mlConvertResult(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    imagePath = data.get("image_path")
    colors = data.get("colors")
    sensitivity = data.get("sensitivity")

    print(colors)
    print(sensitivity)

    fileName = imagePath.split("/")[len(imagePath.split("/")) - 1]
    realName = os.path.splitext(fileName)[0]
    tempPath = tempfile.mkdtemp()
    print("ml-convert-result-filename:", realName)
    csvPath = os.path.splitext(imagePath)[0] + '_300.csv'
    outputFolder = '/app/mainApi/app/static' + tempPath
    
    
    originFilePath = '/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + data.get("original_image_path")
    originCSVPath = os.path.splitext(originFilePath)[0] + '_300.csv'

    print(originFilePath)

    ttpath = originFilePath.split(".")[0] + ".jpg"
    

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    realPath = os.path.splitext(imagePath)[0] + 'a_2.jpg'
    countPath = os.path.splitext(imagePath)[0] + 'a_13.jpg'

    color_segmentedImage1path = os.path.splitext(imagePath)[0] + 'a_3.jpg'
    color_segmentedImage2path = os.path.splitext(imagePath)[0] + '_252.jpg'
    color_segmentedImage3path = os.path.splitext(imagePath)[0] + '_251.jpg'


    getMergeImageByCountResult(ttpath,  colors, sensitivity, color_segmentedImage1path)


    outputFolder = '/app/mainApi/app/static' + tempPath
    outputPath = outputFolder + '/' + realName + 'a_2.ome.tiff'
    outputPath1 = outputFolder + '/' + realName + 'a_2_temp.tiff'
    outputPath2 = outputFolder + '/' + realName + 'a_2_temp.ome.tiff'


    count_reult_path =  outputFolder + '/' + realName + 'a_13.ome.tiff'
    color_segment1_path = outputFolder + '/' + realName + 'a_3.ome.tiff'
    color_segment2_path = outputFolder + '/' + realName + '_252.ome.tiff'
    color_segment3_path = outputFolder + '/' + realName + '_251.ome.tiff'

    bfconv_cmd = f"sh /app/mainApi/bftools/bfconvert -noflat -separate -tilex 1024 -tiley 1024 -pyramid-scale 2 -pyramid-resolutions 4 -separate -overwrite '{countPath}' '{count_reult_path}'"
    await shell(bfconv_cmd)

    print("===>before convert by PILLOW")
    # Open the JPG file
    #input_file = '/path/to/input.jpg'
    image = Image.open(realPath)
    # Convert the image to OME-TIFF
    output_file = outputPath1
    image.save(outputPath1, format='TIFF', compression='tiff_lzw')



    cmd_str = "sh /app/mainApi/bftools/bfconvert -noflat -separate -tilex 1024 -tiley 1024 -pyramid-scale 2 -pyramid-resolutions 4 -separate -overwrite '" + realPath + "' '" + outputPath1 + "'"
    print('=====>', cmd_str)
    subprocess.run(cmd_str, shell=True)

    cmd_str = "sh /app/mainApi/bftools/bfconvert -noflat -separate -tilex 1024 -tiley 1024 -pyramid-scale 2 -pyramid-resolutions 4 -separate -overwrite '" + color_segmentedImage1path + "' '" + color_segment1_path + "'"
    print('=====>', cmd_str)
    subprocess.run(cmd_str, shell=True)

    cmd_str = "sh /app/mainApi/bftools/bfconvert -noflat -separate -tilex 1024 -tiley 1024 -pyramid-scale 2 -pyramid-resolutions 4 -separate -overwrite '" + color_segmentedImage2path + "' '" + color_segment2_path + "'"
    print('=====>', cmd_str)
    subprocess.run(cmd_str, shell=True)

    cmd_str = "sh /app/mainApi/bftools/bfconvert -noflat -separate -tilex 1024 -tiley 1024 -pyramid-scale 2 -pyramid-resolutions 4 -separate -overwrite '" + color_segmentedImage3path + "' '" + color_segment3_path + "'"
    print('=====>', cmd_str)
    subprocess.run(cmd_str, shell=True)


    print("===>after convert by PILLOW")





    data = pd.read_csv(csvPath)
    data = processBasicMeasureData(csvPath, data)
    data.to_csv(csvPath)

    shutil.copy(csvPath, originCSVPath)

    

    return JSONResponse({
        "success": "success",
        "image_path": tempPath + '/' + realName + 'a_2_temp.tiff',
        "count_path": tempPath + '/' + realName + 'a_13.ome.tiff',
        'color_segment1_result' : tempPath + '/' + realName + 'a_3.ome.tiff',
        'color_segment2_result' : tempPath + '/' + realName + '_252.ome.tiff',
        'color_segment3_result' : tempPath + '/' + realName + '_251.ome.tiff',
        "csv_path": csvPath
    })

@router.post(
    "/ml_convert_result_select",
    response_description="ML Convert Processed images to Ome.Tiff file",
)
async def mlConvertResultSelect(request: Request, current_user: UserModelDB = Depends(get_current_user)):
    data = await request.form()
    imagePath = data.get("image_path")
    originalImagePath = '/app/mainApi/app/static/' + str(PyObjectId(current_user.id)) + '/' + data.get("original_image_path")
    fileName = imagePath.split("/")[len(imagePath.split("/")) - 1]
    realName = os.path.splitext(fileName)[0]
    tempPath = tempfile.mkdtemp()
    print("ml-convert-result-filename:", realName)
    csvPath = os.path.splitext(imagePath)[0] + '_300.csv'
    outputFolder = '/app/mainApi/app/static' + tempPath


    originCSVPath = os.path.splitext(originalImagePath)[0] + '_300.csv'

    print(originCSVPath)



    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    realPath = os.path.splitext(imagePath)[0] + 'a_2.jpg'
    outputFolder = '/app/mainApi/app/static' + tempPath

    mergedPath = outputFolder + '/' + realName + '_merged.ome.tiff'

    # Load the OME-TIFF file
    ome_tiff = tifffile.imread(originalImagePath)

    # Get the number of channels
    num_channels = ome_tiff.shape[0]
    input_files = [outputFolder + '/' + 'output.tiff']

    print("===>origin channels:", num_channels)

    # Loop over each channel and save as a separate TIFF file
    for i in range(num_channels):
        # Get the image data for this channel
        channel_data = ome_tiff[i]

        # Save the channel data as a TIFF file
        tifffile.imsave(outputFolder + '/' + f'channel_{i}.tiff', channel_data)
        input_files.append(outputFolder + '/' + f'channel_{i}.tiff')

    # Load the JPEG file
    img = Image.open(realPath)
    # Convert the image to grayscale
    gray_img = img.convert('L')
    # Save the grayscale image as a TIFF file
    gray_img.save(outputFolder + '/' + 'output.tiff')

    my_string = ' '.join(input_files)
    cmd_str = f'python /app/mainApi/ml_lib/pyramid_assemble.py {my_string} {mergedPath} --pixel-size 1'
    print('=====>', cmd_str)
    subprocess.run(cmd_str, shell=True)

    # metadata = bioformats.get_omexml_metadata('/app/mainApi/app/static/6461894c49dbc4f3496599ba/1/my_test/at3_1m4_01.ome.tiff')
    # xml = bioformats.OMEXML(metadata)
    # print("ome-xml:", xml)

    data = pd.read_csv(csvPath)
    data = processBasicMeasureData(csvPath, data)
    data.to_csv(csvPath)

    shutil.copy(csvPath, originCSVPath)



    return JSONResponse({
        "success": "success",
        "image_path": tempPath + '/' + realName + '_merged.ome.tiff',
        "csv_path": csvPath
    })

@router.get("/test")
def read_root():
    print('sdfsdfsdf')
    with h5.File('example.h5', 'w') as f:
    # create a group
        group = f.create_group('mygroup')
        
        # create a dataset inside the group
        data = [1, 2, 3, 4, 5]
        group.create_dataset('mydata', data=data)

# read the data from the file
    with h5.File('example.h5', 'r') as f:
        # get the dataset
        dataset = f['mygroup/mydata']
        
        # print the dataset
        print(dataset[:])
    return {"Ping": "Pang"}




@router.post('/measure/update_measure_data')
async def update_measure_data(
    request: Request,
    keyList: List[str] = Form(...)
):
    print(request)
    data = await request.form()
    print('======> keyList', keyList)
    res = update_h5py_file(data, keyList)
    print(res)

    # for key in keyList:
    #     value = data.get(key)
    #     print(json.loads(value))
    #     print('=======>', key)
    return res





@router.post('/measure/create_measure_data')
async def create_measure_data(
    request: Request,
    keyList: List[str] = Form(...)
):
    print(request)
    data = await request.form()
    print('======> keyList', keyList)
    res = update_h5py_file(data, keyList)

    filename = data["originPath"].split('/')[-1]

    originPath = data["originPath"]
    originPath = originPath.split("download/?path=")[1]
    originPath = originPath.split(filename)[0]
    originPath = '/app/mainApi/app/static/' + originPath

    csv_fileName = res["file_path"] .split('/')[-1]


    originPath = originPath + csv_fileName
    
    tempFilePath = res["file_path"] 
    tempFilePath = '/app/mainApi/app/static/measure_out/' +  tempFilePath
    print(res)

    print(tempFilePath)
    print(originPath)

    
    shutil.copy(tempFilePath, originPath)



    # for key in keyList:
    #     value = data.get(key)
    #     print(json.loads(value))
    #     print('=======>', key)
    return res


@router.post('/measure/processBasicMeasure',
    response_description="process Basic Measure",
    status_code=status.HTTP_200_OK,
    )
async def processBasicMeasure(
    request: Request,
):
    body_bytes = await request.body()
    params = json.loads(body_bytes)
    
    filepath = params["path"]

    pathlist = filepath.split("/image/download_csv?path=")
    filepath = '/app/mainApi/app/static' + pathlist[1]

    data = pd.read_csv(filepath)

    data = processBasicMeasureData(filepath, data)

    data.to_csv(filepath)
    
    print(data)
    print(filepath)

    return JSONResponse({"Result" : "Success"})



@router.post(
    "/deconv2D",
    response_description="Deconvolution 2D",
    status_code=status.HTTP_200_OK,
)
async def processDeconv2D(
    request: Request,
):
    body_bytes = await request.body()
    params = json.loads(body_bytes)

    filepath = params["filename"]
    effectiveness = params['effectiveness']
    isroi = params['isroi']
    dictRoiPts = params['dictRoiPts']

    #print(params)
    print("Start Processing for Deconvolution 2D")

    abs_path = await Deconv.FlowDecDeconvolution2D(
        filepath, effectiveness, isroi, dictRoiPts
    )



    return JSONResponse(abs_path) 




@router.post(
    "/deconv3D",
    response_description="Deconvolution 3D",
    status_code=status.HTTP_200_OK,
)
async def processDeconv3D(
    request: Request,
):
    body_bytes = await request.body()
    params = json.loads(body_bytes)

    filepath = params["filename"]
    effectiveness = params['effectiveness']
    isroi = params['isroi']
    dictRoiPts = params['dictRoiPts']

    #print(params)
    print("Start Processing for Deconvolution 3D")

    abs_path = await Deconv.FlowDecDeconvolution2D(
        filepath, effectiveness, isroi, dictRoiPts
    )







@router.get(
    "/getUsageDiskSpace",
    response_description="Get the Usage of Disk Space",
    status_code=status.HTTP_200_OK,
)
async def getUsageDiskSpace(
    user: UserModelDB = Depends(get_current_user), 
):
    user_dir = os.path.join("/app/mainApi/app/static/", str(PyObjectId(user.id)) )
    print(user_dir)


    #assign total size
    totSize = 0


    #get sizes
    for path, dirs, files in os.walk(user_dir):

        for f in files:
            fp = os.path.join(path, f)
            totSize += os.path.getsize(fp)
    

@router.post(
    "/processHeatMap",
    response_description="process the Heatmap",
    status_code=status.HTTP_200_OK,
)
async def processHeatMap(
    request: Request,
):
    body_bytes = await request.body()
    print(body_bytes)
    params = json.loads(body_bytes)

    file_path = params["image_path"]
    print("this is file_path",file_path)
    
    split_path = file_path.split("download/?path=")[1]
    filename = split_path.split("/")[-1]
    print("this is filename",filename)

    temp_dir = split_path.split(filename)[0]
    print("this is filename",temp_dir)

    if "Overlay" in temp_dir:
        temp_dir = temp_dir.split("Overlay")[0]

    work_dir = os.path.join("/app/mainApi/app/static/",temp_dir )

    input_path = work_dir
    saveHeatmap(input_path, work_dir + "1.png",3,12)
    saveHeatmap(input_path, work_dir + "2.png",2,4)
    saveHeatmap(input_path, work_dir + "3.png",2,13,'cividis' )
    saveHeatmap(input_path, work_dir + "4.png",4,15,'magma' )
    saveHeatmap(input_path, work_dir + "5.png",8,15,'gist_gray_r' )

    print(work_dir)

    addROIInHeatMapImage(work_dir + "1.png",work_dir + "roi_1.png", [65,85,25,45])
    addROIInHeatMapImage(work_dir + "2.png",work_dir + "roi_2.png", [65,85,25,45])
    addROIInHeatMapImage(work_dir + "3.png",work_dir + "roi_3.png", [65,85,25,45])
    addROIInHeatMapImage(work_dir + "4.png",work_dir + "roi_4.png", [65,85,25,45])
    addROIInHeatMapImage(work_dir + "5.png",work_dir + "roi_5.png", [65,85,25,45])

    return work_dir


@router.post(
    "/ml_mouse_tracking_process_upload",
    response_description="Mouse tracking upload process",
    status_code=status.HTTP_200_OK,
)
async def processMouseTrackingUpload(
    request: Request,
    user: UserModelDB = Depends(get_current_user), 
):

    user_dir = os.path.join("/app/mainApi/app/static/", str(PyObjectId(user.id)) )
    print(user_dir)
    data = await request.form()
    video = data.get("file")

    folder_path = os.path.join(user_dir, 'mouse_track')

    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder created at: {folder_path}")
        except OSError as e:
            print(f"Error creating folder: {e}")
    else:
        print(f"Folder already exists at: {folder_path}")

    file_path = os.path.join(user_dir, 'mouse_track',time.strftime("%H-%M-%S") + ".mp4")

    with open(file_path, "wb") as f:
        f.write(video.file.read())
  
    return {"result": "success", "file_path" : file_path}




@router.post(
    "/ml_mouse_tracking_process",
    response_description="Mouse tracking Trainning process",
    status_code=status.HTTP_200_OK,
)
async def processMouseTrackingProcess(
    request: Request,
    user: UserModelDB = Depends(get_current_user), 
):

    # user_dir = os.path.join("/app/mainApi/app/static/", str(PyObjectId(user.id)) )
    # print(user_dir)
    data = await request.form()
    file_path = data.get("file")
    print(file_path)

    #processTopViewMouseTracking(file_path)

    return {"result": "success", "file_path" : file_path}