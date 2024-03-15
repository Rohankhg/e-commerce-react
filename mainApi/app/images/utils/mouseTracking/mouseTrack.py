import deeplabcut
import os



async def processTopViewMouseTracking (video_path):

    superanimal_name  = "superanimal_topviewmouse" #@param ["superanimal_topviewmouse", "superanimal_quadruped"]

    #The pcutoff is for visualization only, namely only keypoints with a value over what you set are shown. 0 is low confidience, 1 is perfect confidience of the model.
    pcutoff = 0.8 #@param {type:"slider", min:0, max:1, step:0.05}


    # The purpose of the scale list is to aggregate predictions from various image sizes. We anticipate the appearance size of the animal in the images to be approximately 400 pixels.
    scale_list = range(50, 350, 5)

    deeplabcut.video_inference_superanimal([video_path], superanimal_name, scale_list=scale_list, video_adapt = False)

    return video_path