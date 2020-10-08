from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
global text
text=[]
def texti(img_path):
    endpoint="https://texti.cognitiveservices.azure.com/"
    subscription_key="855ccecb6ea649c1a3372d5758c4b4ed"
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    local_image_printed_text_path = img_path
    local_image_printed_text = open(local_image_printed_text_path, "rb")

    ocr_result_local = computervision_client.recognize_printed_text_in_stream(local_image_printed_text)
    for region in ocr_result_local.regions:
        for line in region.lines:
            #print("Bounding box: {}".format(line.bounding_box))
            s=''
            for word in line.words:
                s += word.text + " "
            print(s)
            text.append(s)
            print()
    return text
    