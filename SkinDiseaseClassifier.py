#Importing Libraries
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image

#Converting Color images to Grey Scale

#================================================================================================================================
# 1. Renaming files in the folders

def renameFiles(dr):
    import os

    directory = os.fsencode(dr)

    i = 1
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".webp"):
            old_file = os.path.join(dr,filename)
            folderName = os.path.basename(os.path.dirname(os.path.join(dr,filename)))
            print(folderName)
            new_file = os.path.join(dr,f'{i}_{folderName}.png')
            os.rename(old_file, new_file)
            i += 1
        else:
            continue

dr1 = "/Users/thejakamahaulpatha/PycharmProjects/HealthImageClassifier/dataset/test/Basal cell carcinoma"
dr2 = "/Users/thejakamahaulpatha/PycharmProjects/HealthImageClassifier/dataset/test/Melanoma"

renameFiles(dr1)
renameFiles(dr2)


#================================================================================================================================


#
# # 1. Converting Training Data
#
# test_Basal_gray = []
#
# dr = "/Users/thejakamahaulpatha/PycharmProjects/HealthImageClassifier/dataset/test/Basal_cell_carcinoma"
# directory = os.fsencode(dr)
#
#
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     # print(filename)
#     if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(
#             ".webp"):
#         imageResized = Image.open(os.path.join(dr,filename)).resize((28,28))
#         imageResized.save(os.path.join(dr,filename))
#         # image = cv2.imread(os.path.join(dr,filename))
#         # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # test_Basal_gray.append(gray)
#
#     else:
#         continue


# test_Basal_gray_np = np.array(test_Basal_gray)
# print(test_Basal_gray_np.shape)
# print(test_Basal_gray_np)
# print("="*40)

# testlist = []
# image = cv2.imread(os.path.join(dr, '15_Basal.png'))
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# testlist.append(gray)
# print(np.array(testlist))
# print(np.array(testlist).shape)
#
#
# filepath = os.path.join(dr,'15_Basal.png')
# width, height = Image.open(filepath).size
# print(width*height)


