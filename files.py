import pandas as pd
import shutil
from imutils import paths
import os

# noise_files = pd.read_excel(r"E:\scaler\Audio denoising\ESC-50-master\ESC-50-master\meta\need.xlsx")
# for index,row in noise_files.iterrows():
#     file_name = row['filename']
#     src_path = r"E:\\scaler\\Audio denoising\\ESC-50-master\\ESC-50-master\\audio\\"+file_name
#     dest_path = r"E:\\scaler\\Audio denoising\\Train\\noise\\"
#     shutil.copy2(src_path,dest_path)


for dir1 in os.listdir(r"E:\\scaler\\Audio denoising\\dev-clean\\LibriSpeech\\dev-clean\\"):
    for dir2 in os.listdir(r"E:\\scaler\\Audio denoising\\dev-clean\\LibriSpeech\\dev-clean\\"+dir1+"\\"):
        files = os.listdir(r"E:\\scaler\\Audio denoising\\dev-clean\\LibriSpeech\\dev-clean\\"+dir1+"\\"+dir2+"\\")
        for file in files:
            if file.endswith(".flac"):
                shutil.copy2(r"E:\\scaler\\Audio denoising\\dev-clean\\LibriSpeech\\dev-clean\\"+dir1+"\\"+dir2+"\\"+file,r"E:\\scaler\\Audio denoising\\Train\\clean_voice\\")
