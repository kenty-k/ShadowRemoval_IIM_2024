import cv2
import glob
import os
directory_path = '../results/probabilistic_model'
png_files = glob.glob(os.path.join(directory_path, '*.png'))
for file_path in png_files:
    image = cv2.imread(file_path)
    cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])