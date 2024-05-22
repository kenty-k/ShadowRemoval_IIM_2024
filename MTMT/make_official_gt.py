#%%
import os
import sys
import shutil
import numpy as np
from pathlib import Path

import cv2

def gamma_correction(img, gamma):
    # テーブルを作成する。
    table = (np.arange(256) / 255) ** gamma * 255
    # [0, 255] でクリップし、uint8 型にする。
    table = np.clip(table, 0, 255).astype(np.uint8)

    return cv2.LUT(img, table)

datasets_dir = Path('datasets/ntire2023_official_warped')
image_registration_dir = Path('image_registration')

remove_list = [
    '0010',
    '0099',
    '0526',
    '0532',
]
not_warped_list = [
    '0528',
    '0679',
    '0681',
    '0682',
    '0698',
    '0704',
    '0715',
    '0747',
]

for split_set in ['train', 'val']:
    output_dir = datasets_dir / split_set / 'diff'
    os.makedirs(output_dir, exist_ok=True)
    for diff_path in list((image_registration_dir / split_set / 'diff_warped').iterdir()):
        if diff_path.stem in remove_list: continue
        # if diff_path.stem not in not_warped_list:
        #     img = cv2.imread(str(image_registration_dir / split_set / 'diff_warped' / diff_path.name))
        # else:
        #     img = cv2.imread(str(image_registration_dir / split_set / 'diff' / diff_path.name))
        # img = gamma_correction(img, 0.6)
        # cv2.imwrite(str(output_dir / diff_path.name), img)
        if diff_path.stem not in not_warped_list:
            shutil.copyfile(image_registration_dir / split_set / 'diff_warped' / diff_path.name, output_dir / diff_path.name)
        else:
            shutil.copyfile(image_registration_dir / split_set / 'diff' / diff_path.name, output_dir / diff_path.name)

