import os
import cv2
from pathlib import Path

root_path = Path('input data path')


for split_set in ["val"]:
    mask_paths = root_path / split_set / 'mask_SASMA'
    output_dir = root_path / split_set / 'mask_SASMA_edge'
    os.makedirs(output_dir, exist_ok=True)
    # for i in range(900):
    #     if not os.path.exists(output_dir / f"{str(i).zfill(4)}.png"):
    #         print(f"{str(i).zfill(4)}.png")
    for mask_path in mask_paths.iterdir():
        img = cv2.imread(str(mask_path))
        canny_img = cv2.Canny(img, 10, 180)
        cv2.imwrite(str(output_dir / mask_path.name), canny_img)