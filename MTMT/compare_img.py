import os
from pathlib import Path

import cv2

dataset_path = Path('datasets/ntire2023_official_warped')
compare_path = Path('log/mask_v_epoch5e5')
compare_paths = [
    dataset_path / 'val' / 'input',
    dataset_path / 'val' / 'mask_v_mtmt',
    compare_path / 'results10000',
    compare_path / 'results20000',
    compare_path / 'results30000',
    compare_path / 'results40000',
]

output_dir = Path('compare/1')
os.makedirs(output_dir, exist_ok=True)
with open(output_dir / 'memo.txt', 'w') as f:
    f.write('compare list\n')
    for path in compare_paths:
        f.write(f"* {path}\n")

compare_files = []
for path in compare_paths:
    compare_files.append(list(path.iterdir()))

for i in range(len(compare_files[0])):
    img_list = []
    for j in range(len(compare_paths)):
        img = cv2.imread(str(compare_files[j][i]))
        img_with_margin = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
        img_list.append(img_with_margin)
    result_img = cv2.hconcat(img_list)
    cv2.imwrite(str(output_dir / compare_files[0][i].name), result_img)
