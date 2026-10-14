import numpy as np
import cv2
from pathlib import Path

DATA = Path("runs/datas")
OUT = Path("runs/vis")
OUT.mkdir(exist_ok=True)

for folder in sorted(DATA.iterdir()):
    if not folder.is_dir():
        continue

    left = np.load(folder / f"{folder.name}_left.npy")
    right = np.load(folder / f"{folder.name}_right.npy")
    depth = np.load(folder / f"{folder.name}_depth.npy")

    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)
    depth = cv2.resize(depth, (640, 480))
    
    print(folder.name)
    print("left :", left.shape, left.dtype)
    print("right:", right.shape, right.dtype)
    print("depth:", depth.shape, depth.dtype)

    image = cv2.hconcat([left, right, depth])
    cv2.imwrite(str(OUT / f"{folder.name}.jpg"), image)

print(f"Done! Images saved to {OUT}/")