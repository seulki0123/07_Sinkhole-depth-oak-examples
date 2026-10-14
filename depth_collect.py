import depthai as dai
import numpy as np
import cv2
from pathlib import Path
from datetime import timedelta
from utils.arguments import initialize_argparser

fps = 10
SAVE = Path("runs/datas")
SAVE.mkdir(exist_ok=True)
i = 1

with dai.Pipeline() as p:
    left = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    l = left.requestOutput((640, 480), dai.ImgFrame.Type.NV12, fps=fps)
    r = right.requestOutput((640, 480), dai.ImgFrame.Type.NV12, fps=fps)

    depth = p.create(dai.node.Depth)
    depth.build(dai.node.Depth.Algorithm.AUTO, fps=fps, size=(640, 480))

    sync = p.create(dai.node.Sync)
    sync.setRunOnHost(True)
    sync.setSyncThreshold(timedelta(seconds=0.5 / fps))

    l.link(sync.inputs["left"])
    r.link(sync.inputs["right"])
    depth.depth.link(sync.inputs["depth"])
    q = sync.out.createOutputQueue(maxSize=8, blocking=True)

    p.start()
    try:
        while p.isRunning():
            group = q.get()
            folder = SAVE / f"{i:06d}"
            folder.mkdir(exist_ok=True)

            for name in ("left", "right", "depth"):
                msg = group[name]
                frame = msg.getFrame() if name == "depth" else cv2.cvtColor(
                    msg.getCvFrame(), cv2.COLOR_BGR2GRAY
                )
                np.save(folder / f"{i:06d}_{name}.npy", frame)

            if i % 100 == 0:
                print(f"Saved {i}")
            i += 1
    except KeyboardInterrupt:
        pass

print(f"Done: {i - 1} sets")