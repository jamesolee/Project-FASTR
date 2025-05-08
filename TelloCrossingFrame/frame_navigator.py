# frame_navigator.py  — 仅保留“靠检测框中心对准”逻辑
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class FrameNavigator:
    """
    按检测框(bbox)中心点对准并穿越。
    * gate_index 用于多框（Gate1→Gate2→Gate3……）依次飞。
    """

    def __init__(self, tello, auto_speed=30, align_threshold=50):
        self.tello = tello
        self.AUTO_SPEED = auto_speed          # cm/s
        self.align_threshold = align_threshold  # 允许的像素误差
        self.gate_index = 0

    # ------------------------------------------------------------------ #
    # ↓↓↓↓↓ 新逻辑：使用 YOLO 检测框中心 ↓↓↓↓↓
    # ------------------------------------------------------------------ #
    @staticmethod
    def bbox_center(xyxy):
        """xyxy -> (cx, cy)"""
        x1, y1, x2, y2 = xyxy
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def select_gate(self, all_bboxes):
        """取当前 gate_index 的框（假设按左→右或近→远顺序排序进列表）"""
        if len(all_bboxes) <= self.gate_index:
            return None
        return all_bboxes[self.gate_index]

    # -------------------- 核心：对准函数 -------------------- #
    def align_with_frame(self, frame_data, image_shape):
        """
        根据框中心对齐至画面中心。
        frame_data['all_bboxes'] 是 [xyxy, xyxy, …]
        """
        h, w = image_shape[:2]
        cx_img, cy_img = w // 2, h // 2

        bbox = self.select_gate(frame_data["all_bboxes"])
        if bbox is None:
            logger.warning("Not enough gates detected – waiting …")
            return False

        cx_box, cy_box = self.bbox_center(bbox)
        err_x = cx_box - cx_img
        err_y = cy_box - cy_img

        lr = int(np.clip(err_x / 8, -self.AUTO_SPEED, self.AUTO_SPEED))
        ud = int(np.clip(-err_y / 8, -self.AUTO_SPEED, self.AUTO_SPEED))

        logger.info(f"[Gate {self.gate_index+1}] err_x={err_x} err_y={err_y}")

        if abs(err_x) > self.align_threshold or abs(err_y) > self.align_threshold:
            self.tello.send_rc_control(lr, 0, ud, 0)
            return False
        return True
