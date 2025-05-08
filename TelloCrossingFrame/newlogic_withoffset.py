import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO

# ===== CONFIGURATION =====
MODEL_PATH = r"D:\AAAUAVdata\Yolo8nTrainingBasicFiles\bestBy11s.pt"
CAM_INDEX   = 0  # 0 for on‑board / webcam in sim

# Alignment & offset
ALIGN_THRESHOLD_PX = 40          # when |dx| or |dy| < threshold, consider aligned
GAIN_YAW           = 0.15        # proportional gain for yaw correction
GAIN_THROTTLE      = 0.15        # proportional gain for vertical correction

OFFSET_RATIO  = 0.35             # percentage of bbox height to shift target downward
OFFSET_PIXELS = None
# 优先级：class id 0 → Gate1，1 → Gate2，2 → Gate3
GATE_PRIORITY = [0, 1, 2]
# 搜索时的旋转速度（yaw）
SEARCH_YAW_VEL = 20
# set to int (e.g. 50) to use fixed px instead.  None = use ratio

# ==========================

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    将Tello无人机传回的BGR格式图像转换为RGB格式
    参数:
        image (numpy.ndarray): BGR格式的图像数组，维度为(高, 宽, 3)
    返回:
        numpy.ndarray: RGB格式的图像数组，维度为(高, 宽, 3)
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def draw_overlay(frame, results, target_pt=None):
    """Draw YOLO detections and target point for visual debug."""
    annotated = results[0].plot() if results else frame
    if target_pt is not None:
        tx, ty = target_pt
        cv2.circle(annotated, (tx, ty), 6, (0, 0, 255), -1)
        cv2.drawMarker(annotated, (annotated.shape[1]//2, annotated.shape[0]//2), (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
    return annotated

class FrameNavigator:
    """Utility class with static helpers for gate tracking & drone alignment"""

    @staticmethod
    def bbox_center(bbox, offset_ratio: float = OFFSET_RATIO,
                    offset_pixels: int | None = OFFSET_PIXELS):
        """Return (cx, cy) shifted **downward** by offset from bbox center."""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if offset_pixels is not None:
            cy += offset_pixels
        else:
            h = y2 - y1
            cy += int(h * offset_ratio)
        return cx, cy

    @staticmethod
    def align_with_target(tello: Tello, frame_shape, target_pt):
        """Send RC control signals so that frame center moves toward target_pt."""
        h, w, _ = frame_shape
        cx_frame, cy_frame = w // 2, h // 2
        tx, ty = target_pt
        dx = tx - cx_frame  # +right / -left
        dy = ty - cy_frame  # +down  / -up

        # Yaw (horizontal) & Throttle (vertical) corrections
        yaw_vel   = int(GAIN_YAW      * dx)
        thr_vel   = int(-GAIN_THROTTLE * dy)  # negative because up is -

        # clip small errors
        if abs(dx) < ALIGN_THRESHOLD_PX:
            yaw_vel = 0
        if abs(dy) < ALIGN_THRESHOLD_PX:
            thr_vel = 0

        # pitch forward constantly once aligned horizontally & vertically enough
        pitch_vel = 20 if yaw_vel == 0 and thr_vel == 0 else 0

        tello.send_rc_control(roll=0, pitch=pitch_vel, throttle=thr_vel, yaw=yaw_vel)
        return pitch_vel > 0


def main():
    tello = Tello()
    tello.connect()
    tello.streamon()

    model = YOLO(MODEL_PATH)

    flying = False   # 是否已起飞
    auto_mode = False  # 手动/自动模式

    try:
        while True:
            # 1. 读取并转换图像
            frame = tello.get_frame_read().frame
            frame_rgb = bgr_to_rgb(frame)

            # 2. YOLO 检测
            results = model.predict(frame_rgb, conf=0.4, verbose=False, save=False)
            detections = results[0].boxes

            # 收集所有检测到的门 (class_id, bbox)
            gates = [
                (int(box.cls), box.xyxy[0].cpu().numpy().astype(int))
                for box in detections
                if int(box.cls) in GATE_PRIORITY
            ]

            target_pt = None
            # 3. 自动模式逻辑
            if auto_mode:
                if gates:
                    # 按优先级依次选 Gate1→Gate2→Gate3
                    for cls_id in GATE_PRIORITY:
                        cls_bboxes = [bbox for c, bbox in gates if c == cls_id]
                        if not cls_bboxes:
                            continue
                        # 同类别中取最大框
                        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in cls_bboxes]
                        biggest = cls_bboxes[int(np.argmax(areas))]
                        target_pt = FrameNavigator.bbox_center(biggest)
                        FrameNavigator.align_with_target(tello, frame_rgb.shape, target_pt)
                        break
                else:
                    # 找不到任何门 → 原地旋转搜索
                    tello.send_rc_control(0, 0, 0, SEARCH_YAW_VEL)

            # 4. 键盘监听
            key = cv2.waitKey(1) & 0xFF

            # 起飞 / 降落
            if key == ord('t') and not flying:
                tello.takeoff()
                flying = True
                auto_mode = False   # 起飞后保持手动
            elif key == ord('l') and flying:
                tello.land()
                flying = False

            # 模式切换（只能在飞行中切换）
            elif key == ord('m') and flying:
                auto_mode = not auto_mode
                print(f"[MODE] {'AUTO' if auto_mode else 'MANUAL'}")

            # 5. 手动模式遥控
            if flying and not auto_mode:
                roll = pitch = thr = yaw = 0
                vel = 25
                if key == ord('w'): pitch =  vel
                if key == ord('s'): pitch = -vel
                if key == ord('a'): roll  = -vel
                if key == ord('d'): roll  =  vel
                if key == ord('o'): thr   =  vel
                if key == ord('k'): thr   = -vel
                if key == ord('q'): yaw   = -vel
                if key == ord('e'): yaw   =  vel
                tello.send_rc_control(roll, pitch, thr, yaw)

            # 6. 可视化
            show = draw_overlay(frame_rgb, results, target_pt)
            mode_txt = "AUTO" if auto_mode else "MANUAL"
            cv2.putText(show, f"Mode: {mode_txt}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Tello Gate Navigation", show)

            # 退出脚本
            if key == ord('q'):
                break

    finally:
        tello.send_rc_control(0, 0, 0, 0)
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


