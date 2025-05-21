import cv2
import numpy as np
import os
from pynput.mouse import Listener

class DataCollector:
    def __init__(self, save_dir="dataset"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.video_capture = cv2.VideoCapture(0)
        
        # 设置相机分辨率为1280x720
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 获取实际设置的分辨率
        self.width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"相机分辨率设置为: {self.width}x{self.height}")

    def normalize(self, x):
        minn, maxx = x.min(), x.max()
        return (x - minn) / (maxx - minn)

    def scan(self, image_size=(32, 32)):
        _, frame = self.video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(gray, 1.3, 10)
        
        if len(boxes) == 2:
            eyes = []
            for box in boxes:
                x, y, w, h = box
                eye = frame[y:y + h, x:x + w]
                eye = cv2.resize(eye, image_size)
                eye = self.normalize(eye)
                eye = eye[10:-10, 5:-5]
                eyes.append(eye)
            return (np.hstack(eyes) * 255).astype(np.uint8)
        return None

    def on_click(self, x, y, button, pressed):
        if pressed:
            eyes = self.scan()
            if eyes is not None:
                filename = os.path.join(self.save_dir, f"{x} {y} {button}.jpeg")
                cv2.imwrite(filename, eyes)
                print(f"保存图像: {filename}")

    def start_collection(self):
        print("开始数据采集...")
        print("请注视屏幕上的目标位置并点击鼠标进行采集")
        with Listener(on_click=self.on_click) as listener:
            listener.join()

if __name__ == "__main__":
    collector = DataCollector()
    collector.start_collection()