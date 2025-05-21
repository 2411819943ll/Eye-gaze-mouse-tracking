import torch
import numpy as np
import cv2
import pyautogui
from train import ResNeXt18
from data_collection import DataCollector

class EyeTracker:
    def __init__(self, model_path="eye_tracking_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNeXt18().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.collector = DataCollector()
        self.width = self.collector.width
        self.height = self.collector.height

    def track(self):
        print("开始眼动追踪...")
        while True:
            eyes = self.collector.scan()
            if eyes is not None:
                eyes = np.expand_dims(eyes / 255.0, axis=0)
                eyes = torch.FloatTensor(eyes).permute(0, 3, 1, 2).to(self.device)
                
                with torch.no_grad():
                    x, y = self.model(eyes)[0].cpu().numpy()
                
                pyautogui.moveTo(x * self.width, y * self.height)

if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.track()