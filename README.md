## 使用说明
1. 数据采集
   
   ```
   python data_collection.py
   ```
   - 运行后，注视屏幕上的目标位置并点击鼠标
   - 每次点击都会保存当前的眼睛图像和鼠标位置
2. 模型训练
   
   ```
   python train.py
   ```
   - 使用采集的数据集训练模型
   - 训练完成后会保存模型权重
3. 眼动追踪
   
   ```
   python predict.py
   ```
   - 加载训练好的模型
   - 实时追踪眼动并控制鼠标移动
## 依赖库
- OpenCV
- PyTorch
- NumPy
- pyautogui
- pynput
## 安装依赖
```
pip install opencv-python torch numpy pyautogui pynput
```
## 注意事项
1. 确保摄像头正常工作且能够清晰捕获眼睛图像
2. 数据采集时保持头部相对稳定
3. 建议采集足够多的样本以提高模型准确性
4. 使用时保持适当的光照条件
## 许可证
MIT License
