import serial
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque


# Change COM port if needed
ser = serial.Serial('COM3', 115200, timeout=1)

plt.ion()

# Small temporal buffer (reduce blur)
FRAME_AVG = 3
frame_buffer = deque(maxlen=FRAME_AVG)

while True:
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        data = line.split(",")

        if len(data) == 768:

            img = np.array(list(map(float, data))).reshape(24, 32)

            # -------------------------
            # 1. Light Bad Pixel Removal
            # -------------------------
            median = cv2.medianBlur(img.astype(np.float32), 3)
            img = np.where(np.abs(img - median) > 4, median, img)

            # -------------------------
            # 2. Mild Temporal Averaging
            # -------------------------
            frame_buffer.append(img)
            avg_img = np.mean(frame_buffer, axis=0)

            # -------------------------
            # 3. Very Mild Bilateral Filter
            # -------------------------
            filtered = cv2.bilateralFilter(
                avg_img.astype(np.float32),
                d=3,
                sigmaColor=10,
                sigmaSpace=10
            )
            
            # Smart adaptive window
            # Focus window around hottest 20% region
            hot_mean = np.mean(filtered[filtered > np.percentile(filtered, 70)])

            min_temp = hot_mean - 4
            max_temp = hot_mean + 2
            windowed = np.clip(filtered, min_temp, max_temp)

            norm = (windowed - min_temp) / (max_temp - min_temp)
            norm = norm.astype(np.float32)

            # Edge-preserving smoothing
            guided = cv2.ximgproc.guidedFilter(
                guide=norm,
                src=norm,
                radius=4,
                eps=0.01
            )

            # Upscale
            upscale = cv2.resize(
                guided,
                (256,192),
                interpolation=cv2.INTER_CUBIC
            )

            upscale = (upscale * 255).astype(np.uint8)
           
            # -------------------------
            # Display
            # -------------------------
            plt.clf()
            plt.imshow(upscale, cmap='inferno')
            plt.title("Thermal Image (Optimized)")
            plt.colorbar()
            plt.pause(0.01)
            

    except Exception as e:
        print("Error:", e)