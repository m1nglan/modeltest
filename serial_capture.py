"""
serial_capture.py
=================
从串口捕获 ESP32 回传的 raw RGB 图片数据，保存为 PNG 文件。

用法：
  1. 在 classifier_driver.cpp 中设置 #define DUMP_IMG 1
  2. 重新编译烧录: idf.py build flash monitor
  3. 在另一个终端运行本脚本:
     python serial_capture.py --port COM3 --output esp32_raw.png
  4. 同时在 PC 上生成对比图:
     python serial_capture.py --compare

输出文件:
  - esp32_raw.png          — ESP32 JPEG 解码后的原始 640x480 图片
  - pc_cropped.png         — PC 端用相同参数 resize+crop 后的 224x224 图片
"""

import argparse
import serial
import numpy as np
from PIL import Image
import os
import sys
import time

# PC 预处理参数（与 verify_model.py 一致）
TARGET_SHORT = 256
TARGET_SIZE = 224
IMG_DIR = "main"


def capture_raw_image(port, baud=115200, timeout=30):
    """从串口捕获 ESP32 回传的 raw 图片"""
    ser = serial.Serial(port, baud, timeout=5)
    print(f"Connected to {port} at {baud} baud")

    # 等待 !IMG_BEGIN 标记
    start_time = time.time()
    header = b""
    while True:
        c = ser.read(1)
        if not c:
            if time.time() - start_time > timeout:
                print("Timeout waiting for image data")
                ser.close()
                return None, None, None
            continue
        header += c
        if header.endswith(b"!IMG_BEGIN "):
            break
        if len(header) > 100:
            header = header[-50:]  # keep last 50 bytes

    # 读取宽度和高度
    dims = b""
    while True:
        c = ser.read(1)
        if c == b'\n':
            break
        dims += c
    parts = dims.decode().strip().split()
    width, height = int(parts[0]), int(parts[1])
    print(f"Receiving image: {width}x{height}")

    # 读取原始二进制数据
    total_bytes = width * height * 3
    raw_data = bytearray()
    while len(raw_data) < total_bytes:
        chunk = ser.read(min(4096, total_bytes - len(raw_data)))
        if not chunk:
            if time.time() - start_time > timeout:
                print(f"Timeout: received {len(raw_data)}/{total_bytes} bytes")
                break
            continue
        raw_data.extend(chunk)

    ser.close()
    print(f"Received {len(raw_data)} bytes")

    if len(raw_data) != total_bytes:
        print(f"Warning: expected {total_bytes} bytes, got {len(raw_data)}")
        return None, None, None

    return np.frombuffer(raw_data, dtype=np.uint8), width, height


def save_as_png(data, width, height, output_path):
    """保存 raw RGB 数据为 PNG 图片"""
    arr = data.reshape(height, width, 3)
    img = Image.fromarray(arr, 'RGB')
    img.save(output_path)
    print(f"Saved {output_path} ({width}x{height})")
    return img


def pc_preprocess(image_path):
    """PC 端 resize+crop（与 verify_model.py 一致）"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    print(f"  PC source: {w}x{h}")

    # resize 短边到 256
    if w < h:
        new_w, new_h = TARGET_SHORT, int(h * TARGET_SHORT / w)
    else:
        new_w, new_h = int(w * TARGET_SHORT / h), TARGET_SHORT
    img = img.resize((new_w, new_h), Image.BILINEAR)
    print(f"  PC resized: {new_w}x{new_h}")

    # center crop 224
    left = (new_w - TARGET_SIZE) // 2
    top = (new_h - TARGET_SIZE) // 2
    img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))
    print(f"  PC cropped: ({left},{top}) -> {TARGET_SIZE}x{TARGET_SIZE}")
    return img


def main():
    parser = argparse.ArgumentParser(description="Capture ESP32 raw image from serial")
    parser.add_argument("--port", "-p", default="COM3", help="Serial port")
    parser.add_argument("--baud", "-b", type=int, default=115200, help="Baud rate")
    parser.add_argument("--output", "-o", default="esp32_raw.png", help="Output PNG file")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Also generate PC comparison images")
    args = parser.parse_args()

    # 1. 捕获 ESP32 回传的图片
    print("=" * 50)
    print("Step 1: Capture ESP32 raw image from serial")
    print("=" * 50)
    print("Make sure ESP32 is running with DUMP_IMG=1 and monitor is NOT running.")
    print("Press any key to start capturing...")
    input()

    data, width, height = capture_raw_image(args.port, args.baud)
    if data is None:
        print("Failed to capture image")
        return

    esp_img = save_as_png(data, width, height, args.output)

    # 2. 生成 PC 对比图
    if args.compare:
        print("\n" + "=" * 50)
        print("Step 2: Generate PC comparison images")
        print("=" * 50)

        # PC 原图
        t1_path = os.path.join(IMG_DIR, "t1.jpg")
        if os.path.exists(t1_path):
            pc_cropped = pc_preprocess(t1_path)
            pc_cropped.save("pc_cropped.png")
            print("Saved pc_cropped.png")

            # 如果 ESP32 图片也是 640x480，也 crop 一份对比
            if width == 640 and height == 480:
                # 用相同参数在 ESP32 图片上 crop
                esp_resized = esp_img.resize((341, 256), Image.BILINEAR)
                esp_cropped = esp_resized.crop((58, 16, 58 + 224, 16 + 224))
                esp_cropped.save("esp32_cropped.png")
                print("Saved esp32_cropped.png")
        else:
            print(f"  {t1_path} not found, skipping PC comparison")

        print("\nGenerated files:")
        print(f"  {args.output}         - ESP32 raw decoded image ({width}x{height})")
        if os.path.exists("pc_cropped.png"):
            print("  pc_cropped.png      - PC PIL resize+crop (224x224)")
            print("  esp32_cropped.png   - ESP32 image with same resize+crop (224x224)")
        print("\nOpen both images to visually compare.")

    print("\nDone.")


if __name__ == "__main__":
    main()
