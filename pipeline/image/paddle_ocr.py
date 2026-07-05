#!/usr/bin/env python
"""
Paddle OCR 脚本 - 接收图片列表，返回识别结果
用法: python paddle_ocr.py <image1> <image2> ... 
或者: python paddle_ocr.py --json <image1> <image2> ...  (返回JSON格式)

环境变量:
  CUDA_VISIBLE_DEVICES - 指定使用的GPU，例如: CUDA_VISIBLE_DEVICES=7 python paddle_ocr.py ...
"""
import sys
import json
import os
from paddleocr import TextRecognition

def main():
    if len(sys.argv) < 2:
        print("用法: python paddle_ocr.py <image1> <image2> ...", file=sys.stderr)
        sys.exit(1)
    
    # 检查是否是 JSON 输出模式
    json_output = "--json" in sys.argv
    if json_output:
        sys.argv.remove("--json")
    
    # 获取图片路径列表
    image_paths = sys.argv[1:]
    
    # 初始化模型
    # PaddleOCR 通过环境变量 CUDA_VISIBLE_DEVICES 指定使用哪个 GPU
    # 例如: CUDA_VISIBLE_DEVICES=7 python paddle_ocr.py ...
    # 这里固定使用 "gpu"，具体使用哪个 GPU 由调用者通过环境变量控制
    model = TextRecognition(device="gpu")
    
    # 批量识别
    try:
        output = model.predict(input=image_paths)
        
        # 提取 rec_text
        results = []
        for res in output:
            rec_text = res.get('rec_text', '')
            results.append(rec_text)
        
        # 输出结果
        if json_output:
            # JSON 格式输出
            print(json.dumps(results, ensure_ascii=False))
        else:
            # 每行一个结果
            for text in results:
                print(text)
        
        sys.exit(0)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
    