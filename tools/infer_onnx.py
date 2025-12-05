import onnxruntime as ort
import numpy as np
import cv2
import time

ONNX_PATH = "/home/paco/work_prjs/open_prjs/OpenStereo/output/LightStereo-S-KITTI.onnx"
# left_img_path = "/home/paco/work_prjs/open_prjs/OpenStereo/data/HjzlData/361530971.log/images/0/138288560012.png"
# right_img_path = "/home/paco/work_prjs/open_prjs/OpenStereo/data/HjzlData/361530971.log/images/1/138288560012.png"

left_img_path = "/home/paco/work_prjs/open_prjs/OpenStereo/data/KITTI15/training/image_2/000012_10.png"
right_img_path = "/home/paco/work_prjs/open_prjs/OpenStereo/data/KITTI15/training/image_3/000012_10.png"


save_directory = "/home/paco/work_prjs/open_prjs/OpenStereo"

# 相机参数
BASELINE = 0.08
FOCAL = 700

# input img demand
H, W = 256, 512


def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (W, H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # normalize
    img = img.transpose(2, 0, 1)  # CHW
    return img[np.newaxis, :]  # NCHW


def main(Provider_Name):
    # ----------------------------------------------
    # 1. 创建 ORT Session（TensorRT EP 优先）
    # ----------------------------------------------
    providers = []
    if Provider_Name == "tensorrt":
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "../output/trt_cache/",  # Engine 缓存目录, 启动时ORT会自动将ONNX转为TRT engine, 不需要你手动构建。
                    "trt_fp16_enable": True,  # 开启 FP16, 如果显卡不支持 FP16，它会自动 fallback 到 FP32。
                    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2GB
                }
            )
        ]
    elif Provider_Name == "cuda":
        providers = [
            "CUDAExecutionProvider",
        ]
    elif Provider_Name == "cpu":
        providers = [
            "CPUExecutionProvider"
        ]

    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    print("Using EP:", session.get_providers())

    # 获取输入名
    in0 = session.get_inputs()[0].name
    in1 = session.get_inputs()[1].name

    # ----------------------------------------------
    # 2. 图像预处理
    # ----------------------------------------------
    left = preprocess(left_img_path)
    right = preprocess(right_img_path)

    # ----------------------------------------------
    # 3. 推理
    # ----------------------------------------------
    inputs = {in0: left, in1: right}

    outputs = session.run(None, inputs)

    disp = outputs[0][0, 0]  # [H,W]

    # ----------------------------------------------
    # 4. 深度计算
    # ----------------------------------------------
    depth = FOCAL * BASELINE / (disp + 1e-6)

    # ----------------------------------------------
    # 5. 保存结果
    # ----------------------------------------------
    disp_save_name = save_directory + f"/ort_{Provider_Name}_disp.png"
    depth_save_name = save_directory + f"/ort_{Provider_Name}_depth.png"
    cv2.imwrite(disp_save_name, (disp / disp.max() * 255).astype(np.uint8))
    cv2.imwrite(depth_save_name, (depth / depth.max() * 255).astype(np.uint8))

    print(f"Saved: {disp_save_name}, {depth_save_name}")


if __name__ == '__main__':
    Provider_Name = "cuda"
    # Provider_Name = "cpu"
    # Provider_Name = "tensorrt"

    infer_start = time.time()
    main("cpu")
    time1 = time.time()
    print(f" cpu infer time ms = {(time1 - infer_start) * 1000}")

    main("cuda")
    time2 = time.time()
    print(f" cuda infer time ms = {(time2 - time1) * 1000}")

    main("tensorrt")
    time3 = time.time()
    print(f" tensorrt infer time ms = {(time3 - time2) * 1000}")
