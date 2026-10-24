# 【修改地图】换权重改 CHECKPOINT_PATH；改端口改 PORT 并同步 start_ui.ps1；改中文类别名改 CLASS_LABELS。
# 输入尺寸在 load_runtime 的 build_transforms(img_size=224)；必须与训练设置一致。
# 详细离线操作见 ../readme/离线改代码指南.md。
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image, UnidentifiedImageError


# __file__ 是当前源码位置，因此这些路径不依赖终端当前目录；PROJECT_DIR 是工程根目录。
UI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = UI_DIR.parent
CODE_DIR = PROJECT_DIR / "code"
# 【更换模型权重】相对于本 app.py 定位，不依赖终端当前目录；改文件名后重启服务。
# 不要只改 model_name 欺骗加载：训练时若换了 REPVIT_MODEL_NAME，启动前也必须设置为相同值。
CHECKPOINT_PATH = UI_DIR / "model_train_list" / "best_model_original.pth"
# 32 MiB 上传体积限制；像素上限另由 MAX_IMAGE_PIXELS 控制。调大限制也会增加内存压力。
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
# 浏览器访问 http://127.0.0.1:8765；改端口时同步修改 start_ui.ps1 的 URL。
PORT = 8765

# 复用训练文件中的建模和预处理，避免两份模型逻辑不一致；训练文件末尾有 main 保护，不会因此训练。
sys.path.insert(0, str(CODE_DIR))
from benchmark_grain_4class_repvit_project_v3_eval_misclass_fixed import (  # noqa: E402
    build_model,
    build_transforms,
)


# 这里只改展示中文名；字典键必须匹配训练类别文件夹名称，不能靠改显示文字把旧模型变成新分类器。
CLASS_LABELS = {
    "normal": "正常",
    "pad_surface_mark_defect": "Pad 表面标记缺陷",
    "probe_mark_offset_defect": "探针标记偏移缺陷",
    "surface_contamination_defect": "表面污染缺陷",
}


# 启动时加载一次模型和权重，返回模型、类别顺序、预处理、设备和界面信息。
# 权重必须与 build_model 构造出的结构一致；类别数相同不代表类别含义相同，class_names 顺序必须沿用训练存档。
def load_runtime():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"找不到训练权重：{CHECKPOINT_PATH}")

    # checkpoint 是训练存档；只使用可信来源的权重文件。weights_only=True 限制加载内容类型。
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    class_names = checkpoint["class_names"]
    model_name = checkpoint["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    # 切到评估模式：Dropout 关闭，BatchNorm 使用已学习统计量；单图展示不应调用 model.train()。
    model.eval()
    # 【尺寸联动】这里固定为 224，存档未自动恢复训练 img_size；训练若改尺寸，本行要同步调整。
    _, image_transform = build_transforms(img_size=224)

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    info = {
        "model_name": model_name,
        "seed": checkpoint.get("seed"),
        "epoch": checkpoint.get("epoch"),
        "device": str(device),
        "device_name": device_name,
        "best_val_macro_f1": checkpoint.get("best_val_macro_f1"),
        "checkpoint": CHECKPOINT_PATH.name,
    }
    return model, class_names, image_transform, device, info


# 模块加载时立即读取权重，之后请求复用同一个模型；替换磁盘权重文件不会自动刷新内存模型。
MODEL, CLASS_NAMES, IMAGE_TRANSFORM, DEVICE, MODEL_INFO = load_runtime()


# 单图推理（用已训练的参数作判断，不更新参数）：图片字节→RGB→与训练一致的预处理→网络→各类得分。
# 只判断整张图所属类别，不返回缺陷位置；Softmax 得分是相对分数，不等于经过校准的真实正确概率。
def predict_image(image_bytes: bytes) -> dict:
    with Image.open(BytesIO(image_bytes)) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError("图片分辨率过大，请选择不超过 4000 万像素的图片。")
        image = source.convert("RGB")

    # transform 输出 [3,H,W]，unsqueeze(0) 在最前面增加批维度，得到 [1,3,H,W]；1 表示只有一张图。
    batch = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    # 推理模式不记录梯度；Softmax 将原始分数转成各类非负且总和约为 1 的得分。
    with torch.inference_mode():
        logits = MODEL(batch)
        scores = torch.softmax(logits[0], dim=0).cpu().tolist()

    # 选择分数最高的类别编号，再用权重中的 CLASS_NAMES 还原类别名；不能自行按中文名重新排序。
    predicted_index = max(range(len(scores)), key=scores.__getitem__)
    predicted_class = CLASS_NAMES[predicted_index]
    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_LABELS.get(predicted_class, predicted_class),
        "score": scores[predicted_index],
        "classes": [
            {
                "name": name,
                "label": CLASS_LABELS.get(name, name),
                "score": score,
            }
            for name, score in zip(CLASS_NAMES, scores)
        ],
    }


# HTTP 请求处理器（接收浏览器请求并发送结果的类）。
# 页面 GET /，模型信息 GET /api/info，识别 POST /api/predict；改接口名称需同时改 index.html。
class UIHandler(BaseHTTPRequestHandler):
    # 把结果编码成 UTF-8 JSON，并设置状态码和长度；ensure_ascii=False 让中文标签保持可读。
    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # 提供首页或模型信息；其他路径返回 404。
    # 不是任意目录文件服务器，新增页面需要明确添加对应路由。
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/info":
            self._send_json(200, MODEL_INFO)
            return
        if path in {"/", "/index.html"}:
            data = (UI_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        self._send_json(404, {"error": "页面不存在。"})

    # 只接收 /api/predict 的原始图片字节，检查请求体长度并返回预测或错误。
    # 前端 body 直接发送 File；如果改为 FormData（表单封装），后端也必须相应解析，不能直接 Image.open。
    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self._send_json(404, {"error": "接口不存在。"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "图片请求格式无效。"})
            return
        if length <= 0:
            self._send_json(400, {"error": "请先选择一张图片。"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(413, {"error": "图片文件过大，请选择小于 32 MB 的图片。"})
            return

        try:
            result = predict_image(self.rfile.read(length))
        except (UnidentifiedImageError, OSError):
            self._send_json(400, {"error": "无法读取该图片，请换一张常见格式的图片。"})
            return
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except Exception:
            self._send_json(500, {"error": "模型推理失败，请查看运行窗口中的错误信息。"})
            raise

        self._send_json(200, result)

    # 把请求日志打印到启动终端；排查访问与接口错误可以先看这里。
    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


# 直接运行时启动本机服务；Ctrl+C 停止。--smoke-test 是已有的单图检查入口，不是准确率测试。
# 启动前模块顶层已加载模型；仅做语法检查时不要 import 本模块，以免实际加载权重。
def main() -> None:
    if sys.argv[1:] == ["--smoke-test"]:
        sample_dir = PROJECT_DIR / "Task_data_30_10_60_aug" / "val" / CLASS_NAMES[0]
        sample_path = next((path for path in sample_dir.iterdir() if path.is_file()), None)
        if sample_path is None:
            raise FileNotFoundError(f"找不到单图自检样本：{sample_dir}")
        result = predict_image(sample_path.read_bytes())
        score_sum = sum(item["score"] for item in result["classes"])
        assert len(result["classes"]) == 4
        assert abs(score_sum - 1.0) < 1e-5
        print(f"Smoke check passed: classes=4, score_sum={score_sum:.6f}, device={DEVICE}")
        return

    # 仅绑定 127.0.0.1，本机可访问；这不是对外部署方案。端口已占用时先确认是否已有本服务。
    server = ThreadingHTTPServer(("127.0.0.1", PORT), UIHandler)
    print(f"模型：{MODEL_INFO['model_name']} / seed {MODEL_INFO['seed']} / epoch {MODEL_INFO['epoch']}")
    print(f"运行设备：{MODEL_INFO['device_name']} ({MODEL_INFO['device']})")
    print(f"可视化界面：http://127.0.0.1:{PORT}")
    print("按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("正在停止服务。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
