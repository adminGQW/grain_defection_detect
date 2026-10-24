# 晶粒缺陷可视化识别 UI

本地浏览器单图四分类页面。使用项目已有的 `repvit_v4 / seed_2026` 权重；显示整图类别和四类 Softmax 分数，不定位缺陷区域。

## 启动

双击 `启动识别界面.bat` 即可启动并打开浏览器。BAT 调用同目录 `start_ui.ps1`，使用已配置的 Conda 环境；如果同一权重的服务已在运行，会打开现有页面。若端口上是旧权重的服务，会提示先停止旧服务，避免误用旧模型。

启动后保持终端窗口开启，按 `Ctrl+C` 停止。失败时窗口会暂停，保留错误信息。BAT 采用 ASCII 内容和 Windows CRLF 换行，避免中文 UTF-8 注释在 CMD 的 GBK 代码页下被拆成错误命令；PowerShell 脚本采用 UTF-8 BOM，供 Windows PowerShell 正确读取中文注释。编辑保存时请保持这些格式。

环境路径在 BAT 的 `GRAIN_UI_PYTHON` 中修改；服务端口在 `app.py` 的 PORT 与 `start_ui.ps1` 的 URL 中同步修改。

也可以手动启动：

在 PowerShell 中运行：

```powershell
conda activate D:\ANACONDA\envs\grain-defect-py311
Set-Location 'E:\Codex\科研\wafer_defection_detect\ZIP\晶粒缺陷检测代码\晶粒缺陷检测代码\detect_ui'
python app.py
```

然后打开 <http://127.0.0.1:8765>。按 `Ctrl+C` 停止服务。

需要检查模型能否加载和完成一次推理时，在服务停止后运行 `python app.py --smoke-test`。这只检查调用链，不是准确率测试。

不需要重新训练。权重采用相对于 `app.py` 的路径：`model_train_list/best_model_original.pth`。当前目录只有这一份模型；没有按文件排序自动挑选，避免多放一个文件就意外换模型。

实现为 `CHECKPOINT_PATH = UI_DIR / "model_train_list" / "best_model_original.pth"`，所以无论从哪个终端目录启动，都从 UI 所在目录取权重。更换模型文件名时同步修改 app.py 和 start_ui.ps1 的 expectedCheckpoint，并停止旧服务后重启。骨干与输入尺寸仍必须匹配训练配置。

界面复用训练代码的 224×224 Resize、RGB、ImageNet Normalize 预处理。页面只监听本机回环地址，地址为 `http://127.0.0.1:8765`。
