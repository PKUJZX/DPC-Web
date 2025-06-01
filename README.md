## DPC 图像重建项目

本项目提供了一个基于 Web 界面的工具，用于执行差分相衬 (DPC) 显微图像的重建。用户可以上传包含特定 TIFF 图像的文件夹，调整重建参数，并获得定性和定量的相位重建结果。

### 项目概述

差分相衬 (DPC) 显微技术是一种计算成像方法，通过捕获多张不同照明角度下的强度图像来恢复样本的相位信息。本项目实现了一个 DPC 重建算法，并提供了一个用户友好的 Web 界面来简化操作流程。

主要功能包括：

- **数据上传**: 用户通过选择包含 `lower.tif`, `up.tif`, `left.tif`, `right.tif` 四张原始 DPC 强度图像的文件夹来上传数据。
- **参数配置**: 用户可以调整波长、物镜放大倍数、数值孔径 (NA)、相机像素尺寸和正则化参数等。
- **图像重建**:
  - **定性重建**: 快速生成上下 (UD) 和左右 (LR) 方向的差分图像，用于初步评估。
  - **定量重建**:
    - 基于上下两张图像 (UD) 的相位重建。
    - 基于左右两张图像 (LR) 的相位重建。
    - 基于全部四张图像 (UDLR) 的完整相位重建。
- **结果展示**: 在网页上直接显示重建后的 PNG 图像，并提供对应 `.npy` 格式数据文件的下载链接。
  
![image](https://github.com/user-attachments/assets/cac14868-21f4-4d95-8d4f-717663eab9a2)
### 项目结构

```
.
├── data/                     # (自动创建) 存储上传的 TIFF 文件转换后的 .mat 文件
│   └── [experiment_name].mat
├── outputs/                  # (自动创建) 存储重建结果
│   └── [experiment_name]/
│       ├── UD_qualitative.png
│       ├── LR_qualitative.png
│       ├── UDLR.png
│       ├── UDLR.npy
│       ├── UD.png
│       ├── UD.npy
│       ├── LR.png
│       └── LR.npy
├── app.py                    # Flask 后端应用，处理文件上传、参数接收和重建逻辑调用
├── dpc_algorithm.py          # DPC 重建核心算法实现
├── UI.html                   # 前端用户界面
└── README.md                 # 本文档
```

### 技术栈与依赖

- **后端**: Python, Flask
- **前端**: HTML, Tailwind CSS, JavaScript
- **核心算法**: NumPy, SciPy, Pillow (PIL), Matplotlib

#### 依赖库

确保已安装以下 Python 库：

- `Flask`
- `Flask-CORS`
- `numpy`
- `scipy`
- `Pillow`
- `matplotlib`

可以使用 pip 安装所有依赖：

```
pip install Flask Flask-CORS numpy scipy Pillow matplotlib
```

### 安装与运行

1. **克隆/下载项目**: 将项目文件下载到本地。

2. **安装依赖**: 按照上一节的说明安装所有必需的 Python 库。

3. **运行 Flask 应用**: 在项目根目录下，打开终端并执行以下命令启动后端服务器：

   ```
   python app.py
   ```

   默认情况下，服务器将在 `http://127.0.0.1:5001` 上运行。

4. **访问用户界面**: 打开您的网络浏览器，访问 `UI.html` 文件。可以直接在文件系统中双击打开它，或者如果需要通过 HTTP 服务器访问（例如，为了避免某些浏览器的安全限制），可以将其放置在任何 HTTP 服务器的文档根目录下。 **注意**: `UI.html` 中的 JavaScript 默认会尝试连接到 `http://127.0.0.1:5001` 上的 Flask 服务器。如果您的 Flask 服务器运行在不同的地址或端口，请相应修改 `UI.html` 文件中的 `flaskServerUrl` JavaScript 变量。

### 使用说明

1. **准备数据文件夹**:
   - 创建一个文件夹，例如 `my_sample_data`。
   - 将四张必需的 TIFF 图像放入此文件夹中，并确保它们的文件名分别为：
     - `lower.tif` (下半圆照明)
     - `up.tif` (上半圆照明)
     - `left.tif` (左半圆照明)
     - `right.tif` (右半圆照明)
2. **打开 Web 界面**: 在浏览器中打开 `UI.html`。
3. **选择数据文件夹**:
   - 点击 "选择数据文件夹" 按钮。
   - 在文件对话框中，选择您在步骤1中准备的文件夹 (例如 `my_sample_data`)。
   - 文件夹名称（例如 `my_sample_data`）将被用作本次实验的名称 (`dir_name`)，并在服务器端用于组织数据和输出文件。
4. **配置参数 (可选)**:
   - 展开 "可选参数" 部分。
   - 根据需要修改以下参数的默认值：
     - **波长 (μm)**: 光源波长。
     - **物镜放大倍数**: 显微镜物镜的放大倍数。
     - **数值孔径 (NA)**: 物镜的数值孔径。
     - **相机像素尺寸 (μm)**: 相机传感器的单位像素物理尺寸。
     - **正则化参数 p**: 用于 Tikhonov 正则化的参数。
5. **开始重建**:
   - 点击 "开始重建" 按钮。
   - 按钮文本将变为 "处理中..."，并显示一个加载动画。
6. **查看结果**:
   - 重建完成后，状态消息将更新。
   - "重建结果" 部分将显示生成的图像。
   - 每张图像下方会显示其文件名，并提供 `.npy` 文件的下载链接。
   - 所有输出文件将保存在服务器上的 `outputs/[experiment_name]/` 目录中。

#### 输出文件说明

对于每个实验 (由上传的文件夹名称定义)，将在服务器的 `outputs` 目录下创建一个子目录。例如，如果上传的文件夹名为 `sample1`，则输出将在 `outputs/sample1/` 中。

- **`UD_qualitative.png`**: 上下图像的定性差分结果。
- **`LR_qualitative.png`**: 左右图像的定性差分结果。
- **`UDLR.png`**: 使用全部四张图像进行定量重建得到的相位图 (PNG格式)。
- **`UDLR.npy`**: 使用全部四张图像进行定量重建得到的相位数据 (NumPy数组格式)。
- **`UD.png`**: 仅使用上下两张图像进行定量重建得到的相位图 (PNG格式)。
- **`UD.npy`**: 仅使用上下两张图像进行定量重建得到的相位数据 (NumPy数组格式)。
- **`LR.png`**: 仅使用左右两张图像进行定量重建得到的相位图 (PNG格式)。
- **`LR.npy`**: 仅使用左右两张图像进行定量重建得到的相位数据 (NumPy数组格式)。

### 后端 API

后端 Flask 应用提供了一个主要的 API 端点：

- **`POST /reconstruct`**:

  - 接收 `multipart/form-data`。

  - **表单字段**:

    - `dir_name` (string, 必需): 实验名称，通常是上传的文件夹名称。
    - `lower_tif` (file, 必需): `lower.tif` 文件。
    - `up_tif` (file, 必需): `up.tif` 文件。
    - `left_tif` (file, 必需): `left.tif` 文件。
    - `right_tif` (file, 必需): `right.tif` 文件。
    - `wavelength` (float, 可选): 波长。
    - `mag` (float, 可选): 物镜放大倍数。
    - `na` (float, 可选): 数值孔径。
    - `pixel_size_cam` (float, 可选): 相机像素尺寸。
    - `reg_p` (float, 可选): 正则化参数 p。

  - **成功响应 (200 OK)**:

    ```
    {
        "message": "重建成功",
        "output_directory_relative": "your_dir_name",
        "generated_files": {
            "qualitative_UD_png": "UD_qualitative.png",
            "qualitative_LR_png": "LR_qualitative.png",
            "full_UDLR_npy": "UDLR.npy",
            "full_UDLR_png": "UDLR.png",
            "UD_npy": "UD.npy",
            "UD_png": "UD.png",
            "LR_npy": "LR.npy",
            "LR_png": "LR.png"
        }
    }
    ```

  - **错误响应 (4xx, 5xx)**:

    ```
    {
        "error": "错误描述信息"
    }
    ```

- **`GET /outputs/<path:dir_name>/<path:filename>`**:

  - 用于从 `outputs` 目录提供生成的静态文件（图像和 .npy 数据）。
