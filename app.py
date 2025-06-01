import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
import os
import time
import shutil # For cleaning up output directory if needed

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS # 导入 CORS

from dpc_algorithm import DPCSolver


app = Flask(__name__)
CORS(app) # 为整个应用启用 CORS

# --- 配置 ---
DATA_BASE_DIR = 'data' # 数据基础目录
OUTPUT_BASE_DIR = 'outputs' # 输出基础目录

# 确保基础目录存在
os.makedirs(DATA_BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def save_uploaded_tiffs_and_convert_to_mat(dir_name, data_base_dir, files):
    """
    保存上传的TIFF图像（lower, up, left, right）到 'data/<dir_name>/' 目录,
    然后将它们转换为 .mat 文件。
    .mat 文件保存在 data_base_dir 中，命名为 'dir_name.mat'。
    """
    tiff_input_path = os.path.join(data_base_dir, dir_name)
    os.makedirs(tiff_input_path, exist_ok=True) # 创建用于存放上传TIFF的子目录

    output_filename = dir_name + '.mat'
    
    required_files = {
        'lower_tif': 'lower.tif',
        'up_tif': 'up.tif',
        'left_tif': 'left.tif',
        'right_tif': 'right.tif'
    }
    
    saved_tif_paths = []

    for form_field_name, target_filename in required_files.items():
        if form_field_name not in files or files[form_field_name].filename == '':
            raise FileNotFoundError(f"请求中缺少必需的 TIFF 文件: '{form_field_name}' (应为 {target_filename})")
        
        file_storage = files[form_field_name]
        # 验证文件扩展名 (基本检查)
        if not file_storage.filename.lower().endswith(('.tif', '.tiff')):
            raise ValueError(f"文件 '{file_storage.filename}' (来自字段 '{form_field_name}') 不是有效的 TIFF 文件。")

        try:
            saved_path = os.path.join(tiff_input_path, target_filename)
            file_storage.save(saved_path)
            saved_tif_paths.append(saved_path)
            print(f"已保存上传的文件: {saved_path}")
        except Exception as e:
            raise RuntimeError(f"保存上传的 TIFF 文件 '{target_filename}' 时出错: {e}")

    if len(saved_tif_paths) != 4: # 双重检查，尽管上面的逻辑应该已经确保了
        raise ValueError(f"期望保存4个TIFF文件，但实际保存了 {len(saved_tif_paths)} 个")

    try:
        first_img = Image.open(saved_tif_paths[0])
        first_array = np.array(first_img)
        H, W = first_array.shape
        IDPC = np.zeros((4, H, W), dtype=np.uint16) # 匹配原始脚本的 dtype
        
        # 按特定顺序读取：lower, up, left, right
        # 这是基于它们在 required_files 中的顺序以及它们如何附加到 saved_tif_paths
        # 假设 files 是一个字典，并且我们按 keywords 的顺序迭代来填充 IDPC
        # 我们的 saved_tif_paths 是按照 lower, up, left, right 的顺序创建的
        
        # 我们需要确保 tif_files_paths 的顺序与 IDPC[i] 的期望顺序一致
        # 即 IDPC[0] = lower, IDPC[1] = up, IDPC[2] = left, IDPC[3] = right
        # 我们的 saved_tif_paths 已经按照这个顺序了。

        for i, file_path in enumerate(saved_tif_paths): # saved_tif_paths 的顺序已经是 lower, up, left, right
            img = Image.open(file_path)
            img_array = np.array(img, dtype=np.uint16) # 匹配 dtype
            IDPC[i, :, :] = img_array
            
        mat_file_path = os.path.join(data_base_dir, output_filename) # .mat 文件直接在 data_base_dir 中
        sio.savemat(mat_file_path, {'IDPC': IDPC})
        print(f".mat 文件已保存至 {mat_file_path}")
        return mat_file_path
    except Exception as e:
        raise RuntimeError(f"处理已保存的 TIFF 文件或保存 .mat 文件时出错: {e}")


def run_reconstruction_logic(dir_name, wavelength, mag, na, pixel_size_cam, reg_p, uploaded_files):
    """
    主 DPC 重建逻辑。
    现在它接收 uploaded_files 并调用 save_uploaded_tiffs_and_convert_to_mat。
    """
    
    # --- 本次运行的输出目录 ---
    current_output_dir_abs = os.path.join(OUTPUT_BASE_DIR, dir_name)
    os.makedirs(current_output_dir_abs, exist_ok=True)

    # --- 1. 保存上传的 TIFF 图像并转换为 .mat 文件 ---
    print(f"[{dir_name}] 正在保存上传的 TIFF 图像并转换为 .mat 文件...")
    try:
        # 注意：传递 'uploaded_files' (request.files) 给这个新函数
        mat_file_path = save_uploaded_tiffs_and_convert_to_mat(dir_name, DATA_BASE_DIR, uploaded_files)
    except Exception as e:
        print(f"[{dir_name}] 在 save_uploaded_tiffs_and_convert_to_mat 中出错: {e}")
        raise # 重新抛出以便端点捕获
    
    # --- 加载数据 ---
    # .mat 文件应位于 DATA_BASE_DIR (例如, data/my_sample.mat)
    data_mat_name = f"{dir_name}.mat" 
    full_mat_path = os.path.join(DATA_BASE_DIR, data_mat_name)

    if not os.path.exists(full_mat_path):
        raise FileNotFoundError(f"转换尝试后未找到 MAT 文件: {full_mat_path}")
        
    dpc_images = loadmat(full_mat_path)["IDPC"]  # 原始 DPC 图像 [4, H, W]

    # --- 定性重建 ---
    img1 = dpc_images[0, :, :].astype(np.float32)
    img2 = dpc_images[1, :, :].astype(np.float32)
    img3 = dpc_images[2, :, :].astype(np.float32)
    img4 = dpc_images[3, :, :].astype(np.float32)
    print(f'[{dir_name}] 开始定性重建...')
    start_time_qual = time.time()
    
    dpc_y_num = img1 - img2 
    dpc_x_num = img3 - img4 
    dpc_y_den = img1 + img2
    dpc_x_den = img3 + img4

    epsilon = 1e-6 * (dpc_y_den.max() if dpc_y_den.max() > 0 else 1) 
    dpc_y_signal = dpc_y_num / (dpc_y_den + epsilon)

    epsilon = 1e-6 * (dpc_x_den.max() if dpc_x_den.max() > 0 else 1) 
    dpc_x_signal = dpc_x_num / (dpc_x_den + epsilon)

    def normalize_to_uint8(array):
        min_val = array.min()
        max_val = array.max()
        if max_val == min_val:
            return np.full(array.shape, 128, dtype=np.uint8)
        normalized_array = ((array - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
        return normalized_array

    res1_qual = normalize_to_uint8(dpc_y_signal) 
    res2_qual = normalize_to_uint8(dpc_x_signal) 
    
    end_time_qual = time.time()
    print(f"[{dir_name}] 定性重建完成，耗时: {end_time_qual - start_time_qual:.4f} 秒")

    qual_ud_filename = "UD_qualitative.png"
    qual_lr_filename = "LR_qualitative.png"
    Image.fromarray(res1_qual).save(os.path.join(current_output_dir_abs, qual_ud_filename))
    Image.fromarray(res2_qual).save(os.path.join(current_output_dir_abs, qual_lr_filename))
    print(f"[{dir_name}] 定性重建结果已保存。")

    pixel_size = pixel_size_cam / mag

    results_paths = {
        "qualitative_UD_png": qual_ud_filename,
        "qualitative_LR_png": qual_lr_filename,
    }

    print(f"[{dir_name}] 开始进行4图像完整重建...")
    start_time_full = time.time()
    rotation_full = [90, 270, 0, 180] # 对应 IDPC[0]=lower(bottom, 90), IDPC[1]=up(top, 270), IDPC[2]=left(0), IDPC[3]=right(180)
    dpc_solver_full = DPCSolver(
        dpc_images, wavelength, na,
        na_in=0.0, pixel_size=pixel_size, rotation=rotation_full, dpc_num=4
    )
    dpc_solver_full.setRegularizationParameters(reg_u=1e-1, reg_p=reg_p)
    result_tv_full = dpc_solver_full.solve(method="Tikhonov")
    end_time_full = time.time()
    print(f"[{dir_name}] 4图像完整重建完成，耗时: {end_time_full - start_time_full:.4f} 秒")

    phase_tv_full = result_tv_full[0].imag
    full_npy_filename = 'UDLR.npy'
    full_png_filename = 'UDLR.png'
    np.save(os.path.join(current_output_dir_abs, full_npy_filename), phase_tv_full)

    height, width = phase_tv_full.shape[:2]
    dpi = 300
    fig_width = width / dpi
    fig_height = height / dpi
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(phase_tv_full, cmap='gray', interpolation='none')
    plt.savefig(os.path.join(current_output_dir_abs, full_png_filename),
                bbox_inches='tight',
                pad_inches=0,
                dpi=dpi)
    plt.close()

    print(f"[{dir_name}] 4图像完整重建结果已保存。")
    results_paths["full_UDLR_npy"] = full_npy_filename
    results_paths["full_UDLR_png"] = full_png_filename

    print(f"[{dir_name}] 开始进行 UD (上下) 2图像重建...")
    start_time_ud = time.time()
    rotation_first2 = [90, 270] # Lower, Up
    dpc_images_first2 = dpc_images[:2] # IDPC[0], IDPC[1]
    dpc_solver_first2 = DPCSolver(
        dpc_images_first2, wavelength, na,
        na_in=0.0, pixel_size=pixel_size, rotation=rotation_first2, dpc_num=2
    )
    dpc_solver_first2.setRegularizationParameters(reg_u=1e-1, reg_p=reg_p)
    result_tv_first2 = dpc_solver_first2.solve(method="Tikhonov")
    end_time_ud = time.time()
    print(f"[{dir_name}] UD 重建完成，耗时: {end_time_ud - start_time_ud:.4f} 秒")

    phase_tv_first2 = result_tv_first2[0].imag
    ud_npy_filename = 'UD.npy'
    ud_png_filename = 'UD.png'
    np.save(os.path.join(current_output_dir_abs, ud_npy_filename), phase_tv_first2)

    plt.figure(figsize=(fig_width, fig_height), dpi=dpi) # 使用与上面相同的尺寸逻辑
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(phase_tv_first2, cmap='gray', interpolation='none')
    plt.savefig(os.path.join(current_output_dir_abs, ud_png_filename),
                bbox_inches='tight',
                pad_inches=0,
                dpi=dpi)
    plt.close()
    print(f"[{dir_name}] UD 2图像重建结果已保存。")
    results_paths["UD_npy"] = ud_npy_filename
    results_paths["UD_png"] = ud_png_filename

    print(f"[{dir_name}] 开始进行 LR (左右) 2图像重建...")
    start_time_lr = time.time()
    rotation_last2 = [0, 180] # Left, Right
    dpc_images_last2 = dpc_images[2:] # IDPC[2], IDPC[3]
    dpc_solver_last2 = DPCSolver(
        dpc_images_last2, wavelength, na,
        na_in=0.0, pixel_size=pixel_size, rotation=rotation_last2, dpc_num=2
    )
    dpc_solver_last2.setRegularizationParameters(reg_u=1e-1, reg_p=reg_p)
    result_tv_last2 = dpc_solver_last2.solve(method="Tikhonov")
    end_time_lr = time.time()
    print(f"[{dir_name}] LR 重建完成，耗时: {end_time_lr - start_time_lr:.4f} 秒")

    phase_tv_last2 = result_tv_last2[0].imag
    lr_npy_filename = 'LR.npy'
    lr_png_filename = 'LR.png'
    np.save(os.path.join(current_output_dir_abs, lr_npy_filename), phase_tv_last2)
    
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi) # 使用与上面相同的尺寸逻辑
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(phase_tv_last2, cmap='gray', interpolation='none')
    plt.savefig(os.path.join(current_output_dir_abs, lr_png_filename),
                bbox_inches='tight',
                pad_inches=0,
                dpi=dpi)
    plt.close()

    print(f"[{dir_name}] LR 2图像重建结果已保存。")
    results_paths["LR_npy"] = lr_npy_filename
    results_paths["LR_png"] = lr_png_filename
    
    print(f"[{dir_name}] 所有重建任务已完成！")
    return results_paths


@app.route('/reconstruct', methods=['POST'])
def reconstruct_endpoint():
    """
    Flask 端点，用于触发 DPC 重建。
    期望 multipart/form-data 包含 'dir_name', TIFF 文件, 和可选的重建参数。
    """
    if 'dir_name' not in request.form:
        return jsonify({"error": "请求中缺少 'dir_name'"}), 400
    
    dir_name = request.form['dir_name']

    if not dir_name.strip(): # 检查是否为空或仅空格
        return jsonify({"error": "'dir_name' 不能为空"}), 400
    
    if ".." in dir_name or "/" in dir_name or "\\" in dir_name:
        return jsonify({"error": "无效的 'dir_name'。不能包含路径说明符。"}), 400

    # 检查 TIFF 文件是否已上传
    required_tiff_fields = ['lower_tif', 'up_tif', 'left_tif', 'right_tif']
    for field_name in required_tiff_fields:
        if field_name not in request.files or request.files[field_name].filename == '':
            return jsonify({"error": f"请求中缺少 TIFF 文件: {field_name}"}), 400
            
    uploaded_files = request.files # 获取所有上传的文件

    try:
        wavelength = float(request.form.get('wavelength', 0.465))
        mag = float(request.form.get('mag', 10.0))
        na = float(request.form.get('na', 0.30))
        pixel_size_cam = float(request.form.get('pixel_size_cam', 3.45))
        reg_p = float(request.form.get('reg_p', 5e-3))
    except ValueError as e:
        return jsonify({"error": f"参数转换错误: {e}. 请确保所有参数都是有效的数字。"}), 400

    try:
        print(f"收到针对 dir_name: {dir_name} 的重建请求")
        # 可选：清理此 dir_name 的先前结果
        output_dir_to_clean = os.path.join(OUTPUT_BASE_DIR, dir_name)
        if os.path.exists(output_dir_to_clean):
            # 为了安全，暂时禁用自动清理。如果需要，可以取消注释。
            # shutil.rmtree(output_dir_to_clean)
            # print(f"已清理先前的输出目录: {output_dir_to_clean}")
            pass # 如果目录存在，允许覆盖文件

        # 将 uploaded_files 传递给重建逻辑
        generated_files_relative = run_reconstruction_logic(
            dir_name, wavelength, mag, na, pixel_size_cam, reg_p, uploaded_files
        )
        return jsonify({
            "message": "重建成功",
            "output_directory_relative": dir_name, 
            "generated_files": generated_files_relative 
        }), 200
    except FileNotFoundError as e:
        app.logger.error(f"{dir_name} 发生 FileNotFoundError: {e}")
        return jsonify({"error": f"文件未找到: {e}"}), 404
    except ValueError as e: # 例如，文件类型错误或参数转换问题
        app.logger.error(f"{dir_name} 发生 ValueError: {e}")
        return jsonify({"error": f"无效输入或配置: {e}"}), 400
    except RuntimeError as e: # 例如，文件处理或保存错误
        app.logger.error(f"{dir_name} 发生 RuntimeError: {e}")
        return jsonify({"error": f"重建过程失败: {e}"}), 500
    except Exception as e:
        app.logger.error(f"{dir_name} 发生意外错误: {e}", exc_info=True)
        return jsonify({"error": f"发生意外错误: {e}"}), 500

@app.route('/outputs/<path:dir_name>/<path:filename>')
def serve_output_file(dir_name, filename):
    """提供 output 目录中的文件服务"""
    directory = os.path.join(app.root_path, OUTPUT_BASE_DIR, dir_name)
    # 对于 .npy 文件，我们希望浏览器总是下载它们
    if filename.lower().endswith('.npy'):
        return send_from_directory(directory, filename, as_attachment=True)
    return send_from_directory(directory, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
