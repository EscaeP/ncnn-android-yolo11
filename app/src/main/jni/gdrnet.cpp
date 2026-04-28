// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2026 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "gdrnet.h"

#include <android/log.h>
#include <android/asset_manager.h>

#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NCNN_DEBUG", __VA_ARGS__)

const int INPUT_RES = 256;

GDRNet::GDRNet()
{
}

GDRNet::~GDRNet()
{
    gdrnet.clear();
}

void GDRNet::setCameraParams(float fx, float fy, float cx, float cy)
{
    default_camera_params.fx = fx;
    default_camera_params.fy = fy;
    default_camera_params.cx = cx;
    default_camera_params.cy = cy;

    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Camera params set to - fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f",
                        fx, fy, cx, cy);
}

int GDRNet::load(const char* parampath, const char* modelpath, bool use_gpu)
{
    gdrnet.clear();

    ncnn::Option opt;
    opt.use_packing_layout = false; // 必须在这里设置
    opt.use_fp16_storage = false;
    opt.use_vulkan_compute = use_gpu;
    gdrnet.opt = opt;

    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Loading model: %s and %s", parampath, modelpath);

    int param_result = gdrnet.load_param(parampath);
    if (param_result != 0)
    {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Failed to load param file: %s (error: %d)", parampath, param_result);
        return param_result;
    }

    int model_result = gdrnet.load_model(modelpath);
    if (model_result != 0)
    {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Failed to load model file: %s (error: %d)", modelpath, model_result);
        return model_result;
    }

    return 0;
}

int GDRNet::load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu)
{
    gdrnet.clear();

    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_storage = false;
    opt.use_vulkan_compute = use_gpu;
    gdrnet.opt = opt;

    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Loading model from assets: %s and %s", parampath, modelpath);

    if (!mgr) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "AAssetManager is null!");
        return -1;
    }

    int param_result = gdrnet.load_param(mgr, parampath);
    int model_result = gdrnet.load_model(mgr, modelpath);

    if (param_result != 0 || model_result != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Failed to load model file (param: %d, model: %d)", param_result, model_result);
        return -1;
    }

    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Model loaded successfully!");
    return 0;
}

int GDRNet::inference(const cv::Mat& full_img, const cv::Rect& bbox, int object_label, PoseResult& result)
{
    // ===============================
    // 1. 预处理：生成 ROI (对齐 demo.py)
    // ===============================
    float cx = bbox.x + bbox.width / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;
    // demo.py: scale = max(w, h) * 1.5
    float scale = std::max((float)bbox.width, (float)bbox.height) * 1.5f;
    float resize_ratio = (float)INPUT_RES / scale;

    // 仿射变换矩阵：将 ROI 区域映射到 256x256
    cv::Matx23f M;
    M(0, 0) = resize_ratio; M(0, 1) = 0;            M(0, 2) = INPUT_RES / 2.0f - cx * resize_ratio;
    M(1, 0) = 0;            M(1, 1) = resize_ratio; M(1, 2) = INPUT_RES / 2.0f - cy * resize_ratio;

    cv::Mat resized;
    cv::warpAffine(full_img, resized, M, cv::Size(INPUT_RES, INPUT_RES), cv::INTER_LINEAR);

    // [输入1] in0: 图像归一化与 Mean/Std
    ncnn::Mat in0 = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, INPUT_RES, INPUT_RES);
    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {1.0f/57.375f, 1.0f/57.12f, 1.0f/58.395f};
    in0.substract_mean_normalize(mean_vals, norm_vals);

    // [输入2] in1: 2D 坐标网格 [64, 64, 2]
    ncnn::Mat in1(64, 64, 2);

    for (int y = 0; y < 64; y++) {
        // Python np.linspace(0, 1, 64) 对应的步长
        float norm_y = (float)y / 63.0f;

        float* ptr_x = in1.channel(0).row(y);
        float* ptr_y = in1.channel(1).row(y);

        for (int x = 0; x < 64; x++) {
            float norm_x = (float)x / 63.0f;
            ptr_x[x] = norm_x;
            ptr_y[x] = norm_y;
        }
    }

    // ===============================
    // 2. 准备物理尺寸 extents (替换掉原先复杂的其余5个输入)
    // ===============================
    // [输入3] in2: 3D bounding box 的物理尺寸
    ncnn::Mat in2(3);
    float scale_factor = 0.005f; // 单位转换
    switch (object_label) {
        case 1:  in2[0] = 75.9f * scale_factor; in2[1] = 77.6f * scale_factor; in2[2] = 91.8f * scale_factor; break;
        case 5:  in2[0] = 100.8f * scale_factor; in2[1] = 181.8f * scale_factor; in2[2] = 193.7f * scale_factor; break;
        case 6:  in2[0] = 67.0f * scale_factor; in2[1] = 127.6f * scale_factor; in2[2] = 117.5f * scale_factor; break;
        case 8:  in2[0] = 229.5f * scale_factor; in2[1] = 75.5f * scale_factor; in2[2] = 208.0f * scale_factor; break;
        case 9:  in2[0] = 104.4f * scale_factor; in2[1] = 77.4f * scale_factor; in2[2] = 85.7f * scale_factor; break;
        case 10: in2[0] = 150.2f * scale_factor; in2[1] = 107.1f * scale_factor; in2[2] = 69.2f * scale_factor; break;
        case 11: in2[0] = 36.7f * scale_factor; in2[1] = 77.9f * scale_factor; in2[2] = 172.8f * scale_factor; break;
        case 12: in2[0] = 100.9f * scale_factor; in2[1] = 108.5f * scale_factor; in2[2] = 90.8f * scale_factor; break;
        default: in2.fill(0.f); break;
    }

    // ===============================
    // 3. 执行 NCNN 推理
    // ===============================
    
    // 打印非图片输入信息

    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Input in2 (extents) shape: w=%d, h=%d, c=%d", in2.w, in2.h, in2.c);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Input in2 (extents): [%.4f, %.4f, %.4f]", in2[0], in2[1], in2[2]);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Object label: %d", object_label);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "bbox: x=%d, y=%d, width=%d, height=%d", bbox.x, bbox.y, bbox.width, bbox.height);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "ROI center: (%.2f, %.2f), scale: %.2f, resize_ratio: %.4f", cx, cy, scale, resize_ratio);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Camera params: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", 
        default_camera_params.fx, default_camera_params.fy, default_camera_params.cx, default_camera_params.cy);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Object label: %d", resize_ratio);   

    ncnn::Extractor ex = gdrnet.create_extractor();

    // 输入节点名与生成的 .param 文件完全对齐
    ex.input("in0", in0);
    ex.input("in1", in1);
    ex.input("in2", in2);

    ncnn::Mat out0, out1;
    // out0 对应 [1, 6] 的 6D旋转，out1 对应 [1, 3] 的平移预测 (dx, dy, dz)
    int ret_r = ex.extract("out0", out0);
    int ret_t = ex.extract("out1", out1);

    if (ret_r != 0 || ret_t != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "extract failed! ret_r: %d, ret_t: %d", ret_r, ret_t);
        return -1;
    }

    // 打印out0和out1的原始输出值
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "out0 shape: w=%d, h=%d, c=%d", out0.w, out0.h, out0.c);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "out1 shape: w=%d, h=%d, c=%d", out1.w, out1.h, out1.c);
    
    float* out0_data = (float*)out0.data;
    float* out1_data = (float*)out1.data;
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "out0: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
        out0_data[0], out0_data[1], out0_data[2], out0_data[3], out0_data[4], out0_data[5]);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "out1: [%.4f, %.4f, %.4f]",
        out1_data[0], out1_data[1], out1_data[2]);

    // ===============================
    // 4. 后处理：手动接管数学解算 (关键脱敏步骤)
    // ===============================

    // [A] 解算 3D 平移坐标 (严格对齐 Python 修正版的 decode_centroid_z)
    float pred_dx = out1[0];
    float pred_dy = out1[1];
    float pred_z  = out1[2];


    // 2. 还原中心点偏移 (由于网络输出是在 256x256 下的，需要除以 ratio 还原)
    float pixel_x = cx + pred_dx / resize_ratio;
    float pixel_y = cy + pred_dy / resize_ratio;

    // 3. 还原绝对深度 (tz = dz * resize_ratio, 解决近大远小的透视问题)
    float tz = pred_z * resize_ratio;

    // 4. 利用相机内参进行反投影，解算出世界坐标 X, Y, Z
    result.translation[0] = (pixel_x - default_camera_params.cx) * tz / default_camera_params.fx;
    result.translation[1] = (pixel_y - default_camera_params.cy) * tz / default_camera_params.fy;
    result.translation[2] = tz;

    // [B] 6D 向量转 3x3 旋转矩阵 (包含 Allocentric 到 Egocentric 的转换)

    // 1. 获取网络预测的原始 Allocentric 旋转基
    cv::Vec3f x_raw(out0[0], out0[1], out0[2]);
    cv::Vec3f y_raw(out0[3], out0[4], out0[5]);

    cv::Vec3f x = cv::normalize(x_raw);
    cv::Vec3f z_raw = x.cross(y_raw);
    cv::Vec3f z = cv::normalize(z_raw);
    cv::Vec3f y = z.cross(x);

    // 组合成原始的 R_allo 矩阵
    cv::Matx33f R_allo(
            x[0], y[0], z[0],
            x[1], y[1], z[1],
            x[2], y[2], z[2]
    );

    // 2. 计算相机射线的旋转矩阵 R_ray
    // 利用前面已经解算出来的绝对平移坐标 tx, ty, tz
    float t_x = result.translation[0];
    float t_y = result.translation[1];
    float t_z = result.translation[2];

    float norm_t = std::sqrt(t_x * t_x + t_y * t_y + t_z * t_z);
    float proj_xz = std::sqrt(t_x * t_x + t_z * t_z);

    // 防止数学除 0 异常
    float c_y = (proj_xz == 0.0f) ? 1.0f : (t_z / proj_xz);
    float s_y = (proj_xz == 0.0f) ? 0.0f : (t_x / proj_xz);
    float c_x = (norm_t == 0.0f) ? 1.0f : (proj_xz / norm_t);
    float s_x = (norm_t == 0.0f) ? 0.0f : (-t_y / norm_t);

    cv::Matx33f R_ray(
            c_y,  s_y * s_x,  s_y * c_x,
            0.0f,       c_x,       -s_x,
            -s_y,  c_y * s_x,  c_y * c_x
    );

    // 3. 将其转换回以相机为绝对基准的 Egocentric 旋转
    cv::Matx33f R_ego = R_ray * R_allo;

    // 4. 将最终结果赋值给 PoseResult
    result.rotation[0] = R_ego(0, 0); result.rotation[1] = R_ego(0, 1); result.rotation[2] = R_ego(0, 2);
    result.rotation[3] = R_ego(1, 0); result.rotation[4] = R_ego(1, 1); result.rotation[5] = R_ego(1, 2);
    result.rotation[6] = R_ego(2, 0); result.rotation[7] = R_ego(2, 1); result.rotation[8] = R_ego(2, 2);

    // ===============================
    // 5. 日志打印以供调试检查
    // ===============================
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "R: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
                        result.rotation[0], result.rotation[1], result.rotation[2],
                        result.rotation[3], result.rotation[4], result.rotation[5],
                        result.rotation[6], result.rotation[7], result.rotation[8]);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "T: [%.2f, %.2f, %.2f]",
                        result.translation[0], result.translation[1], result.translation[2]);

    return 0;
}

// ---------------- 以下 3D 投影和绘制相关代码保留不动 ----------------
static inline bool project3DTo2D(const cv::Point3f& pt3d, const PoseResult& pose, const CameraParams& cam, cv::Point& pt2d) {
    // 1. points_in_world = np.matmul(R, points.T) + T.reshape((3, 1))
    float x_w = pose.rotation[0] * pt3d.x + pose.rotation[1] * pt3d.y + pose.rotation[2] * pt3d.z + pose.translation[0];
    float y_w = pose.rotation[3] * pt3d.x + pose.rotation[4] * pt3d.y + pose.rotation[5] * pt3d.z + pose.translation[1];
    float z_w = pose.rotation[6] * pt3d.x + pose.rotation[7] * pt3d.y + pose.rotation[8] * pt3d.z + pose.translation[2];

    // 2. points_in_camera = np.matmul(K, points_in_world)
    float x_c = cam.fx * x_w + 0.0f * y_w + cam.cx * z_w;
    float y_c = 0.0f * x_w + cam.fy * y_w + cam.cy * z_w;
    float z_c = 0.0f * x_w + 0.0f * y_w + 1.0f * z_w;

    // 防止除以0或物体跑到镜头背后
    if (z_c <= 1e-5) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet","z_c:%.2f",z_c);
        return false;
    }

    // 3. points_2D[0, :] = points_in_camera[0, :] / (points_in_camera[2, :] + 1e-15)
    pt2d.x = cvRound(x_c / z_c);
    pt2d.y = cvRound(y_c / z_c);

    return true;
}

int GDRNet::draw3DBox(cv::Mat& rgb, const PoseResult& pose, const CameraParams& camera_params, float size_x, float size_y, float size_z)
{
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet","绘制");
    float dx = size_x / 2.0f;
    float dy = size_y / 2.0f;
    float dz = size_z / 2.0f;

    std::vector<cv::Point3f> points_3d = {
            cv::Point3f( dx,  dy,  dz), // 0
            cv::Point3f(-dx,  dy,  dz), // 1
            cv::Point3f(-dx, -dy,  dz), // 2
            cv::Point3f( dx, -dy,  dz), // 3
            cv::Point3f( dx,  dy, -dz), // 4
            cv::Point3f(-dx,  dy, -dz), // 5
            cv::Point3f(-dx, -dy, -dz), // 6
            cv::Point3f( dx, -dy, -dz)  // 7
    };

    std::vector<cv::Point> qs(8);
    for (int i = 0; i < 8; ++i) {
        if (!project3DTo2D(points_3d[i], pose, camera_params, qs[i])) {
            __android_log_print(ANDROID_LOG_ERROR, "GDRNet","失败");
            return -1;
        }
    }

    int thickness = 2;
    cv::Scalar box_color(0, 255, 0);

    for (int k = 0; k < 4; ++k) {
        int i_bot = k + 4, j_bot = (k + 1) % 4 + 4;
        cv::line(rgb, qs[i_bot], qs[j_bot], box_color, thickness, cv::LINE_AA);

        int i_mid = k, j_mid = k + 4;
        cv::line(rgb, qs[i_mid], qs[j_mid], box_color, thickness, cv::LINE_AA);

        int i_top = k, j_top = (k + 1) % 4;
        cv::line(rgb, qs[i_top], qs[j_top], box_color, thickness, cv::LINE_AA);
    }

    return 0;
}

int GDRNet::draw3DAxes(cv::Mat& rgb, const PoseResult& pose, const CameraParams& camera_params, const cv::Rect& roi)
{
    float axis_length = 100.0f; // 100毫米长度

    std::vector<cv::Point3f> points_3d = {
            cv::Point3f(0.0f, 0.0f, 0.0f),
            cv::Point3f(axis_length, 0.0f, 0.0f),
            cv::Point3f(0.0f, axis_length, 0.0f),
            cv::Point3f(0.0f, 0.0f, axis_length)
    };

    std::vector<cv::Point> pts(4);
    for (int i = 0; i < 4; ++i) {
        if (!project3DTo2D(points_3d[i], pose, camera_params, pts[i])) return -1;
    }

    int thickness = 2;
    double tipLength = 0.2;

    cv::arrowedLine(rgb, pts[0], pts[1], cv::Scalar(0, 0, 255), thickness, cv::LINE_AA, 0, tipLength);
    cv::arrowedLine(rgb, pts[0], pts[2], cv::Scalar(0, 255, 0), thickness, cv::LINE_AA, 0, tipLength);
    cv::arrowedLine(rgb, pts[0], pts[3], cv::Scalar(255, 0, 0), thickness, cv::LINE_AA, 0, tipLength);

    return 0;
}
