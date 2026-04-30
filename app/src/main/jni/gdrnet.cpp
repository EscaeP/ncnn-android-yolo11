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

    // 检查param文件是否存在
    AAsset* param_asset = AAssetManager_open(mgr, parampath, AASSET_MODE_STREAMING);
    if (!param_asset) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Param file not found: %s", parampath);
        return -1;
    } else {
        off_t param_size = AAsset_getLength(param_asset);
        __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Param file found: %s, size: %ld bytes", parampath, param_size);
        AAsset_close(param_asset);
    }

    // 检查model文件是否存在
    AAsset* model_asset = AAssetManager_open(mgr, modelpath, AASSET_MODE_STREAMING);
    if (!model_asset) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Model file not found: %s", modelpath);
        return -1;
    } else {
        off_t model_size = AAsset_getLength(model_asset);
        __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Model file found: %s, size: %ld bytes", modelpath, model_size);
        AAsset_close(model_asset);
    }

    int param_result = gdrnet.load_param(mgr, parampath);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "load_param result: %d", param_result);

    int model_result = gdrnet.load_model(mgr, modelpath);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "load_model result: %d", model_result);

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
    // 1. 预处理：生成 ROI (与原代码保持一致)
    // ===============================
    float cx = bbox.x + bbox.width / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;
    float scale = std::max((float)bbox.width, (float)bbox.height) * 1.5f;
    float resize_ratio = (float)INPUT_RES / scale;

    // 打印输入基础信息
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_IN", "Full image: %dx%d", full_img.cols, full_img.rows);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_IN", "Bbox: x=%d, y=%d, w=%d, h=%d", bbox.x, bbox.y, bbox.width, bbox.height);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_IN", "Object label: %d", object_label);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_IN", "ROI center: (%.2f, %.2f), scale: %.2f, resize_ratio: %.4f", cx, cy, scale, resize_ratio);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_IN", "Camera params: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", 
        default_camera_params.fx, default_camera_params.fy, default_camera_params.cx, default_camera_params.cy);

    cv::Matx23f M;
    M(0, 0) = resize_ratio; M(0, 1) = 0;            M(0, 2) = INPUT_RES / 2.0f - cx * resize_ratio;
    M(1, 0) = 0;            M(1, 1) = resize_ratio; M(1, 2) = INPUT_RES / 2.0f - cy * resize_ratio;

    cv::Mat resized;
    cv::warpAffine(full_img, resized, M, cv::Size(INPUT_RES, INPUT_RES), cv::INTER_LINEAR);

    // [输入1] in0: 图像
    ncnn::Mat in0 = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, INPUT_RES, INPUT_RES);
    // 根据配置文件，仅仅将像素值缩放到 [0, 1] 即可
    const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    const float norm_vals[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
    in0.substract_mean_normalize(mean_vals, norm_vals);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_IN", "Input in0 (image): w=%d, h=%d, c=%d", in0.w, in0.h, in0.c);

    // [输入2] in1: 2D 坐标网格
    ncnn::Mat in1(64, 64, 2);
    for (int y = 0; y < 64; y++) {
        float norm_y = (float)y / 63.0f;
        float* ptr_x = in1.channel(0).row(y);
        float* ptr_y = in1.channel(1).row(y);
        for (int x = 0; x < 64; x++) {
            ptr_x[x] = (float)x / 63.0f;
            ptr_y[x] = norm_y;
        }
    }
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_IN", "Input in1 (coord grid): w=%d, h=%d, c=%d", in1.w, in1.h, in1.c);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_IN", "in1 sample: (0,0)=(%.4f,%.4f), (32,32)=(%.4f,%.4f), (63,63)=(%.4f,%.4f)",
        in1.channel(0).row(0)[0], in1.channel(1).row(0)[0],
        in1.channel(0).row(32)[32], in1.channel(1).row(32)[32],
        in1.channel(0).row(63)[63], in1.channel(1).row(63)[63]);

    // [输入3] in2: 3D bounding box 物理尺寸
    ncnn::Mat in2(3);
    float scale_factor = 0.005f;
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
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_IN", "Input in2 (extents): w=%d, h=%d, c=%d", in2.w, in2.h, in2.c);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_IN", "in2 values: [%.6f, %.6f, %.6f]", in2[0], in2[1], in2[2]);

    // ===============================
    // 2. 执行 NCNN 推理：使用动态索引提取 (无视幽灵字符)
    // ===============================
    ncnn::Extractor ex = gdrnet.create_extractor();
    ex.input("in0", in0);
    ex.input("in1", in1);
    ex.input("in2", in2);


    ncnn::Mat out0, out1;
    int ret_r = ex.extract("out0", out0);
    int ret_t = ex.extract("out1", out1);

    if (ret_r != 0 || ret_t != 0) {
        LOGE("extract by ID failed! ret_r: %d, ret_t: %d", ret_r, ret_t);
        return -1;
    }

    // ===============================
    // 3. 后处理：手动接管数学解算 (完全对齐 Python)
    // ===============================
    // 获取底层数据指针
    float* out0_data = (float*)out0.data;
    float* out1_data = (float*)out1.data;

    // 打印 6D 原始旋转向量
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_RAW",
                        "Raw Rot (6D): [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]",
                        out0_data[0], out0_data[1], out0_data[2],
                        out0_data[3], out0_data[4], out0_data[5]);

    // 打印 3D 原始平移向量
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet_RAW",
                        "Raw Trans (dx, dy, dz): [%.6f, %.6f, %.6f]",
                        out1_data[0], out1_data[1], out1_data[2]);

    float pred_dx = out1_data[0];
    float pred_dy = out1_data[1];
    float pred_z  = out1_data[2];

    // [A] 解算绝对平移 (Translation)
    float pixel_x = cx + pred_dx / resize_ratio;
    float pixel_y = cy + pred_dy / resize_ratio;
    float tz = pred_z * resize_ratio;

    result.translation[0] = (pixel_x - default_camera_params.cx) * tz / default_camera_params.fx;
    result.translation[1] = (pixel_y - default_camera_params.cy) * tz / default_camera_params.fy;
    result.translation[2] = tz;

    // [B] 解算旋转矩阵 (严格对齐 demo.py)
    cv::Vec3f x_raw(out0[0], out0[1], out0[2]);
    cv::Vec3f y_raw(out0[3], out0[4], out0[5]);

    // 1. 归一化 x
    float norm_x = std::max((float)cv::norm(x_raw), 1e-8f);
    cv::Vec3f x = x_raw / norm_x;

    // 2. 减去投影算 y，并归一化
    cv::Vec3f y_proj = y_raw - x * x.dot(y_raw);
    float norm_y = std::max((float)cv::norm(y_proj), 1e-8f);
    cv::Vec3f y = y_proj / norm_y;

    // 3. 叉乘算 z
    cv::Vec3f z = x.cross(y);

    // 4. 直接赋值给 PoseResult
    result.rotation[0] = x[0]; result.rotation[1] = y[0]; result.rotation[2] = z[0];
    result.rotation[3] = x[1]; result.rotation[4] = y[1]; result.rotation[5] = z[1];
    result.rotation[6] = x[2]; result.rotation[7] = y[2]; result.rotation[8] = z[2];

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
