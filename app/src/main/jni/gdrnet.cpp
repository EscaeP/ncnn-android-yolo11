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

    gdrnet.opt = ncnn::Option();

#if NCNN_VULKAN
    gdrnet.opt.use_vulkan_compute = use_gpu;
#endif

    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Loading model: %s and %s", parampath, modelpath);

    ncnn::Option opt;
    opt.use_packing_layout = false; // 必须在这里设置
    opt.use_fp16_storage = false;
    opt.use_vulkan_compute = false;
    gdrnet.opt = opt; // 先赋值 opt
    
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
    
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Load result - param: %d, model: %d", param_result, model_result);

    return 0;
}

int GDRNet::load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu)
{
    gdrnet.clear();

    gdrnet.opt = ncnn::Option();

#if NCNN_VULKAN
    gdrnet.opt.use_vulkan_compute = use_gpu;
#endif

    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Loading model from assets: %s and %s", parampath, modelpath);
    
    // Check if AAssetManager is valid
    if (!mgr) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "AAssetManager is null!");
        return -1;
    }
    
    // Try to open the param file to verify it exists
    AAsset* param_asset = AAssetManager_open(mgr, parampath, AASSET_MODE_STREAMING);
    if (!param_asset) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Failed to open param file: %s", parampath);
        return -1;
    } else {
        off_t param_size = AAsset_getLength(param_asset);
        __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Param file size: %ld bytes", param_size);
        if (param_size == 0) {
            __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Param file is empty: %s", parampath);
            AAsset_close(param_asset);
            return -1;
        }
        AAsset_close(param_asset);
    }
    
    // Try to open the model file to verify it exists
    AAsset* model_asset = AAssetManager_open(mgr, modelpath, AASSET_MODE_STREAMING);
    if (!model_asset) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Failed to open model file: %s", modelpath);
        return -1;
    } else {
        off_t model_size = AAsset_getLength(model_asset);
        __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Model file size: %ld bytes", model_size);
        if (model_size == 0) {
            __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Model file is empty: %s", modelpath);
            AAsset_close(model_asset);
            return -1;
        }
        AAsset_close(model_asset);
    }

    LOGE("准备加载 param，路径是: [%s]", parampath);
    LOGE("准备加载 bin，路径是: [%s]", modelpath);

    int param_result = gdrnet.load_param(mgr, parampath);
    int model_result = gdrnet.load_model(mgr, modelpath);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "param load result: %d", param_result);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Model load result: %d", model_result);
    
    if (param_result != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Failed to load model file: %s (error: %d)", parampath, param_result);
        return -1;
    }
    if (model_result != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Failed to load model file: %s (error: %d)", modelpath, model_result);
        return -1;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Model loaded successfully!");

    return 0;
}

int GDRNet::inference(const cv::Mat& roi, const cv::Rect& bbox, int object_label, PoseResult& result)
{
    // ==========================================
    // 【新增防护】：过滤掉极小的误检框，防止出现距离 9.9 米的离谱数据
    // ==========================================
    if (bbox.width < 30 || bbox.height < 30) {
        __android_log_print(ANDROID_LOG_WARN, "GDRNet", "ROI too small (%dx%d), skipping inference.", bbox.width, bbox.height);
        return -1;
    }

    // 1. 调整大小到 256x256
    cv::Mat resized;
    cv::resize(roi, resized, cv::Size(INPUT_RES, INPUT_RES));

    // 2. 归一化: 像素值 / 255.0
    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR2RGB, INPUT_RES, INPUT_RES);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(0, norm_vals);

    // 3. 计算辅助输入
    float roi_center_x = bbox.x + bbox.width / 2.0f;
    float roi_center_y = bbox.y + bbox.height / 2.0f;
    float roi_width = bbox.width;
    float roi_height = bbox.height;
    float resize_ratio = (float)INPUT_RES / std::max(roi_width, roi_height);

    // 4. 准备所有必要的输入
    ncnn::Mat in0 = in; // [1, 3, 256, 256] 图像已就绪

    // in1: [1, 2, 64, 64] -> 2D 坐标网格 (coord_2d)
    float img_w = default_camera_params.cx * 2.0f;
    float img_h = default_camera_params.cy * 2.0f;

    ncnn::Mat in1(64, 64, 2);
    for (int y = 0; y < 64; y++) {
        float* ptr_x = in1.channel(0).row(y);
        float* ptr_y = in1.channel(1).row(y);
        for (int x = 0; x < 64; x++) {
            // 计算当前像素在原图中的绝对坐标
            float orig_x = bbox.x + ((float)x / 63.0f) * bbox.width;
            float orig_y = bbox.y + ((float)y / 63.0f) * bbox.height;

            // 归一化到 0~1 (除以原图真实宽高)
            ptr_x[x] = (orig_x - default_camera_params.cx) / default_camera_params.fx;
            ptr_y[x] = (orig_y - default_camera_params.cy) / default_camera_params.fy;
        }
    }

    // in2: [1, 3, 3] -> 相机内参 K
    ncnn::Mat in2(3, 3);
    in2.fill(0.f);
    in2.row(0)[0] = default_camera_params.fx;
    in2.row(0)[2] = default_camera_params.cx;
    in2.row(1)[1] = default_camera_params.fy;
    in2.row(1)[2] = default_camera_params.cy;
    in2.row(2)[2] = 1.0f;

    // in3: [1, 3] -> 物体的 3D 尺寸范围 (3D bbox extents)
    ncnn::Mat in3(3);
    switch (object_label) {
        switch (object_label) {
            case 39: // 瓶子
                in3[0] = 80.0f;  in3[1] = 80.0f;  in3[2] = 180.0f;
                break;
            case 41: // 杯子
                in3[0] = 70.0f;  in3[1] = 70.0f;  in3[2] =110.0f;
                break;
            case 64: // 鼠标
                in3[0] = 70.0f;  in3[1] = 100.0f;  in3[2] = 40.0f;
                break;
            default:
                in3.fill(0.0f);
                break;
        }
    }

    // in4: [1, 2] -> ROI 中心 (roi_centers)
    ncnn::Mat in4(1, 1, 2);
    in4.channel(0)[0] = roi_center_x;
    in4.channel(1)[0] = roi_center_y;

    // in5: [1, 2] -> ROI 缩放/尺寸 (roi_whs)
    ncnn::Mat in5(1, 1, 2);
    in5.channel(0)[0] = roi_width;
    in5.channel(1)[0] = roi_height;

    // in6: [1] -> 缩放比例 (resize_ratios)
    ncnn::Mat in6(1);
    in6[0] = resize_ratio;

    // 5. 创建提取器并设置输入
    ncnn::Extractor ex = gdrnet.create_extractor();

    ex.input("in0", in0);
    ex.input("in1", in1);
    ex.input("in2", in2);
    ex.input("in3", in3);
    ex.input("in4", in4);
    ex.input("in5", in5);
    ex.input("in6", in6);

    // 6. 执行推理
    ncnn::Mat out0, out1;

    int ret_r = ex.extract("out0", out0);  // 旋转矩阵
    int ret_t = ex.extract("out1", out1);  // 平移向量

    if (ret_r != 0 || ret_t != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "找不到输出节点，提取失败！out0: %d, out1: %d", ret_r, ret_t);
        return -1;
    }

    // 7. 解析输出
    if (out0.w * out0.h * out0.c >= 6 && out1.w * out1.h * out1.c >= 3)
    {
        // 使用指针读取，防止内存越界
        const float* rot_data = (const float*)out0.data;
        float r1 = rot_data[0], r2 = rot_data[1], r3 = rot_data[2];
        float r4 = rot_data[3], r5 = rot_data[4], r6 = rot_data[5];

        const float* trans_data = (const float*)out1.data;
        float pred_tx = trans_data[0];
        float pred_ty = trans_data[1];
        float pred_tz = trans_data[2];


        // 平移向量反投影
        float cx_img = pred_tx * roi_width + roi_center_x;
        float cy_img = pred_ty * roi_height + roi_center_y;

        float true_tz = pred_tz * resize_ratio;

        float true_tx = (cx_img - default_camera_params.cx) * true_tz / default_camera_params.fx;
        float true_ty = (cy_img - default_camera_params.cy) * true_tz / default_camera_params.fy;
        // 6D 旋转正交化 (Gram-Schmidt)
        cv::Vec3f x_vec(r1, r2, r3);
        cv::Vec3f y_vec(r4, r5, r6);

        x_vec = x_vec / cv::norm(x_vec);
        y_vec = y_vec - (x_vec.dot(y_vec)) * x_vec;
        y_vec = y_vec / cv::norm(y_vec);
        cv::Vec3f z_vec = x_vec.cross(y_vec);

        cv::Matx33f R_allo(
                x_vec[0], y_vec[0], z_vec[0],
                x_vec[1], y_vec[1], z_vec[1],
                x_vec[2], y_vec[2], z_vec[2]
        );

        // 消除透视畸变 (Allo to Ego Transform)
        cv::Vec3f ray(true_tx, true_ty, true_tz);
        ray = ray / cv::norm(ray);

        cv::Vec3f z_axis_base(0, 0, 1.0f);
        cv::Vec3f axis = z_axis_base.cross(ray);
        float sin_angle = cv::norm(axis);
        float cos_angle = z_axis_base.dot(ray);

        cv::Matx33f R_ray = cv::Matx33f::eye();
        if (sin_angle > 1e-5) {
            axis = axis / sin_angle;
            float angle = std::atan2(sin_angle, cos_angle);

            float c = std::cos(angle);
            float s = std::sin(angle);
            float v = 1.0f - c;
            float kx = axis[0], ky = axis[1], kz = axis[2];

            R_ray(0,0) = kx*kx*v + c;      R_ray(0,1) = kx*ky*v - kz*s;   R_ray(0,2) = kx*kz*v + ky*s;
            R_ray(1,0) = kx*ky*v + kz*s;   R_ray(1,1) = ky*ky*v + c;      R_ray(1,2) = ky*kz*v - kx*s;
            R_ray(2,0) = kx*kz*v - ky*s;   R_ray(2,1) = ky*kz*v + kx*s;   R_ray(2,2) = kz*kz*v + c;
        }

        cv::Matx33f R_ego = R_ray * R_allo;

        // 保存旋转
        result.rotation[0] = R_ego(0,0); result.rotation[1] = R_ego(0,1); result.rotation[2] = R_ego(0,2);
        result.rotation[3] = R_ego(1,0); result.rotation[4] = R_ego(1,1); result.rotation[5] = R_ego(1,2);
        result.rotation[6] = R_ego(2,0); result.rotation[7] = R_ego(2,1); result.rotation[8] = R_ego(2,2);

        // 保存平移 (单位为米)
        result.translation[0] = true_tx;
        result.translation[1] = true_ty;
        result.translation[2] = true_tz;

        __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Final Ego Pose - T: [%.2f, %.2f, %.2f]", true_tx, true_ty, true_tz);
    }
    else
    {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "输出节点大小异常");
    }

    return 0;
}


int GDRNet::draw3DAxes(cv::Mat& rgb, const PoseResult& pose, const CameraParams& camera_params, const cv::Rect& roi)
{
    float axis_length = 1.0f;

    // 原点
    cv::Point3f origin(0, 0, 0);

    // X轴（红色）
    cv::Point3f x_axis(axis_length, 0, 0);

    // Y轴（绿色）
    cv::Point3f y_axis(0, axis_length, 0);

    // Z轴（蓝色）
    cv::Point3f z_axis(0, 0, axis_length);

    // 应用旋转和平移
    auto transform_point = [&](const cv::Point3f& p) {
        float x = p.x;
        float y = p.y;
        float z = p.z;

        // 应用旋转矩阵
        float rx = pose.rotation[0] * x + pose.rotation[1] * y + pose.rotation[2] * z;
        float ry = pose.rotation[3] * x + pose.rotation[4] * y + pose.rotation[5] * z;
        float rz = pose.rotation[6] * x + pose.rotation[7] * y + pose.rotation[8] * z;

        // 应用平移
        return cv::Point3f(rx + pose.translation[0], ry + pose.translation[1], rz + pose.translation[2]);
    };

    cv::Point3f transformed_origin = transform_point(origin);
    cv::Point3f transformed_x_axis = transform_point(x_axis);
    cv::Point3f transformed_y_axis = transform_point(y_axis);
    cv::Point3f transformed_z_axis = transform_point(z_axis);


    // 投影到2D
    auto project_point = [&](const cv::Point3f& p) {
        if (p.z > 0.01f) {
            float u = (camera_params.fx * p.x / p.z) + camera_params.cx;
            float v = (camera_params.fy * p.y / p.z) + camera_params.cy;
            return cv::Point2f(u, v);
        }
        return cv::Point2f(-1, -1);
    };

    cv::Point2f origin_2d = project_point(transformed_origin);
    cv::Point2f x_axis_2d = project_point(transformed_x_axis);
    cv::Point2f y_axis_2d = project_point(transformed_y_axis);
    cv::Point2f z_axis_2d = project_point(transformed_z_axis);

    // 绘制坐标轴
    if (origin_2d.x >= 0) {
        // 【核心修正】：删掉冗余的 cv::line，避免边缘重叠模糊，并缩小箭头尺寸 tipLength = 0.15

        // 绘制X轴（红色：BGR中的 R 在最后）
        if (x_axis_2d.x >= 0) {
            cv::arrowedLine(rgb, origin_2d, x_axis_2d, cv::Scalar(0, 0, 255), 3, 8, 0, 0.15);
        }

        // 绘制Y轴（绿色：BGR中的 G 在中间）
        if (y_axis_2d.x >= 0) {
            cv::arrowedLine(rgb, origin_2d, y_axis_2d, cv::Scalar(0, 255, 0), 3, 8, 0, 0.15);
        }

        // 绘制Z轴（蓝色：BGR中的 B 在最前）
        if (z_axis_2d.x >= 0) {
            cv::arrowedLine(rgb, origin_2d, z_axis_2d, cv::Scalar(255, 0, 0), 3, 8, 0, 0.15);
        }

        // 绘制原点
        cv::circle(rgb, origin_2d, 5, cv::Scalar(255, 255, 255), -1);
    }

    return 0;
}

int GDRNet::draw3DBox(cv::Mat& rgb, const PoseResult& pose, const CameraParams& camera_params, float ext_x_m, float ext_y_m, float ext_z_m)
{
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "draw3DBox called with size: %.3f x %.3f x %.3f", ext_x_m, ext_y_m, ext_z_m);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Camera params - fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f", 
        camera_params.fx, camera_params.fy, camera_params.cx, camera_params.cy);
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Pose translation - tx: %.3f, ty: %.3f, tz: %.3f", 
        pose.translation[0], pose.translation[1], pose.translation[2]);
    
    // 1. 根据物体的物理尺寸，定义 3D 框的 8 个顶点（以物体中心为原点）
    float dx = ext_x_m / 2.0f;
    float dy = ext_y_m / 2.0f;
    float dz = ext_z_m / 2.0f;

    std::vector<cv::Point3f> corners_3d = {
            cv::Point3f( dx,  dy, -dz),
            cv::Point3f(-dx,  dy, -dz),
            cv::Point3f(-dx, -dy, -dz),
            cv::Point3f( dx, -dy, -dz),
            cv::Point3f( dx,  dy,  dz),
            cv::Point3f(-dx,  dy,  dz),
            cv::Point3f(-dx, -dy,  dz),
            cv::Point3f( dx, -dy,  dz)
    };

    std::vector<cv::Point2f> corners_2d;

    // 2. 将每个 3D 点进行旋转、平移，并投影到 2D 屏幕上
    for (size_t i = 0; i < corners_3d.size(); i++) {
        const auto& pt = corners_3d[i];
        // 应用绝对旋转矩阵 (R_ego)
        float rx = pose.rotation[0] * pt.x + pose.rotation[1] * pt.y + pose.rotation[2] * pt.z;
        float ry = pose.rotation[3] * pt.x + pose.rotation[4] * pt.y + pose.rotation[5] * pt.z;
        float rz = pose.rotation[6] * pt.x + pose.rotation[7] * pt.y + pose.rotation[8] * pt.z;

        // 应用绝对平移 (T_ego)
        float tx = rx + pose.translation[0];
        float ty = ry + pose.translation[1];
        float tz = rz + pose.translation[2];



        // 投影到 2D 像素坐标 (加入 > 0.01f 的防御，防止点在相机背面导致畸变)
        if (tz > 0.01f) {
            float u = (camera_params.fx * tx / tz) + camera_params.cx;
            float v = (camera_params.fy * ty / tz) + camera_params.cy;
            corners_2d.push_back(cv::Point2f(u, v));
        } else {
            // 如果跑到相机背面了，返回一个无效点
            corners_2d.push_back(cv::Point2f(-1, -1));
        }
    }

    // 3. 检查是否所有的点都在相机前方（有效）
    int invalid_points = 0;
    for (int i = 0; i < corners_2d.size(); i++) {
        const auto& pt = corners_2d[i];
        if (pt.x < 0 && pt.y < 0) {
            invalid_points++;
            __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Invalid corner %d: (%.2f, %.2f)", i, pt.x, pt.y);
        }
    }
    
    if (invalid_points > 0) {
        __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Found %d invalid points, skipping draw", invalid_points);
        return -1; // 只要有任何一个点在相机背面，为了画面不崩，就不画框了
    }

    // 4. 定义连线关系并绘制 3D 框的 12 条边
    // 颜色为黄色 BGR: (0, 255, 255)，线宽为 2
    cv::Scalar box_color(0, 255, 255);
    int thickness = 2;

    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Drawing 3D box edges");

    // 前面 (Front Face) 0-1, 1-2, 2-3, 3-0
    cv::line(rgb, corners_2d[0], corners_2d[1], box_color, thickness);
    cv::line(rgb, corners_2d[1], corners_2d[2], box_color, thickness);
    cv::line(rgb, corners_2d[2], corners_2d[3], box_color, thickness);
    cv::line(rgb, corners_2d[3], corners_2d[0], box_color, thickness);

    // 后面 (Back Face) 4-5, 5-6, 6-7, 7-4
    cv::line(rgb, corners_2d[4], corners_2d[5], box_color, thickness);
    cv::line(rgb, corners_2d[5], corners_2d[6], box_color, thickness);
    cv::line(rgb, corners_2d[6], corners_2d[7], box_color, thickness);
    cv::line(rgb, corners_2d[7], corners_2d[4], box_color, thickness);

    // 连接前后面的 4 条支柱 0-4, 1-5, 2-6, 3-7
    cv::line(rgb, corners_2d[0], corners_2d[4], box_color, thickness);
    cv::line(rgb, corners_2d[1], corners_2d[5], box_color, thickness);
    cv::line(rgb, corners_2d[2], corners_2d[6], box_color, thickness);
    cv::line(rgb, corners_2d[3], corners_2d[7], box_color, thickness);

    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "draw3DBox completed successfully");

    return 0;
}