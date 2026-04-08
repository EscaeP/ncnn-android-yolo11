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

    // 4. 准备所有必要的输入 (严格遵循 gdrn_ycbv_scripted.pt 的 Input Shapes)

    ncnn::Mat in0 = in; // [1, 3, 256, 256] 图像已就绪

    // in1: [1, 2, 64, 64] -> 2D 坐标网格 (coord_2d)
    // GDRNet 通常需要一个相对于特征图的 XY 网格图
    ncnn::Mat in1(64, 64, 2);
    for (int y = 0; y < 64; y++) {
        float* ptr_x = in1.channel(0).row(y);
        float* ptr_y = in1.channel(1).row(y);
        for (int x = 0; x < 64; x++) {
            ptr_x[x] = (float)x / 63.0f; // 归一化 X
            ptr_y[x] = (float)y / 63.0f; // 归一化 Y
        }
    }

    // in2: [1, 3, 3] -> 相机内参 K
    ncnn::Mat in2(3, 3);
    in2.fill(0.f); // 先全部清零
    in2.row(0)[0] = default_camera_params.fx;
    in2.row(0)[2] = default_camera_params.cx;
    in2.row(1)[1] = default_camera_params.fy;
    in2.row(1)[2] = default_camera_params.cy;
    in2.row(2)[2] = 1.0f;

    // in3: [1, 3] -> 物体的 3D 尺寸范围 (3D bbox extents)
    ncnn::Mat in3(3);
    // 根据物体类别设置不同的尺寸
    // 这里使用的是 YCB-V 数据集的物体尺寸（单位：毫米）
    switch (object_label) {
        case 39: //瓶子
            in3[0] = 80.0f;
            in3[1] = 80.0f;
            in3[2] = 180.0f;
            __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "瓶子");
            break;
        case 0: //
            in3[0] = 80.0f;
            in3[1] = 80.0f;
            in3[2] = 150.0f;
            break;
        default: // 默认尺寸
            in3[0] = 0.0f;
            in3[1] = 0.0f;
            in3[2] = 0.0f;
            break;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Object label: %d, size: %.2f x %.2f x %.2f", 
        object_label, in3[0], in3[1], in3[2]);

    // in4: [1, 2] -> ROI 中心 (roi_centers)
    ncnn::Mat in4(1, 1, 2);
    in4.channel(0)[0] = roi_center_x; // 第一个通道放 X
    in4.channel(1)[0] = roi_center_y; // 第二个通道放 Y

    // in5: [1, 2] -> ROI 缩放/尺寸 (roi_whs 或 roi_scales)

    ncnn::Mat in5(1, 1, 2);
    in5.channel(0)[0] = roi_width;
    in5.channel(1)[0] = roi_height;
    //
    //    ncnn::Mat in5(2);
    //    in5[0] = roi_width;
    //    in5[1] = roi_height;

    // in6: [1] -> 物体 ID (obj_id)
    ncnn::Mat in6(1);
    in6[0] = (float)(object_label + 1); // 物体 ID 从 1 开始

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
    
    // 尝试提取输出节点
    int ret_r = ex.extract("out0", out0);  // 旋转矩阵 [B,3,3]
    int ret_t = ex.extract("out1", out1);  // 平移向量 [B,3]

    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Extract out0 result: %d, out1 result: %d", ret_r, ret_t);
    
    
    if (ret_r != 0)
    {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "找不到 out0 节点，旋转矩阵提取失败！");
    }
    
    if (ret_t != 0)
    {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "找不到 out1 节点，平移向量提取失败！");
        
        // 尝试打印所有可用的输出节点，以便调试
        __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Attempting to list all output nodes...");
        // 注意：ncnn 没有直接提供列出所有输出节点的 API
        // 这里我们可以尝试一些常见的输出节点名称
        const char* possible_outputs[] = {"out0", "out1", };
        for (size_t i = 0; i < sizeof(possible_outputs)/sizeof(possible_outputs[0]); i++)
        {
            ncnn::Mat test_out;
            int test_ret = ex.extract(possible_outputs[i], test_out);
            if (test_ret == 0)
            {
                __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Found output node: %s", possible_outputs[i]);
                // 打印输出节点的形状
                if (test_out.dims > 0)
                {
                    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "  Shape: %d x %d x %d", test_out.w, test_out.h, test_out.c);
                }
            }
        }

    }

    // 7. 解析输出
    // 根据用户说明，out0是6D旋转向量，out1是平移向量
    if (ret_r == 0 && ret_t == 0)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Processing out0 as 6D rotation vector and out1 as translation vector");
        __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "out0 - dims: %d, w: %d, h: %d, c: %d", out0.dims, out0.w, out0.h, out0.c);
        __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "out1 - dims: %d, w: %d, h: %d, c: %d", out1.dims, out1.w, out1.h, out1.c);
    

        
        // 检查out0形状（6D旋转向量应该是6个元素）
        if (out0.dims == 1 && out0.w == 6 && out0.h == 1 && out0.c == 1)
        {
            // 提取6D旋转向量
            float r1, r2, r3, r4, r5, r6;
            // 处理1维6元素的情况
            r1 = out0[0];
            r2 = out0[1];
            r3 = out0[2];
            r4 = out0[3];
            r5 = out0[4];
            r6 = out0[5];
            
            __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "6D rotation vector: %.6f %.6f %.6f %.6f %.6f %.6f", 
                r1, r2, r3, r4, r5, r6);
            
            // 检查out1形状（平移向量应该是3个元素）
            if (out1.dims == 2 && out1.w == 1 && out1.h == 3 && out1.c == 1)
            {
                // 提取平移向量
                float tx, ty, tz;
                // 处理2维1x3的情况
                tx = out1[0];
                ty = out1[1];
                tz = out1[2];
                
                __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "Translation vector: %.6f %.6f %.6f", tx, ty, tz);
                
                // 将6D旋转向量转换为旋转矩阵
                // 6D旋转向量格式为 [r1, r2, r3, r4, r5, r6]
                // 其中第一列是 [r1, r4, r5]，第二列是 [r2, r3, r6]
                // 第三列通过前两列的叉积计算
                cv::Vec3f col1(r1, r4, r5); // 第一列
                cv::Vec3f col2(r2, r3, r6); // 第二列
                cv::Vec3f col3 = col1.cross(col2); // 第三列（叉积）
                
                // 归一化列向量以确保正交性
                col1 = col1 / cv::norm(col1);
                col2 = col2 / cv::norm(col2);
                col3 = col3 / cv::norm(col3);
                
                // 重新计算第二列以确保正交（Gram-Schmidt正交化）
                col2 = col2 - col1.dot(col2) * col1;
                col2 = col2 / cv::norm(col2);
                
                // 重新计算第三列以确保正交
                col3 = col1.cross(col2);
                col3 = col3 / cv::norm(col3);
                
                // 构建旋转矩阵
                result.rotation[0] = col1[0];
                result.rotation[1] = col2[0];
                result.rotation[2] = col3[0];
                result.rotation[3] = col1[1];
                result.rotation[4] = col2[1];
                result.rotation[5] = col3[1];
                result.rotation[6] = col1[2];
                result.rotation[7] = col2[2];
                result.rotation[8] = col3[2];
                
                // 设置平移向量（转换为毫米单位）
                result.translation[0] = tx * 1000.0f;
                result.translation[1] = ty * 1000.0f;
                result.translation[2] = tz * 1000.0f;

            }
            else
            {
                __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Failed to extract translation vector from out1");
                if (out1.dims > 0)
                {
                    __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "out1 shape: %d x %d x %d", out1.w, out1.h, out1.c);
                }
            }
        }
        else
        {
            __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Failed to extract 6D rotation vector from out0");
            if (out0.dims > 0)
            {
                __android_log_print(ANDROID_LOG_DEBUG, "GDRNet", "out0 shape: %d x %d x %d", out0.w, out0.h, out0.c);
            }
        }
    }
    else
    {
        __android_log_print(ANDROID_LOG_ERROR, "GDRNet", "Failed to extract output nodes - out0: %d, out1: %d", ret_r, ret_t);
    }

    return 0;
}



int GDRNet::draw3DAxes(cv::Mat& rgb, const PoseResult& pose, const CameraParams& camera_params, const cv::Rect& roi)
{
    // 计算坐标轴的3D点
    float axis_length = 200.0f; // 坐标轴长度（毫米），增加到200
    
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
        if (p.z > 0) {
            float u = (camera_params.fx * p.x / p.z) + camera_params.cx;
            float v = (camera_params.fy * p.y / p.z) + camera_params.cy;
            u += roi.x;
            v += roi.y;
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
        // 绘制X轴（红色）
        if (x_axis_2d.x >= 0) {
            cv::line(rgb, origin_2d, x_axis_2d, cv::Scalar(0, 0, 255), 3); // 线宽增加到3
            // 绘制箭头
            cv::arrowedLine(rgb, origin_2d, x_axis_2d, cv::Scalar(0, 0, 255), 3, 8, 0, 0.4); // 线宽增加到3，箭头大小增加到0.4
        }
        
        // 绘制Y轴（绿色）
        if (y_axis_2d.x >= 0) {
            cv::line(rgb, origin_2d, y_axis_2d, cv::Scalar(0, 255, 0), 3); // 线宽增加到3
            // 绘制箭头
            cv::arrowedLine(rgb, origin_2d, y_axis_2d, cv::Scalar(0, 255, 0), 3, 8, 0, 0.4); // 线宽增加到3，箭头大小增加到0.4
        }
        
        // 绘制Z轴（蓝色）
        if (z_axis_2d.x >= 0) {
            cv::line(rgb, origin_2d, z_axis_2d, cv::Scalar(255, 0, 0), 3); // 线宽增加到3
            // 绘制箭头
            cv::arrowedLine(rgb, origin_2d, z_axis_2d, cv::Scalar(255, 0, 0), 3, 8, 0, 0.4); // 线宽增加到3，箭头大小增加到0.4
        }
        
        // 绘制原点
        cv::circle(rgb, origin_2d, 5, cv::Scalar(255, 255, 255), -1); // 原点大小增加到5
    }
    
    return 0;
}