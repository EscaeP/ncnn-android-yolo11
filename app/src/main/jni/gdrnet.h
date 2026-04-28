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

#ifndef GDRNET_H
#define GDRNET_H

#include <opencv2/core/core.hpp>

#include <net.h>

struct PoseResult
{
    float rotation[9];  // 3x3 rotation matrix [B,3,3]
    float translation[3];  // 3D translation [B,3]
    float scale[3];  // 物体3D尺寸 [B,3]
};

struct CameraParams
{
    float fx;  // 焦距x
    float fy;  // 焦距y
    float cx;  // 光心x
    float cy;  // 光心y
};



class GDRNet
{
public:
    GDRNet();
    ~GDRNet();

    int load(const char* parampath, const char* modelpath, bool use_gpu = false);
    int load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu = false);

    int inference(const cv::Mat& roi, const cv::Rect& bbox, int object_label, PoseResult& result);
    
    // 上传图片进行处理
    int inferPicture(const cv::Mat& image, PoseResult& result);
    
    // 3D姿态相关方法
    int draw3DAxes(cv::Mat& rgb, const PoseResult& pose, const CameraParams& camera_params, const cv::Rect& roi);
    // 绘制 3D 边界框
    int draw3DBox(cv::Mat& rgb, const PoseResult& pose, const CameraParams& camera_params, float size_x, float size_y, float size_z);


public:
    // 默认相机参数（需要根据实际设备调整）
    CameraParams default_camera_params = { 400.0f, 400.0f, 200.0f, 320.0f };
    
    // 设置相机参数
    void setCameraParams(float fx, float fy, float cx, float cy);

private:
    ncnn::Net gdrnet;
    const int INPUT_RES = 256;
};

#endif // GDRNET_H