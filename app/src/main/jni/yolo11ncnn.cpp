// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/bitmap.h>
#include <fstream>
#include <string>
#include <stdexcept>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolo11.h"
#include "gdrnet.h"

#include "ndkcamera.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include "json.hpp"

using json = nlohmann::json;

// 使用你定义的全局静态变量
static json g_det_data;
static bool g_json_loaded = false;

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static YOLO11* g_yolo11 = 0;
static GDRNet* g_gdrnet = 0;
static ncnn::Mutex lock;
static bool g_paused = false;
static cv::Mat g_last_frame;
static std::vector<Object> g_last_objects;
static int current_task = 0;

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // 检查是否暂停
    if (g_paused)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "System paused, displaying last frame");
        // 显示最后一帧的追踪结果
        if (!g_last_frame.empty())
        {
            rgb = g_last_frame.clone();
        }
        else
        {
            draw_unsupported(rgb);
        }
        return;
    }

    // yolo11
    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolo11)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Starting YOLO11 detection...");
            std::vector<Object> objects;
            g_yolo11->detect(rgb, objects);
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "YOLO11 detected %d objects", (int)objects.size());

            // GDR-Net processing
            if (g_gdrnet && !objects.empty())
            {
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "GDR-Net is available, processing objects...");
                // 使用固定的相机参数值
                float current_cx = rgb.cols / 2.0f;
                float current_cy = rgb.rows / 2.0f;
                float current_fx = 417.58f;
                float current_fy = 417.58f;

                g_gdrnet->setCameraParams(current_fx, current_fy, current_cx, current_cy);
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Set camera params - fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f", 
                    current_fx, current_fy, current_cx, current_cy);
                
                int object_count = 0;
                for (const Object& obj : objects)
                {
                    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Processing object %d: label=%d, rect=(%.0f,%.0f,%.0f,%.0f)", 
                        object_count, obj.label, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
                    // 1. 裁剪ROI区域
                    cv::Rect bbox = obj.rect;
                    // 确保ROI在图像范围内
                    bbox.x = std::max(0, bbox.x);
                    bbox.y = std::max(0, bbox.y);
                    bbox.width = std::min(rgb.cols - bbox.x, bbox.width);
                    bbox.height = std::min(rgb.rows - bbox.y, bbox.height);

                    if (bbox.width > 0 && bbox.height > 0)
                    {
                        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Valid ROI: (%d,%d,%d,%d)", 
                            bbox.x, bbox.y, bbox.width, bbox.height);
                        
                        // coco和pose任务
                        {
            
                            // 2. 执行GDR-Net推理
                            PoseResult result;
                            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Running GDR-Net inference...");
                            g_gdrnet->inference(rgb, bbox, obj.label, result);

                            // 绘制3维坐标轴表示6D姿态
                            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Drawing 3D axes...");
                            g_gdrnet->draw3DAxes(rgb, result, g_gdrnet->default_camera_params, bbox);

                            // 绘制3D边界框
                            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Drawing 3D box...");
                            // 根据物体类别设置不同的尺寸
                            float size_x = 0.0f;
                            float size_y = 0.0f;
                            float size_z = 0.0f;
                            switch (obj.label) {
                                case 39: // 瓶子
                                    size_x = 0.5f;
                                    size_y = 0.5f;
                                    size_z = 1.5f;
                                    break;
                                case 41: // 杯子
                                    size_x = 1.022f;
                                    size_y = 1.023f;
                                    size_z = 1.393f;
                                    break;
                                case 64: // 鼠标
                                    size_x = 1.016f;
                                    size_y = 0.801f;
                                    size_z = 0.524f;
                                    break;
                                default: // 默认尺寸
                                    size_x = 0.0f;
                                    size_y = 0.0f;
                                    size_z = 0.0f; // 8cm x 8cm x 15cm
                                    break;
                            }
                            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Object label: %d, size: %.3f x %.3f x %.3f", obj.label, size_x, size_y, size_z);
                            int box_result = g_gdrnet->draw3DBox(rgb, result, g_gdrnet->default_camera_params, size_x, size_y, size_z);
                            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "draw3DBox result: %d", box_result);
                        }
                    }
                    else
                    {
                        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Invalid ROI, skipping");
                    }
                    object_count++;
                }
            }
            else if (!g_gdrnet)
            {
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "GDR-Net is not initialized");
            }
            else if (objects.empty())
            {
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "No objects detected, skipping GDR-Net processing");
            }

            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Drawing YOLO11 detections...");
            g_yolo11->draw(rgb, objects);

            // 保存当前帧和检测结果（包含所有绘制的内容）
            g_last_frame = rgb.clone();
            g_last_objects = objects;
        }
        else
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "YOLO11 is not initialized");
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolo11;
        g_yolo11 = 0;

        delete g_gdrnet;
        g_gdrnet = 0;
    }

    ncnn::destroy_gpu_instance();

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int taskid, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_loadModel(JNIEnv *env, jobject thiz, jobject assetManager,
                                                 jint taskid, jint modelid, jint cpugpu) {
    if (taskid < 0 || taskid > 1 || modelid < 0 || modelid > 5 || cpugpu < 0 || cpugpu > 2) {
        return JNI_FALSE;
    }

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char *tasknames[3] =
            {
                    "",
                    "_pose",
                    ""
            };

    const char *modeltypes[6] =
            {
                    "n",
                    "s",
                    "m",
                    "n",
                    "s",
                    "m"
            };

    std::string parampath =
            std::string("yolo11") + modeltypes[(int) modelid] + tasknames[(int) taskid] +
            ".ncnn.param";
    std::string modelpath =
            std::string("yolo11") + modeltypes[(int) modelid] + tasknames[(int) taskid] +
            ".ncnn.bin";
    bool use_gpu = (int) cpugpu == 1;
    bool use_turnip = (int) cpugpu == 2;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        {
            static int old_taskid = 0;
            static int old_modelid = 0;
            static int old_cpugpu = 0;
            if (taskid != old_taskid || (modelid % 3) != old_modelid || cpugpu != old_cpugpu) {
                // taskid or model or cpugpu changed
                delete g_yolo11;
                g_yolo11 = 0;
            }
            old_taskid = taskid;
            old_modelid = modelid % 3;
            old_cpugpu = cpugpu;
            current_task = taskid;

            ncnn::destroy_gpu_instance();

            if (use_turnip) {
                ncnn::create_gpu_instance("libvulkan_freedreno.so");
            } else if (use_gpu) {
                ncnn::create_gpu_instance();
            }

            if (!g_yolo11) {
                if (taskid == 0) g_yolo11 = new YOLO11_det;
                if (taskid == 1) g_yolo11 = new YOLO11_pose;

                g_yolo11->load(mgr, parampath.c_str(), modelpath.c_str(), use_gpu || use_turnip);
            }

            // Load GDR-Net model
            if (!g_gdrnet) {
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Creating GDR-Net instance...");
                g_gdrnet = new GDRNet;
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                    "Loading GDR-Net model: model_sim.ncnn.param, model_sim.ncnn.bin");
                int gdr_load_result = g_gdrnet->load(mgr, "model_sim.ncnn.param",
                                                     "model_sim.ncnn.bin", use_gpu || use_turnip);
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "GDR-Net load result: %d",
                                    gdr_load_result);
                if (gdr_load_result != 0) {
                    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to load GDR-Net model");
                } else {
                    __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                        "GDR-Net model loaded successfully!");
                }
            }
            int target_size = 320;
            if ((int) modelid >= 3)
                target_size = 480;
            if ((int) modelid >= 6)
                target_size = 640;
            g_yolo11->set_det_target_size(target_size);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_openCamera(JNIEnv *env, jobject thiz, jint facing) {
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int) facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_closeCamera(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_setOutputWindow(JNIEnv *env, jobject thiz, jobject surface) {
    ANativeWindow *win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

// public native boolean togglePause();
JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_togglePause(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "togglePause");

    g_paused = !g_paused;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Pause state: %s",
                        g_paused ? "paused" : "resumed");

    return JNI_TRUE;
}

JNIEXPORT jobject JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_processImage(JNIEnv *env, jobject thiz, jobject bitmap,
                                                    jstring imageName, jstring jsonContent) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "========== [ProcessImage START] ==========");

    // ==========================================
    // 0. 解析 JSON
    // ==========================================
    if (!g_json_loaded && jsonContent != nullptr) {
        const char *jsonChars = env->GetStringUTFChars(jsonContent, nullptr);
        g_det_data = json::parse(jsonChars, nullptr, false);
        env->ReleaseStringUTFChars(jsonContent, jsonChars);

        if (g_det_data.is_discarded()) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "JSON Parse Failed! Invalid format.");
        } else {
            g_json_loaded = true;
        }
    }

    // ==========================================
    // 1. Android Bitmap 转 OpenCV Mat
    // ==========================================
    AndroidBitmapInfo info;
    void *pixels;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return nullptr;

    cv::Mat image(info.height, info.width, CV_8UC4, pixels);
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bitmap);

    // ==========================================
    // 2. 解析 dict_key
    // ==========================================
    const char *imageNameChars = env->GetStringUTFChars(imageName, nullptr);
    std::string nameStr(imageNameChars);
    env->ReleaseStringUTFChars(imageName, imageNameChars);

    std::string numPart;
    for (char c: nameStr) {
        if (isdigit(c)) numPart += c;
    }
    int img_id = numPart.empty() ? 0 : std::stoi(numPart);
    std::string dict_key = "2/" + std::to_string(img_id);

    // ==========================================
    // 3. 执行 GDR-Net 位姿估计
    // ==========================================
    const float SCORE_THR = 0.9f;
    std::vector<int> target_ids = {1, 5, 6, 8, 9, 10, 11, 12};

    if (g_gdrnet && g_json_loaded && g_det_data.contains(dict_key)) {
        // 设置相机参数对齐 demo.py
        float current_cx = 325.2611f;
        float current_cy = 242.04899f;
        float current_fx = 572.4114f;
        float current_fy = 573.57043f;
        g_gdrnet->setCameraParams(current_fx, current_fy, current_cx, current_cy);

        for (auto &item: g_det_data[dict_key]) {
            float score = item.value("score", 0.0f);
            int obj_id = item.value("obj_id", item.value("category_id", -1));

            if (score < SCORE_THR) continue;
            if (std::find(target_ids.begin(), target_ids.end(), obj_id) ==
                target_ids.end())
                continue;

            // 解析边界框 (注意：Python给的是 w, h)
            auto &b = item.contains("bbox_est") ? item["bbox_est"] : item["bbox"];
            cv::Rect2f rect(b[0], b[1], b[2], b[3]);

            // 直接推理，裁剪已在 inference 内部完成
            PoseResult result;
            int infer_ret = g_gdrnet->inference(rgb, rect, obj_id, result);

            if (infer_ret == 0) {
                float scale_factor = 0.005f;
                float sx = 0.0f, sy = 0.0f, sz = 0.0f;
                switch (obj_id) {
                    case 1:
                        sx = 75.9f * scale_factor;
                        sy = 77.6f * scale_factor;
                        sz = 91.8f * scale_factor;
                        break;
                    case 5:
                        sx = 100.8f * scale_factor;
                        sy = 181.8f * scale_factor;
                        sz = 193.7f * scale_factor;
                        break;
                    case 6:
                        sx = 67.0f * scale_factor;
                        sy = 127.6f * scale_factor;
                        sz = 117.5f * scale_factor;
                        break;
                    case 8:
                        sx = 229.5f * scale_factor;
                        sy = 75.5f * scale_factor;
                        sz = 208.0f * scale_factor;
                        break;
                    case 9:
                        sx = 104.4f * scale_factor;
                        sy = 77.4f * scale_factor;
                        sz = 85.7f * scale_factor;
                        break;
                    case 10:
                        sx = 150.2f * scale_factor;
                        sy = 107.1f * scale_factor;
                        sz = 69.2f * scale_factor;
                        break;
                    case 11:
                        sx = 36.7f * scale_factor;
                        sy = 77.9f * scale_factor;
                        sz = 172.8f * scale_factor;
                        break;
                    case 12:
                        sx = 100.9f * scale_factor;
                        sy = 108.5f * scale_factor;
                        sz = 90.8f * scale_factor;
                        break;
                }


                // 在原图上绘制
                g_gdrnet->draw3DBox(rgb, result, g_gdrnet->default_camera_params, sx, sy, sz);
                //g_gdrnet->draw3DAxes(rgb, result, g_gdrnet->default_camera_params, rect);

                cv::rectangle(rgb, rect, cv::Scalar(255, 0, 0), 1);
                //char text[32];
                //sprintf(text, "Obj:%d S:%.2f", obj_id, score);
                //cv::putText(rgb, text, cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX,0.5, cv::Scalar(0, 255, 255), 1);
            }
            else{
                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "infer出错");
            }
        }
    }

    // ==========================================
    // 4. 返回给 Java
    // ==========================================
    cv::Mat output;
    cv::cvtColor(rgb, output, cv::COLOR_BGR2RGBA);

    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapMethod = env->GetStaticMethodID(bitmapClass, "createBitmap",
                                                          "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jclass configClass = env->FindClass("android/graphics/Bitmap$Config");
    jfieldID argb8888Field = env->GetStaticFieldID(configClass, "ARGB_8888",
                                                   "Landroid/graphics/Bitmap$Config;");
    jobject argb8888Config = env->GetStaticObjectField(configClass, argb8888Field);

    jobject resultBitmap = env->CallStaticObjectMethod(bitmapClass, createBitmapMethod, output.cols,
                                                       output.rows, argb8888Config);
    if (!resultBitmap) return nullptr;

    void *resultPixels;
    if (AndroidBitmap_lockPixels(env, resultBitmap, &resultPixels) == 0) {
        memcpy(resultPixels, output.data, output.total() * output.elemSize());
        AndroidBitmap_unlockPixels(env, resultBitmap);
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "========== [ProcessImage END] ==========");
    return resultBitmap;
}

}