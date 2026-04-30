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

#include "gdrnet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "json.hpp"

using json = nlohmann::json;

static json g_det_data;
static bool g_json_loaded = false;
static json g_models_info;
static bool g_models_info_loaded = false;

static GDRNet* g_gdrnet = 0;
static ncnn::Mutex lock;

static int draw_fps(cv::Mat& rgb)
{
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
    sprintf(text, "FPS: %.1f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    cv::putText(rgb, text, cv::Point(rgb.cols - label_size.width - 10, label_size.height + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

    return 0;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_gdrnet;
        g_gdrnet = 0;
    }

    ncnn::destroy_gpu_instance();
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_loadModel
        (JNIEnv *env, jobject thiz, jobject assetManager,
         jint modelid, jint cpugpu, jint task)
{
    __android_log_print(ANDROID_LOG_DEBUG,
                        "ncnn", "loadModel - modelid: %d, cpugpu: %d, task: %d"
            , modelid, cpugpu, task);

    {
        ncnn::MutexLockGuard g(lock)
        ;

        if
                (g_gdrnet) {
            delete
                    g_gdrnet;
            g_gdrnet =
                    0
                    ;
        }

        g_gdrnet =
                new
                        GDRNet();
        bool use_gpu = cpugpu == 1
        ;
        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

        int ret = g_gdrnet->load(mgr, "gdrnet.param", "gdrnet.bin"
                , use_gpu);
        if (ret != 0
                ) {
            delete
                    g_gdrnet;
            g_gdrnet =
                    0
                    ;
            return
                    JNI_FALSE;
        }

        // ==========================================
        // 加载 models_info.json
        // ==========================================
        AAsset* info_asset = AAssetManager_open(mgr,
                                                "models/models_info.json"
                , AASSET_MODE_BUFFER);
        if
                (info_asset) {
            const void
                    * data = AAsset_getBuffer(info_asset);
            off_t
                    len = AAsset_getLength(info_asset);
            std::string json_str((const char*)data, len)
            ;

            // 核心修改：使用无异常模式解析 (nullptr, false)
            g_models_info = json::parse(json_str,
                                        nullptr, false
            );

            if
                    (g_models_info.is_discarded()) {
                __android_log_print(ANDROID_LOG_ERROR,
                                    "ncnn", "Parse error: models_info.json format is invalid!"
                );
                g_models_info_loaded =
                        false
                        ;
            }
            else
            {
                g_models_info_loaded =
                        true
                        ;
                __android_log_print(ANDROID_LOG_DEBUG,
                                    "ncnn", "models_info.json loaded successfully!"
                );
            }
            AAsset_close(info_asset);
        }
        else
        {
            __android_log_print(ANDROID_LOG_ERROR,
                                "ncnn", "Failed to open models/models_info.json"
            );
        }
    }

    return
            JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_init(JNIEnv *env, jobject thiz, jobject assetManager,
                                            jint modelid, jint cpugpu) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "init");

    {
        ncnn::MutexLockGuard g(lock);

        if (g_gdrnet)
        {
            delete g_gdrnet;
            g_gdrnet = 0;
        }

        g_gdrnet = new GDRNet();

        bool use_gpu = cpugpu == 1;

        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

        int ret = g_gdrnet->load(mgr, "gdrnet.param", "gdrnet.bin", use_gpu);
        if (ret != 0)
        {
            delete g_gdrnet;
            g_gdrnet = 0;
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_uninit(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "uninit");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_gdrnet;
        g_gdrnet = 0;
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_openCamera(JNIEnv *env, jobject thiz, jint facing) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);
    // Camera functionality removed, this is just a stub
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_closeCamera(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");
    // Camera functionality removed, this is just a stub
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_setOutputWindow(JNIEnv *env, jobject thiz, jobject surface) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow");
    // Camera functionality removed, this is just a stub
    return JNI_TRUE;
}

JNIEXPORT jobject JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_processImage(JNIEnv *env, jobject thiz, jobject bitmap,
                                                    jstring imageName, jstring jsonContent) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "========== [ProcessImage START] ==========");

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

    AndroidBitmapInfo info;
    void *pixels;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) return nullptr;

    cv::Mat image(info.height, info.width, CV_8UC4, pixels);
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bitmap);

    const char *imageNameChars = env->GetStringUTFChars(imageName, nullptr);
    std::string nameStr(imageNameChars);
    env->ReleaseStringUTFChars(imageName, imageNameChars);

    std::string numPart;
    for (char c: nameStr) {
        if (isdigit(c)) numPart += c;
    }
    int img_id = numPart.empty() ? 0 : std::stoi(numPart);
    std::string dict_key = "2/" + std::to_string(img_id);

    const float SCORE_THR = 0.9f;
    std::vector<int> target_ids = {1, 5, 6, 8, 9, 10, 11, 12};

    if (g_gdrnet && g_json_loaded && g_det_data.contains(dict_key)) {
        float current_cx = 325.2611f;
        float current_cy = 242.04899f;
        float current_fx = 572.4114f;
        float current_fy = 573.57043f;
        g_gdrnet->setCameraParams(current_fx, current_fy, current_cx, current_cy);

        for (auto &item: g_det_data[dict_key]) {
            float score = item.value("score", 0.0f);
            int obj_id = item.value("obj_id", item.value("category_id", -1));

            if (score < SCORE_THR) continue;
            if (std::find(target_ids.begin(), target_ids.end(), obj_id) == target_ids.end()) continue;

            auto &b = item.contains("bbox_est") ? item["bbox_est"] : item["bbox"];
            cv::Rect2f rect(b[0], b[1], b[2], b[3]);

            // ==========================================
            // 从 JSON 动态获取当前物体的物理尺寸与不对称边界
            // ==========================================
            std::string obj_key = std::to_string(obj_id);
            if (!g_models_info_loaded || !g_models_info.contains(obj_key)) {
                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Missing models_info for obj: %d", obj_id);
                continue;
            }

            float scale_factor = 0.005f;

            // 获取物理尺寸 (送入网络推理)
            float size_x = g_models_info[obj_key]["size_x"].get<float>() * scale_factor;
            float size_y = g_models_info[obj_key]["size_y"].get<float>() * scale_factor;
            float size_z = g_models_info[obj_key]["size_z"].get<float>() * scale_factor;

            // 获取真实的不对称包围盒边界 (用于 3D 渲染)
            float min_x = g_models_info[obj_key]["min_x"].get<float>() * scale_factor;
            float min_y = g_models_info[obj_key]["min_y"].get<float>() * scale_factor;
            float min_z = g_models_info[obj_key]["min_z"].get<float>() * scale_factor;
            float max_x = min_x + size_x;
            float max_y = min_y + size_y;
            float max_z = min_z + size_z;

            PoseResult result;

            // 注意这里：inference 传入动态提取的 size
            int infer_ret = g_gdrnet->inference(rgb, rect, obj_id, size_x, size_y, size_z, result);

            if (infer_ret == 0) {
                // 注意这里：draw3DBox 传入真实的不对称 max 和 min
                g_gdrnet->draw3DBox(rgb, result, g_gdrnet->default_camera_params,
                                    min_x, max_x, min_y, max_y, min_z, max_z);

                cv::rectangle(rgb, rect, cv::Scalar(255, 0, 0), 1);
            } else {
                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "infer出错, obj_id: %d", obj_id);
            }
        }
    }

    cv::Mat rgba;
    cv::cvtColor(rgb, rgba, cv::COLOR_BGR2RGBA);

    jclass bitmapConfigClass = env->FindClass("android/graphics/Bitmap$Config");
    jfieldID argb8888Field = env->GetStaticFieldID(bitmapConfigClass, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    jobject bitmapConfig = env->GetStaticObjectField(bitmapConfigClass, argb8888Field);

    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapMethod = env->GetStaticMethodID(bitmapClass, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jobject newBitmap = env->CallStaticObjectMethod(bitmapClass, createBitmapMethod, rgba.cols, rgba.rows, bitmapConfig);

    void *newPixels;
    if (AndroidBitmap_lockPixels(env, newBitmap, &newPixels) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to lock new bitmap pixels");
        return nullptr;
    }

    cv::Mat outBitmapMat(rgba.rows, rgba.cols, CV_8UC4, newPixels);
    rgba.copyTo(outBitmapMat);

    AndroidBitmap_unlockPixels(env, newBitmap);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "========== [ProcessImage END] ==========");

    return newBitmap;
}

}
