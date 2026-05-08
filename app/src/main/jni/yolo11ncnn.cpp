#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/bitmap.h>
#include <fstream>
#include <string>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <map>

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

struct TimingStats {
    double total_time_ms = 0;
    double preprocess_time_ms = 0;
    double inference_time_ms = 0;
    double postprocess_time_ms = 0;
    int processed_frames = 0;
    int success_count = 0;
};

static TimingStats g_timing_stats;

#ifdef __ANDROID__
#include <unistd.h>
#include <sys/sysinfo.h>

static long getTotalMemory() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram * info.mem_unit;
    }
    return 0;
}

static long getFreeMemory() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    }
    return 0;
}

static long getUsedMemory() {
    return getTotalMemory() - getFreeMemory();
}
#endif

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
        ncnn::MutexLockGuard g(lock);

        if (g_gdrnet) {
            delete g_gdrnet;
            g_gdrnet = 0;
        }

        g_gdrnet = new GDRNet();
        bool use_gpu = cpugpu == 1;
        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

        int ret = g_gdrnet->load(mgr, "gdrnet.param", "gdrnet.bin", use_gpu);
        if (ret != 0) {
            delete g_gdrnet;
            g_gdrnet = 0;
            return JNI_FALSE;
        }

        AAsset* info_asset = AAssetManager_open(mgr,
                                                "models/models_info.json"
                , AASSET_MODE_BUFFER);
        if (info_asset) {
            const void * data = AAsset_getBuffer(info_asset);
            off_t len = AAsset_getLength(info_asset);
            std::string json_str((const char*)data, len);

            g_models_info = json::parse(json_str, nullptr, false);

            if (g_models_info.is_discarded()) {
                __android_log_print(ANDROID_LOG_ERROR,
                                    "ncnn", "Parse error: models_info.json format is invalid!");
                g_models_info_loaded = false;
            }
            else {
                g_models_info_loaded = true;
                __android_log_print(ANDROID_LOG_DEBUG,
                                    "ncnn", "models_info.json loaded successfully!");
            }
            AAsset_close(info_asset);
        }
        else {
            __android_log_print(ANDROID_LOG_ERROR,
                                "ncnn", "Failed to open models/models_info.json");
        }
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_init(JNIEnv *env, jobject thiz, jobject assetManager,
                                            jint modelid, jint cpugpu) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "init");

    {
        ncnn::MutexLockGuard g(lock);

        if (g_gdrnet) {
            delete g_gdrnet;
            g_gdrnet = 0;
        }

        g_gdrnet = new GDRNet();

        bool use_gpu = cpugpu == 1;

        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

        int ret = g_gdrnet->load(mgr, "gdrnet.param", "gdrnet.bin", use_gpu);
        if (ret != 0) {
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
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_closeCamera(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_setOutputWindow(JNIEnv *env, jobject thiz, jobject surface) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow");
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

            std::string obj_key = std::to_string(obj_id);
            if (!g_models_info_loaded || !g_models_info.contains(obj_key)) {
                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Missing models_info for obj: %d", obj_id);
                continue;
            }

            float scale_factor = 0.005f;

            float size_x = g_models_info[obj_key]["size_x"].get<float>() * scale_factor;
            float size_y = g_models_info[obj_key]["size_y"].get<float>() * scale_factor;
            float size_z = g_models_info[obj_key]["size_z"].get<float>() * scale_factor;

            float min_x = g_models_info[obj_key]["min_x"].get<float>() * scale_factor;
            float min_y = g_models_info[obj_key]["min_y"].get<float>() * scale_factor;
            float min_z = g_models_info[obj_key]["min_z"].get<float>() * scale_factor;
            float max_x = min_x + size_x;
            float max_y = min_y + size_y;
            float max_z = min_z + size_z;

            PoseResult result = {};

            int infer_ret = g_gdrnet->inference(rgb, rect, obj_id, size_x, size_y, size_z, result);

            if (infer_ret == 0) {
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

JNIEXPORT jstring JNICALL
Java_com_tencent_yolo11ncnn_YOLO11Ncnn_batchProcessImages(JNIEnv *env, jobject thiz,
        jobjectArray bitmaps, jobjectArray imageNames, jstring jsonContent) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "========== [BatchProcessImages START] ==========");

    double batch_start_time = ncnn::get_current_time();
    long mem_before = getUsedMemory();

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

    if (!g_json_loaded) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "JSON data not loaded!");
        return env->NewStringUTF("{\"error\": \"JSON data not loaded\"}");
    }

    json result_json = json::array();
    json timing_stats;
    jsize numImages = env->GetArrayLength(bitmaps);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Processing %d images", numImages);

    double total_preprocess_ms = 0;
    double total_inference_ms = 0;
    double total_postprocess_ms = 0;
    int total_objects = 0;
    int success_objects = 0;

    for (int i = 0; i < numImages; i++) {
        jobject bitmapObj = env->GetObjectArrayElement(bitmaps, i);
        jstring nameObj = (jstring)env->GetObjectArrayElement(imageNames, i);

        const char *nameChars = env->GetStringUTFChars(nameObj, nullptr);
        std::string imageName(nameChars);
        env->ReleaseStringUTFChars(nameObj, nameChars);

        double image_start_time = ncnn::get_current_time();
        double preprocess_start = ncnn::get_current_time();

        AndroidBitmapInfo info;
        void *pixels;
        if (AndroidBitmap_getInfo(env, bitmapObj, &info) < 0) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to get bitmap info for image %d", i);
            continue;
        }
        if (AndroidBitmap_lockPixels(env, bitmapObj, &pixels) < 0) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to lock pixels for image %d", i);
            continue;
        }

        cv::Mat image(info.height, info.width, CV_8UC4, pixels);
        cv::Mat rgb;
        cv::cvtColor(image, rgb, cv::COLOR_RGBA2BGR);
        AndroidBitmap_unlockPixels(env, bitmapObj);

        std::string numPart;
        for (char c : imageName) {
            if (isdigit(c)) numPart += c;
        }
        int img_id = numPart.empty() ? 0 : std::stoi(numPart);
        std::string dict_key = "2/" + std::to_string(img_id);

        double preprocess_end = ncnn::get_current_time();
        double preprocess_ms = (preprocess_end - preprocess_start);
        total_preprocess_ms += preprocess_ms;

        json image_result;
        image_result["image_name"] = imageName;
        image_result["image_id"] = img_id;
        image_result["width"] = info.width;
        image_result["height"] = info.height;
        json pose_results = json::array();

        double inference_start = ncnn::get_current_time();

        if (g_gdrnet && g_det_data.contains(dict_key)) {
            float current_cx = 325.2611f;
            float current_cy = 242.04899f;
            float current_fx = 572.4114f;
            float current_fy = 573.57043f;
            g_gdrnet->setCameraParams(current_fx, current_fy, current_cx, current_cy);

            const float SCORE_THR = 0.9f;
            std::vector<int> target_ids = {1, 5, 6, 8, 9, 10, 11, 12};

            for (auto &item : g_det_data[dict_key]) {
                float score = item.value("score", 0.0f);
                int obj_id = item.value("obj_id", item.value("category_id", -1));

                if (score < SCORE_THR) continue;
                if (std::find(target_ids.begin(), target_ids.end(), obj_id) == target_ids.end()) continue;

                auto &b = item.contains("bbox_est") ? item["bbox_est"] : item["bbox"];
                cv::Rect2f rect(b[0], b[1], b[2], b[3]);

                std::string obj_key = std::to_string(obj_id);
                if (!g_models_info_loaded || !g_models_info.contains(obj_key)) {
                    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Missing models_info for obj: %d", obj_id);
                    continue;
                }

                float scale_factor = 0.005f;
                float size_x = g_models_info[obj_key]["size_x"].get<float>() * scale_factor;
                float size_y = g_models_info[obj_key]["size_y"].get<float>() * scale_factor;
                float size_z = g_models_info[obj_key]["size_z"].get<float>() * scale_factor;

                float min_x = g_models_info[obj_key]["min_x"].get<float>() * scale_factor;
                float min_y = g_models_info[obj_key]["min_y"].get<float>() * scale_factor;
                float min_z = g_models_info[obj_key]["min_z"].get<float>() * scale_factor;
                float max_x = min_x + size_x;
                float max_y = min_y + size_y;
                float max_z = min_z + size_z;

                double obj_infer_start = ncnn::get_current_time();
                PoseResult result = {};
                int infer_ret = g_gdrnet->inference(rgb, rect, obj_id, size_x, size_y, size_z, result);
                double obj_infer_end = ncnn::get_current_time();
                double obj_infer_ms = (obj_infer_end - obj_infer_start);

                __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                    "Image %d, Obj %d inference time: %.2f ms", i, obj_id, obj_infer_ms);

                total_objects++;
                if (infer_ret == 0) success_objects++;

                json pose_data;
                pose_data["obj_id"] = obj_id;
                pose_data["score"] = score;
                pose_data["bbox"] = {b[0].get<float>(), b[1].get<float>(), b[2].get<float>(), b[3].get<float>()};
                pose_data["size"] = {size_x, size_y, size_z};
                pose_data["bbox_3d"] = {
                    {"min_x", min_x}, {"min_y", min_y}, {"min_z", min_z},
                    {"max_x", max_x}, {"max_y", max_y}, {"max_z", max_z}
                };

                json rotation = json::array();
                for (int j = 0; j < 9; j++) rotation.push_back(result.rotation[j]);
                pose_data["rotation"] = rotation;

                json translation = json::array();
                for (int j = 0; j < 3; j++) translation.push_back(result.translation[j]);
                pose_data["translation"] = translation;

                json scale = json::array();
                for (int j = 0; j < 3; j++) scale.push_back(result.scale[j]);
                pose_data["estimated_scale"] = scale;

                pose_data["inference_success"] = (infer_ret == 0);
                pose_data["inference_time_ms"] = obj_infer_ms;

                pose_results.push_back(pose_data);
            }
        }

        double inference_end = ncnn::get_current_time();
        double inference_ms = (inference_end - inference_start);
        total_inference_ms += inference_ms;

        double postprocess_start = ncnn::get_current_time();

        image_result["pose_results"] = pose_results;
        image_result["timing"] = {
            {"preprocess_ms", preprocess_ms},
            {"inference_ms", inference_ms}
        };
        result_json.push_back(image_result);

        env->DeleteLocalRef(bitmapObj);
        env->DeleteLocalRef(nameObj);

        double postprocess_end = ncnn::get_current_time();
        double postprocess_ms = (postprocess_end - postprocess_start);
        total_postprocess_ms += postprocess_ms;

        double image_end_time = ncnn::get_current_time();
        double image_total_ms = (image_end_time - image_start_time);

        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
            "Image %d (%s) - Total: %.2f ms, Preprocess: %.2f ms, Inference: %.2f ms, Postprocess: %.2f ms",
            i, imageName.c_str(), image_total_ms, preprocess_ms, inference_ms, postprocess_ms);
    }

    double batch_end_time = ncnn::get_current_time();
    double batch_total_ms = (batch_end_time - batch_start_time);
    long mem_after = getUsedMemory();
    long mem_diff = mem_after - mem_before;

    timing_stats["total_images"] = numImages;
    timing_stats["total_objects_processed"] = total_objects;
    timing_stats["success_objects"] = success_objects;
    timing_stats["total_time_ms"] = batch_total_ms;
    timing_stats["avg_time_per_image_ms"] = numImages > 0 ? batch_total_ms / numImages : 0;
    timing_stats["preprocess_time_ms"] = total_preprocess_ms;
    timing_stats["inference_time_ms"] = total_inference_ms;
    timing_stats["postprocess_time_ms"] = total_postprocess_ms;
    timing_stats["memory_usage"] = {
        {"before_bytes", mem_before},
        {"after_bytes", mem_after},
        {"diff_bytes", mem_diff},
        {"before_mb", mem_before / (1024.0 * 1024.0)},
        {"after_mb", mem_after / (1024.0 * 1024.0)},
        {"diff_mb", mem_diff / (1024.0 * 1024.0)}
    };

    json final_result;
    final_result["results"] = result_json;
    final_result["statistics"] = timing_stats;

    __android_log_print(ANDROID_LOG_INFO, "ncnn", "=== 批量处理完成 ===");
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "处理图片数: %d", numImages);
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "总耗时: %.2f ms", batch_total_ms);
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "内存变化: %.2f MB", mem_diff / (1024.0 * 1024.0));
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "成功推理物体数: %d/%d", success_objects, total_objects);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "========== [BatchProcessImages END] ==========");

    std::string final_json_str = final_result.dump(2);
    return env->NewStringUTF(final_json_str.c_str());
}

}
