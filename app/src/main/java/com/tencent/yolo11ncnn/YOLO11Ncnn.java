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

package com.tencent.yolo11ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.CompressFormat;
import android.view.Surface;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class YOLO11Ncnn
{
    public native boolean loadModel(AssetManager mgr, int taskid, int modelid, int cpugpu);
    public native boolean openCamera(int facing);
    public native boolean closeCamera();
    public native boolean setOutputWindow(Surface surface);
    public native boolean togglePause();
    public native Bitmap processImage(Bitmap bitmap, String imageName, String jsonContent);
    public native String batchProcessImages(Bitmap[] bitmaps, String[] imageNames, String jsonContent);

    public boolean saveImage(Bitmap bitmap, String savePath) {
        if (bitmap == null || savePath == null) {
            return false;
        }
        
        File file = new File(savePath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        try (FileOutputStream fos = new FileOutputStream(file)) {
            return bitmap.compress(CompressFormat.JPEG, 90, fos);
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    static {
        System.loadLibrary("yolo11ncnn");
    }
}
