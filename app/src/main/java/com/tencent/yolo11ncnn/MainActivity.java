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

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;
import java.io.InputStream;
import java.io.IOException;
import android.content.Context;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;
    public static final int REQUEST_STORAGE = 101;

    private YOLO11Ncnn yolo11ncnn = new YOLO11Ncnn();
    private int facing = 0;

    private Spinner spinnerTask;
    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_task = 0;
    private int current_model = 0;
    private int current_cpugpu = 0;
    private boolean isShowingImage = false; // 标记是否正在显示图片

    private SurfaceView cameraView;
    private ImageView imageViewResult;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        imageViewResult = (ImageView) findViewById(R.id.imageViewResult);

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                // 显示SurfaceView，隐藏ImageView
                cameraView.setVisibility(View.VISIBLE);
                imageViewResult.setVisibility(View.GONE);
                isShowingImage = false; // 标记不在显示图片

                int new_facing = 1 - facing;

                yolo11ncnn.closeCamera();

                yolo11ncnn.openCamera(new_facing);

                facing = new_facing;
            }
        });

        Button buttonPause = (Button) findViewById(R.id.buttonPause);
        buttonPause.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                // 显示SurfaceView，隐藏ImageView
                cameraView.setVisibility(View.VISIBLE);
                imageViewResult.setVisibility(View.GONE);
                isShowingImage = false; // 标记不在显示图片
                yolo11ncnn.togglePause();
            }
        });

        Button buttonUploadImage = (Button) findViewById(R.id.buttonUploadImage);
        buttonUploadImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                // 关闭相机，准备处理图片
                yolo11ncnn.closeCamera();
                // 打开图片选择器
                Intent intent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, 1);
            }
        });

        spinnerTask = (Spinner) findViewById(R.id.spinnerTask);
        spinnerTask.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_task)
                {
                    current_task = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_model)
                {
                    current_model = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_cpugpu)
                {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        reload();
    }

    private void reload()
    {
        Log.d("MainActivity", "Loading model: task=" + current_task + ", model=" + current_model + ", cpugpu=" + current_cpugpu);
        boolean ret_init = yolo11ncnn.loadModel(getAssets(), current_task, current_model, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "yolo11ncnn loadModel failed");
        } else {
            Log.d("MainActivity", "Model loaded successfully");
        }
    }
    private String loadJSONFromAsset(Context context, String fileName) {
        String json = null;
        try {
            InputStream is = context.getAssets().open(fileName);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            json = new String(buffer, "UTF-8");
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
        return json;
    }
    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        yolo11ncnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_STORAGE);
        }

        // 只有在不显示图片时才打开相机
        if (!isShowingImage) {
            yolo11ncnn.openCamera(facing);
        }
    }

    @Override
    public void onPause()
    {
        super.onPause();

        yolo11ncnn.closeCamera();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            try {
                // 提取并打印照片名称
                String imageName = getFileName(selectedImage);
                Log.d("MainActivity", "Selected image: " + selectedImage);
                Log.d("MainActivity", "Image name: " + imageName);
                // 读取图片
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                Log.d("MainActivity", "Bitmap loaded: " + bitmap.getWidth() + "x" + bitmap.getHeight());
                // 处理图片
                String jsonString = loadJSONFromAsset(this, "testbbox.json");
                Bitmap resultBitmap = yolo11ncnn.processImage(bitmap, imageName,jsonString);
                Log.d("MainActivity", "Processed bitmap: " + (resultBitmap != null ? resultBitmap.getWidth() + "x" + resultBitmap.getHeight() : "null"));
                // 显示处理后的图片
                if (resultBitmap != null) {
                    Log.d("MainActivity", "Showing result image");
                    // 隐藏SurfaceView，显示ImageView
                    cameraView.setVisibility(View.GONE);
                    imageViewResult.setImageBitmap(resultBitmap);
                    imageViewResult.setVisibility(View.VISIBLE);
                    isShowingImage = true; // 标记正在显示图片
                    Log.d("MainActivity", "ImageView visibility: " + imageViewResult.getVisibility() + ", isShowingImage: " + isShowingImage);
                } else {
                    Log.e("MainActivity", "Process image returned null");
                    // 显示错误信息
                    Toast.makeText(this, "处理图片失败，请重试", Toast.LENGTH_SHORT).show();
                    // 恢复相机预览
                    cameraView.setVisibility(View.VISIBLE);
                    imageViewResult.setVisibility(View.GONE);
                    isShowingImage = false; // 标记不在显示图片
                }
            } catch (Exception e) {
                Log.e("MainActivity", "Error processing image", e);
                e.printStackTrace();
                // 显示错误信息
                Toast.makeText(this, "处理图片失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                // 恢复相机预览
                cameraView.setVisibility(View.VISIBLE);
                imageViewResult.setVisibility(View.GONE);
                isShowingImage = false; // 标记不在显示图片
            }
        }
    }
    
    // 从Uri中提取文件名
    private String getFileName(Uri uri) {
        String result = null;
        if (uri.getScheme().equals("content")) {
            try (Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    int nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                    if (nameIndex != -1) {
                        result = cursor.getString(nameIndex);
                    }
                }
            }
        }
        if (result == null) {
            result = uri.getPath();
            int cut = result.lastIndexOf('/');
            if (cut != -1) {
                result = result.substring(cut + 1);
            }
        }
        return result;
    }
}
