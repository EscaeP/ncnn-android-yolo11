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
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import java.io.File;
import java.io.InputStream;
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import android.content.Context;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

public class MainActivity extends Activity {
    public static final int REQUEST_STORAGE = 101;

    private YOLO11Ncnn yolo11ncnn = new YOLO11Ncnn();

    private ImageView imageViewResult;
    private Bitmap currentResultBitmap;

    private static final int REQUEST_MULTIPLE_IMAGES = 2;
    private static final int REQUEST_WRITE_STORAGE = 102;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        imageViewResult = (ImageView) findViewById(R.id.imageViewResult);

        Button buttonUploadImage = (Button) findViewById(R.id.buttonUploadImage);
        buttonUploadImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent intent = new Intent(Intent.ACTION_PICK, 
                    android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, 1);
            }
        });

        Button buttonBatchUpload = (Button) findViewById(R.id.buttonBatchUpload);
        buttonBatchUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (ContextCompat.checkSelfPermission(getApplicationContext(), 
                    Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED) {
                    ActivityCompat.requestPermissions(MainActivity.this, 
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_WRITE_STORAGE);
                    return;
                }
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                startActivityForResult(intent, REQUEST_MULTIPLE_IMAGES);
            }
        });

        Button buttonSaveImage = (Button) findViewById(R.id.buttonSaveImage);
        buttonSaveImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (currentResultBitmap == null) {
                    Toast.makeText(MainActivity.this, "没有可保存的图片", Toast.LENGTH_SHORT).show();
                    return;
                }

                if (ContextCompat.checkSelfPermission(getApplicationContext(), 
                    Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED) {
                    ActivityCompat.requestPermissions(MainActivity.this, 
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_STORAGE);
                    return;
                }

                String savePath = getExternalFilesDir(Environment.DIRECTORY_PICTURES) 
                    + "/YOLO11Pose/result_" + System.currentTimeMillis() + ".jpg";
                boolean success = yolo11ncnn.saveImage(currentResultBitmap, savePath);

                if (success) {
                    Toast.makeText(MainActivity.this, "图片保存成功!\n" + savePath, Toast.LENGTH_LONG).show();
                    Log.d("MainActivity", "Image saved to: " + savePath);
                } else {
                    Toast.makeText(MainActivity.this, "保存失败", Toast.LENGTH_SHORT).show();
                }
            }
        });

        yolo11ncnn.loadModel(getAssets(), 0, 0, 0);
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
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 1 && resultCode == RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            try {
                String imageName = getFileName(selectedImage);
                Log.d("MainActivity", "Selected image: " + selectedImage);
                Log.d("MainActivity", "Image name: " + imageName);
                
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                Log.d("MainActivity", "Bitmap loaded: " + bitmap.getWidth() + "x" + bitmap.getHeight());
                
                String jsonString = loadJSONFromAsset(this, "testbbox.json");
                Bitmap resultBitmap = yolo11ncnn.processImage(bitmap, imageName, jsonString);
                
                Log.d("MainActivity", "Processed bitmap: " + (resultBitmap != null ? 
                    resultBitmap.getWidth() + "x" + resultBitmap.getHeight() : "null"));

                if (resultBitmap != null) {
                    imageViewResult.setImageBitmap(resultBitmap);
                    imageViewResult.setVisibility(View.VISIBLE);
                    currentResultBitmap = resultBitmap;
                } else {
                    Toast.makeText(this, "处理图片失败，请重试", Toast.LENGTH_SHORT).show();
                }
            } catch (Exception e) {
                Log.e("MainActivity", "Error processing image", e);
                Toast.makeText(this, "处理图片失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        } else if (requestCode == REQUEST_MULTIPLE_IMAGES && resultCode == RESULT_OK && data != null) {
            try {
                List<Bitmap> bitmapList = new ArrayList<>();
                List<String> nameList = new ArrayList<>();

                if (data.getClipData() != null) {
                    int count = data.getClipData().getItemCount();
                    Log.d("MainActivity", "Selected " + count + " images");

                    for (int i = 0; i < count; i++) {
                        Uri uri = data.getClipData().getItemAt(i).getUri();
                        String name = getFileName(uri);
                        Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                        bitmapList.add(bitmap);
                        nameList.add(name);
                    }
                } else if (data.getData() != null) {
                    Uri uri = data.getData();
                    String name = getFileName(uri);
                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    bitmapList.add(bitmap);
                    nameList.add(name);
                }

                if (bitmapList.isEmpty()) {
                    Toast.makeText(this, "未选择任何图片", Toast.LENGTH_SHORT).show();
                    return;
                }

                Bitmap[] bitmaps = bitmapList.toArray(new Bitmap[0]);
                String[] names = nameList.toArray(new String[0]);
                String jsonString = loadJSONFromAsset(this, "testbbox.json");

                Log.d("MainActivity", "Starting batch processing...");
                String resultJson = yolo11ncnn.batchProcessImages(bitmaps, names, jsonString);
                Log.d("MainActivity", "Batch processing completed");

                if (resultJson != null && !resultJson.isEmpty()) {
                    String savePath = saveJsonToFile(resultJson, "pose_results.json");
                    if (savePath != null) {
                        Toast.makeText(this, "批量处理完成!\n结果已保存: " + savePath, Toast.LENGTH_LONG).show();
                    } else {
                        Toast.makeText(this, "处理完成，但保存失败", Toast.LENGTH_SHORT).show();
                    }
                } else {
                    Toast.makeText(this, "批量处理失败", Toast.LENGTH_SHORT).show();
                }

            } catch (Exception e) {
                Log.e("MainActivity", "Error in batch processing", e);
                Toast.makeText(this, "批量处理失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        }
    }

    private String saveJsonToFile(String jsonContent, String fileName) {
        try {
            File dir = new File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "YOLO11Pose");
            if (!dir.exists()) {
                dir.mkdirs();
            }

            File file = new File(dir, fileName);
            FileOutputStream fos = new FileOutputStream(file);
            OutputStreamWriter writer = new OutputStreamWriter(fos, "UTF-8");
            writer.write(jsonContent);
            writer.close();
            fos.close();

            Log.d("MainActivity", "JSON saved to: " + file.getAbsolutePath());
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e("MainActivity", "Error saving JSON file", e);
            return null;
        }
    }

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
