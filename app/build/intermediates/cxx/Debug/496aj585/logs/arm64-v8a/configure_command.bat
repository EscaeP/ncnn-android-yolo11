@echo off
"E:\\Android SDK\\cmake\\3.31.5\\bin\\cmake.exe" ^
  "-HF:\\graduationProject\\ncnn-android-yolo11\\app\\src\\main\\jni" ^
  "-DCMAKE_SYSTEM_NAME=Android" ^
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" ^
  "-DCMAKE_SYSTEM_VERSION=24" ^
  "-DANDROID_PLATFORM=android-24" ^
  "-DANDROID_ABI=arm64-v8a" ^
  "-DCMAKE_ANDROID_ARCH_ABI=arm64-v8a" ^
  "-DANDROID_NDK=E:\\Android SDK\\ndk\\29.0.14206865" ^
  "-DCMAKE_ANDROID_NDK=E:\\Android SDK\\ndk\\29.0.14206865" ^
  "-DCMAKE_TOOLCHAIN_FILE=E:\\Android SDK\\ndk\\29.0.14206865\\build\\cmake\\android.toolchain.cmake" ^
  "-DCMAKE_MAKE_PROGRAM=E:\\Android SDK\\cmake\\3.31.5\\bin\\ninja.exe" ^
  "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=F:\\graduationProject\\ncnn-android-yolo11\\app\\build\\intermediates\\cxx\\Debug\\496aj585\\obj\\arm64-v8a" ^
  "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=F:\\graduationProject\\ncnn-android-yolo11\\app\\build\\intermediates\\cxx\\Debug\\496aj585\\obj\\arm64-v8a" ^
  "-DCMAKE_BUILD_TYPE=Debug" ^
  "-BF:\\graduationProject\\ncnn-android-yolo11\\app\\.cxx\\Debug\\496aj585\\arm64-v8a" ^
  -GNinja ^
  "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON"
