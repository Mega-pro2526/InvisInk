# 📱 InvisInk Android App

Complete Android application implementation of the InvisInk air-drawing calculator.

## 📁 Project Structure

```
android_app/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/invisink/
│   │   │   │   ├── MainActivity.kt          # Main activity
│   │   │   │   ├── HandTracker.kt           # MediaPipe hand tracking
│   │   │   │   ├── SymbolClassifier.kt     # TensorFlow Lite inference
│   │   │   │   └── DrawingCanvas.kt         # Custom drawing view
│   │   │   ├── res/
│   │   │   │   ├── layout/
│   │   │   │   │   └── activity_main.xml    # Main UI layout
│   │   │   │   ├── values/
│   │   │   │   │   ├── strings.xml
│   │   │   │   │   ├── colors.xml
│   │   │   │   │   └── themes.xml
│   │   │   └── AndroidManifest.xml
│   │   └── assets/                          # Put model files here
│   │       ├── invisink_model.tflite         # Your converted model
│   │       └── hand_landmarker.task          # MediaPipe model
│   └── build.gradle.kts
├── build.gradle.kts
├── settings.gradle.kts
└── gradle.properties
```

## 🚀 Setup Instructions

### Step 1: Convert Your Model

First, convert your Keras model to TensorFlow Lite:

```bash
python convert_model_to_tflite.py
```

This creates `invisink_model.tflite` - copy it to `android_app/app/src/main/assets/`

### Step 2: Download MediaPipe Hand Landmarker Model

1. Visit: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
2. Download: `hand_landmarker.task`
3. Place in: `android_app/app/src/main/assets/`

### Step 3: Open in Android Studio

1. Open Android Studio
2. File → Open → Select `android_app` folder
3. Let Gradle sync (this may take a few minutes)

### Step 4: Build and Run

1. Connect an Android device or start an emulator
2. Click Run (▶️) or press Shift+F10
3. Grant camera permission when prompted

## 📋 Requirements

- **Android Studio**: Latest version (Hedgehog or newer)
- **Min SDK**: 24 (Android 7.0)
- **Target SDK**: 34 (Android 14)
- **Kotlin**: 1.9.20+
- **Gradle**: 8.2+

## 🔧 Key Features

### MainActivity.kt
- Camera setup using CameraX
- Hand tracking integration
- Gesture recognition logic (matches Python app)
- Symbol classification
- Expression solving

### HandTracker.kt
- MediaPipe hand landmark detection
- Gesture recognition (FINGERTIP, FIST, THUMBS_UP, OPEN_HAND)
- Real-time hand tracking

### SymbolClassifier.kt
- TensorFlow Lite model loading
- Image preprocessing (30x30 grayscale)
- Symbol classification (16 classes)

### DrawingCanvas.kt
- Custom view for drawing fingertip path
- Overlay on camera preview
- Symbol bitmap extraction

## 🎮 Gesture Controls

- **👆 Index Finger Extended**: Draw symbols in the air
- **✊ Closed Fist**: Recognize the drawn symbol (wait 1.5s between recognitions)
- **👍 Thumbs Up**: Solve the equation and display result
- **✋ Open Hand**: Clear the canvas and reset

## 📝 Notes

1. **Model File**: Ensure `invisink_model.tflite` is in `app/src/main/assets/`
2. **MediaPipe Model**: Ensure `hand_landmarker.task` is in `app/src/main/assets/`
3. **Permissions**: Camera permission is requested at runtime
4. **Performance**: Uses GPU acceleration for MediaPipe (if available)

## 🐛 Troubleshooting

### Model Not Found
- Ensure `invisink_model.tflite` exists in `app/src/main/assets/`
- Check file name matches exactly

### Hand Not Detected
- Ensure good lighting
- Position hand 2-3 feet from camera
- Show palm clearly to camera

### Build Errors
- Sync Gradle: File → Sync Project with Gradle Files
- Clean project: Build → Clean Project
- Invalidate caches: File → Invalidate Caches

## 📚 Dependencies

- **CameraX**: Camera API
- **MediaPipe Tasks Vision**: Hand tracking
- **TensorFlow Lite**: Model inference
- **Kotlin Coroutines**: Async operations

## 🔄 Matching Python App Logic

This Android app mirrors the functionality of `invisink_app.py`:

- ✅ Same gesture recognition logic
- ✅ Same symbol classification (16 classes)
- ✅ Same debounce timing (1.5 seconds)
- ✅ Same expression solving
- ✅ Same UI feedback

## 📱 Testing

Test on a real device for best results:
- Camera access works better on real devices
- Performance is more accurate
- Gesture recognition is more reliable

## 🚀 Next Steps

1. Convert your model: `python convert_model_to_tflite.py`
2. Copy model to assets folder
3. Download MediaPipe model
4. Open in Android Studio
5. Build and run!

---

**Ready to build your Android app!** 🎉

