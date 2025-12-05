# 📱 InvisInk Android App Development Guide

This guide explains how to convert your Python InvisInk application to an Android app.

## 🎯 Overview

Your `invisink_app.py` uses:
- **OpenCV** - Camera capture and image processing
- **MediaPipe** - Hand tracking and gesture recognition
- **TensorFlow/Keras** - CNN model for symbol classification
- **NumPy** - Array operations

For Android, you have **three main options**. Each has different complexity levels.

---

## 🚀 Option 1: Native Android Development (RECOMMENDED) ⭐

**Best for**: Performance, native feel, professional apps  
**Difficulty**: Medium-High (requires learning Kotlin/Java)  
**Time**: 2-4 weeks for experienced developers

### Why This Approach?

- ✅ **Best Performance**: Native code runs faster
- ✅ **Official SDKs**: MediaPipe and TensorFlow Lite have excellent Android support
- ✅ **Smaller APK**: Optimized libraries
- ✅ **Better Battery Life**: Optimized for mobile
- ✅ **Access to All Android Features**: Camera2 API, GPU acceleration, etc.

### Required Tools

1. **Android Studio** (Latest version)
   - Download: https://developer.android.com/studio
   - Includes Android SDK, emulator, and build tools

2. **Java Development Kit (JDK) 17+**
   - Usually included with Android Studio

3. **Model Conversion Tool**: Convert `.h5` to TensorFlow Lite
   - TensorFlow Lite Converter

### Step-by-Step Guide

#### Step 1: Convert Your Model to TensorFlow Lite

Your current model is `invisink_model.h5`. You need to convert it to `.tflite` format for Android.

**Create a conversion script** (`convert_model_to_tflite.py`):

```python
import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('invisink_model.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Optimize for size/performance
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save the converted model
with open('invisink_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted successfully! File: invisink_model.tflite")
```

**Run the conversion**:
```bash
python convert_model_to_tflite.py
```

This creates `invisink_model.tflite` - much smaller and optimized for mobile.

#### Step 2: Set Up Android Project

1. **Open Android Studio** → New Project
2. **Choose**: Empty Activity
3. **Language**: Kotlin (recommended) or Java
4. **Minimum SDK**: API 24 (Android 7.0) or higher
5. **Project Name**: InvisInk

#### Step 3: Add Dependencies

In `app/build.gradle.kts` (or `build.gradle`), add:

```kotlin
dependencies {
    // MediaPipe for hand tracking
    implementation("com.google.mediapipe:tasks-vision:0.10.8")
    
    // TensorFlow Lite for model inference
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-metadata:0.4.4")
    
    // CameraX for camera access
    implementation("androidx.camera:camera-core:1.3.0")
    implementation("androidx.camera:camera-camera2:1.3.0")
    implementation("androidx.camera:camera-lifecycle:1.3.0")
    implementation("androidx.camera:camera-view:1.3.0")
    
    // Coroutines for async operations
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

#### Step 4: Add Permissions

In `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera" android:required="true" />
<uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />
```

#### Step 5: Copy Model to Assets

1. Create folder: `app/src/main/assets/`
2. Copy `invisink_model.tflite` to this folder

#### Step 6: Implement Core Components

You'll need to create these files:

**A. Hand Tracking with MediaPipe** (`HandTracker.kt`):
```kotlin
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerOptions
import android.graphics.Bitmap

class HandTracker(context: Context) {
    private val handLandmarker: HandLandmarker
    
    init {
        val options = HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setMinHandDetectionConfidence(0.7f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result, image ->
                // Process hand landmarks
                processHandLandmarks(result, image)
            }
            .build()
        
        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }
    
    fun detectAsync(bitmap: Bitmap, timestamp: Long) {
        val mpImage = BitmapImageBuilder(bitmap).build()
        handLandmarker.detectAsync(mpImage, timestamp)
    }
    
    private fun processHandLandmarks(result: HandLandmarkerResult, image: MPImage) {
        // Extract landmarks and detect gestures
        // Similar to your Python get_gesture() function
    }
}
```

**B. Symbol Classification with TensorFlow Lite** (`SymbolClassifier.kt`):
```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SymbolClassifier(context: Context) {
    private val interpreter: Interpreter
    private val classes = arrayOf("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "(", ")")
    
    init {
        val modelFile = loadModelFile(context, "invisink_model.tflite")
        val options = Interpreter.Options()
        interpreter = Interpreter(modelFile, options)
    }
    
    fun classify(symbolBitmap: Bitmap): String {
        // Preprocess: Resize to 30x30, convert to grayscale, normalize
        val processedBitmap = preprocessImage(symbolBitmap)
        val inputBuffer = convertBitmapToByteBuffer(processedBitmap)
        
        // Run inference
        val outputBuffer = ByteBuffer.allocateDirect(16 * 4) // 16 classes * 4 bytes (float)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        interpreter.run(inputBuffer, outputBuffer)
        
        // Get prediction
        val predictions = FloatArray(16)
        outputBuffer.rewind()
        outputBuffer.asFloatBuffer().get(predictions)
        
        val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: 0
        return classes[maxIndex]
    }
    
    private fun preprocessImage(bitmap: Bitmap): Bitmap {
        // Resize to 30x30, convert to grayscale
        val resized = Bitmap.createScaledBitmap(bitmap, 30, 30, true)
        // Convert to grayscale if needed
        return resized
    }
}
```

**C. Main Activity with Camera** (`MainActivity.kt`):
```kotlin
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var handTracker: HandTracker
    private lateinit var symbolClassifier: SymbolClassifier
    
    private var drawingPoints = mutableListOf<Point>()
    private var currentExpression = ""
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        previewView = findViewById(R.id.previewView)
        handTracker = HandTracker(this)
        symbolClassifier = SymbolClassifier(this)
        
        startCamera()
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindCamera(cameraProvider)
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun bindCamera(cameraProvider: ProcessCameraProvider) {
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }
        
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { imageProxy ->
                    processImage(imageProxy)
                }
            }
        
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
    }
    
    private fun processImage(imageProxy: ImageProxy) {
        val bitmap = imageProxyToBitmap(imageProxy)
        val timestamp = System.currentTimeMillis()
        handTracker.detectAsync(bitmap, timestamp)
        imageProxy.close()
    }
}
```

#### Step 7: Download MediaPipe Hand Landmarker Model

You need to download the hand landmarker model file:
- Visit: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- Download: `hand_landmarker.task`
- Place in: `app/src/main/assets/`

#### Step 8: UI Layout

Create `activity_main.xml` with:
- Camera preview view
- Drawing canvas overlay
- Text views for gesture and expression
- Buttons for clear/reset

### Resources for Native Android

- **MediaPipe Android Guide**: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/android
- **TensorFlow Lite Android**: https://www.tensorflow.org/lite/android
- **CameraX Guide**: https://developer.android.com/training/camerax

---

## 🐍 Option 2: Kivy + Buildozer (Python-Based)

**Best for**: Reuse existing Python code, faster development  
**Difficulty**: Medium (Python knowledge helpful)  
**Time**: 1-2 weeks

### Why This Approach?

- ✅ **Reuse Python Code**: Less rewriting
- ✅ **Cross-Platform**: Can also build for iOS
- ✅ **Python-Friendly**: If you're comfortable with Python

### Challenges

- ⚠️ **Complex Dependencies**: OpenCV, MediaPipe, TensorFlow on Android
- ⚠️ **Larger APK Size**: Python runtime included (~50-100MB)
- ⚠️ **Performance**: Slower than native, but acceptable for many use cases
- ⚠️ **Limited UI Options**: Kivy has its own UI framework

### Step-by-Step Guide

#### Step 1: Install Kivy and Buildozer

```bash
pip install kivy
pip install buildozer
pip install cython
```

#### Step 2: Create Kivy App Structure

Convert your app to use Kivy's camera and UI:

**Create `main.py`** (Kivy version):
```python
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

class InvisInkApp(App):
    def build(self):
        # Similar structure to your invisink_app.py
        # But using Kivy widgets instead of cv2.imshow
        pass

if __name__ == '__main__':
    InvisInkApp().run()
```

#### Step 3: Create Buildozer Spec File

**Create `buildozer.spec`**:
```ini
[app]
title = InvisInk
package.name = invisink
package.domain = com.yourname.invisink
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite
version = 0.1
requirements = python3,kivy,opencv-python-headless,numpy,mediapipe,tensorflow
orientation = portrait
osx.python_version = 3
osx.kivy_version = 1.9.1

[buildozer]
log_level = 2
warn_on_root = 1
```

**Note**: You'll need to use `opencv-python-headless` for Android, and may need to compile MediaPipe and TensorFlow from source or use pre-built wheels.

#### Step 4: Build APK

```bash
buildozer android debug
```

This will:
- Download Android SDK/NDK
- Compile Python and dependencies
- Create APK file

**Build time**: 30-60 minutes (first time)

### Resources for Kivy

- **Kivy Documentation**: https://kivy.org/doc/stable/
- **Buildozer Guide**: https://buildozer.readthedocs.io/

---

## 🌐 Option 3: Hybrid - Web App + Android WebView

**Best for**: Quick prototype, web developers  
**Difficulty**: Low (if you have Streamlit version)  
**Time**: 1-3 days

### Why This Approach?

- ✅ **Fastest Development**: You already have `streamlit_app.py`
- ✅ **Easy Updates**: Update web app without rebuilding APK
- ✅ **Reuse Streamlit Code**: Minimal changes needed

### Limitations

- ⚠️ **Performance**: WebView is slower than native
- ⚠️ **Limited Features**: Some Android features not accessible
- ⚠️ **Internet Required**: Unless you bundle a local server

### Step-by-Step Guide

#### Step 1: Create Android WebView App

1. Create new Android project in Android Studio
2. Add WebView to layout
3. Point WebView to your Streamlit server (local or remote)

**MainActivity.kt**:
```kotlin
import android.webkit.WebView
import android.webkit.WebViewClient

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        val webView: WebView = findViewById(R.id.webview)
        webView.settings.javaScriptEnabled = true
        webView.settings.mediaPlaybackRequiresUserGesture = false
        webView.webViewClient = WebViewClient()
        
        // Load your Streamlit app
        webView.loadUrl("http://localhost:8501") // Or your server URL
    }
}
```

#### Step 2: Run Streamlit Server Locally

Bundle a lightweight Python server in your app, or host on a server.

---

## 📊 Comparison Table

| Feature | Native Android | Kivy + Buildozer | WebView Hybrid |
|---------|---------------|------------------|----------------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **APK Size** | Small (~10-20MB) | Large (~50-100MB) | Medium (~5-10MB) |
| **Development Time** | 2-4 weeks | 1-2 weeks | 1-3 days |
| **Code Reuse** | Low (rewrite) | High (Python) | Very High (Streamlit) |
| **Learning Curve** | High (Kotlin/Java) | Medium (Kivy) | Low (WebView) |
| **Battery Usage** | Low | Medium | Medium-High |
| **Offline Support** | Yes | Yes | Maybe (depends) |
| **Best For** | Production apps | Prototypes, Python devs | Quick demos |

---

## 🎯 My Recommendation

**For your project, I recommend Option 1 (Native Android)** because:

1. ✅ **MediaPipe has excellent Android SDK** - Easy to integrate
2. ✅ **TensorFlow Lite is optimized for mobile** - Better performance
3. ✅ **Professional result** - Better user experience
4. ✅ **Smaller app size** - Users prefer smaller downloads
5. ✅ **Better battery life** - Important for camera apps

**If you want to prototype quickly**, start with **Option 3 (WebView)** using your existing Streamlit app, then move to native later.

---

## 🛠️ Getting Started - Next Steps

1. **Choose your approach** (I recommend Native Android)
2. **Convert your model** to TensorFlow Lite (see Step 1 in Option 1)
3. **Set up Android Studio**
4. **Start with a simple prototype** - Just camera + MediaPipe hand tracking first
5. **Add symbol classification** - Integrate TensorFlow Lite model
6. **Polish UI** - Add drawing canvas and expression display

---

## 📚 Learning Resources

### For Native Android Development:
- **Android Developer Documentation**: https://developer.android.com/docs
- **Kotlin for Android**: https://developer.android.com/kotlin
- **MediaPipe Android Examples**: https://github.com/google/mediapipe/tree/master/mediapipe/examples/android
- **TensorFlow Lite Android Tutorial**: https://www.tensorflow.org/lite/android

### For Kivy:
- **Kivy Crash Course**: https://www.youtube.com/results?search_query=kivy+android+tutorial
- **Kivy Documentation**: https://kivy.org/doc/stable/

---

## ❓ Need Help?

If you need help with any specific part:
1. Model conversion issues
2. Android Studio setup
3. MediaPipe integration
4. TensorFlow Lite implementation
5. UI/UX design

Just ask! I can help you with code examples, troubleshooting, or breaking down any step into smaller tasks.

---

**Good luck with your Android app development! 🚀📱**

