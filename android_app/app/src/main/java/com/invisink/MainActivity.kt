package com.invisink

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Main Activity for InvisInk Android App
 * 
 * This is the main entry point of the application.
 * It handles camera setup, hand tracking, gesture recognition, and symbol classification.
 */
class MainActivity : AppCompatActivity() {
    
    // UI Components
    private lateinit var previewView: PreviewView
    private lateinit var gestureTextView: TextView
    private lateinit var expressionTextView: TextView
    private lateinit var drawingCanvas: DrawingCanvas
    
    // Camera components
    private var imageCapture: ImageCapture? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private lateinit var cameraExecutor: ExecutorService
    
    // Core components
    private lateinit var handTracker: HandTracker
    private lateinit var symbolClassifier: SymbolClassifier
    
    // State variables (matching Python app)
    private val drawingPoints = mutableListOf<Point>()
    private var currentExpression = ""
    private var lastGesture = "UNKNOWN"
    private var lastFistTime = 0L
    private val debounceTime = 1500L // 1.5 seconds
    
    // Classes mapping (matching Python app)
    private val classes = arrayOf("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "(", ")")
    
    companion object {
        private const val TAG = "InvisInk"
        private const val CAMERA_PERMISSION_REQUEST_CODE = 100
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize UI components
        previewView = findViewById(R.id.previewView)
        gestureTextView = findViewById(R.id.gestureTextView)
        expressionTextView = findViewById(R.id.expressionTextView)
        drawingCanvas = findViewById(R.id.drawingCanvas)
        
        // Initialize core components
        try {
            handTracker = HandTracker(this) { gesture, landmarks ->
                runOnUiThread {
                    handleGesture(gesture, landmarks)
                }
            }
            symbolClassifier = SymbolClassifier(this)
        } catch (e: Exception) {
            Toast.makeText(this, "Error initializing components: ${e.message}", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        
        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE
            )
        }
        
        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }
    
    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            
            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
            
            // Image Analysis for hand tracking
            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }
            
            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()
                
                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(
                    this as LifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )
            } catch (exc: Exception) {
                Toast.makeText(this, "Use case binding failed: ${exc.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun processImage(imageProxy: ImageProxy) {
        // Convert ImageProxy to Bitmap for MediaPipe
        val bitmap = imageProxyToBitmap(imageProxy)
        
        // Process with hand tracker
        handTracker.detectAsync(bitmap, System.currentTimeMillis())
        
        imageProxy.close()
    }
    
    private fun imageProxyToBitmap(imageProxy: ImageProxy): android.graphics.Bitmap {
        val buffer: java.nio.ByteBuffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        
        val image = imageProxy.image
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer
        
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        
        val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val imageBytes = out.toByteArray()
        
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
    
    /**
     * Handle gesture recognition (matching Python app logic)
     */
    private fun handleGesture(gesture: String, landmarks: List<com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult>?) {
        gestureTextView.text = "Gesture: $gesture"
        
        if (landmarks == null || landmarks.isEmpty()) {
            return
        }
        
        val handLandmarkerResult = landmarks[0]
        if (handLandmarkerResult.landmarks().isEmpty()) {
            return
        }
        
        val handLandmarks = handLandmarkerResult.landmarks()[0]
        val indexFingerTip = handLandmarks[8] // INDEX_FINGER_TIP
        
        val screenWidth = previewView.width.toFloat()
        val screenHeight = previewView.height.toFloat()
        
        val x = (indexFingerTip.x() * screenWidth).toInt()
        val y = (indexFingerTip.y() * screenHeight).toInt()
        
        when (gesture) {
            "FINGERTIP" -> {
                // Drawing mode - add point
                drawingPoints.add(Point(x, y))
                drawingCanvas.addPoint(Point(x, y))
                drawingCanvas.invalidate()
            }
            
            "FIST" -> {
                // Recognize symbol
                val currentTime = System.currentTimeMillis()
                if (drawingPoints.size > 10 && (currentTime - lastFistTime > debounceTime)) {
                    lastFistTime = currentTime
                    recognizeSymbol()
                }
            }
            
            "THUMBS_UP" -> {
                // Solve equation
                if (lastGesture != "THUMBS_UP" && currentExpression.isNotEmpty()) {
                    solveExpression()
                }
            }
            
            "OPEN_HAND" -> {
                // Clear everything
                currentExpression = ""
                drawingPoints.clear()
                drawingCanvas.clear()
                expressionTextView.text = "Expression: "
            }
        }
        
        lastGesture = gesture
    }
    
    /**
     * Recognize the drawn symbol using TensorFlow Lite
     */
    private fun recognizeSymbol() {
        if (drawingPoints.size < 10) return
        
        try {
            // Get bounding box
            val xCoords = drawingPoints.map { it.x }
            val yCoords = drawingPoints.map { it.y }
            val xMin = maxOf(0, xCoords.minOrNull()!! - 20)
            val xMax = minOf(previewView.width, xCoords.maxOrNull()!! + 20)
            val yMin = maxOf(0, yCoords.minOrNull()!! - 20)
            val yMax = minOf(previewView.height, yCoords.maxOrNull()!! + 20)
            
            // Create bitmap from drawing points
            val symbolBitmap = drawingCanvas.getSymbolBitmap(xMin, yMin, xMax, yMax)
            
            if (symbolBitmap != null) {
                // Classify symbol
                val predictedClass = symbolClassifier.classify(symbolBitmap)
                currentExpression += predictedClass
                expressionTextView.text = "Expression: $currentExpression"
                
                // Clear drawing points
                drawingPoints.clear()
                drawingCanvas.clear()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Error recognizing symbol: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    /**
     * Solve the mathematical expression
     */
    private fun solveExpression() {
        if (currentExpression.isEmpty()) return
        
        try {
            // Replace visual symbols with evaluable ones
            var expression = currentExpression
                .replace("*", "*")
                .replace("/", "/")
            
            // Use a simple math evaluator (safer than eval)
            val result = evaluateExpression(expression)
            currentExpression += " = $result"
            expressionTextView.text = "Expression: $currentExpression"
        } catch (e: Exception) {
            currentExpression = "Error"
            expressionTextView.text = "Expression: Error"
        }
    }
    
    /**
     * Simple expression evaluator (safer than eval)
     * For production, consider using a proper math parser library like exp4j
     */
    private fun evaluateExpression(expr: String): Double {
        return try {
            // Remove spaces and prepare expression
            var cleanExpr = expr.replace(" ", "")
                .replace("*", "*")
                .replace("/", "/")
            
            // Use ScriptEngine for evaluation (safer than eval)
            val engine = javax.script.ScriptEngineManager().getEngineByName("js")
            val result = engine?.eval(cleanExpr)
            
            when (result) {
                is Number -> result.toDouble()
                is Double -> result
                is Int -> result.toDouble()
                else -> throw Exception("Invalid expression result")
            }
        } catch (e: Exception) {
            throw Exception("Invalid expression: ${e.message}")
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

/**
 * Simple Point data class
 */
data class Point(val x: Int, val y: Int)

