package com.invisink

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Symbol Classifier using TensorFlow Lite
 * 
 * This class handles symbol recognition using the converted TensorFlow Lite model.
 * It processes drawn symbols and classifies them into one of 16 classes.
 */
class SymbolClassifier(context: Context) {
    private val interpreter: Interpreter
    private val classes = arrayOf("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "(", ")")
    
    // Model input/output dimensions
    private val inputImageWidth = 30
    private val inputImageHeight = 30
    private val numClasses = 16
    
    init {
        val model = loadModelFile(context, "invisink_model.tflite")
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setUseXNNPACK(true) // Use optimized CPU operations
        }
        interpreter = Interpreter(model, options)
    }
    
    /**
     * Classify a symbol bitmap
     * 
     * @param symbolBitmap The bitmap containing the drawn symbol
     * @return The predicted class (e.g., "0", "1", "+", etc.)
     */
    fun classify(symbolBitmap: Bitmap): String {
        // Preprocess image: resize to 30x30, convert to grayscale, normalize
        val processedBitmap = preprocessImage(symbolBitmap)
        
        // Convert bitmap to ByteBuffer
        val inputBuffer = convertBitmapToByteBuffer(processedBitmap)
        
        // Prepare output buffer
        val outputBuffer = ByteBuffer.allocateDirect(numClasses * 4) // 16 classes * 4 bytes (float)
        outputBuffer.order(ByteOrder.nativeOrder())
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        
        // Get predictions
        val predictions = FloatArray(numClasses)
        outputBuffer.rewind()
        outputBuffer.asFloatBuffer().get(predictions)
        
        // Find the class with highest probability
        var maxIndex = 0
        var maxValue = predictions[0]
        for (i in 1 until numClasses) {
            if (predictions[i] > maxValue) {
                maxValue = predictions[i]
                maxIndex = i
            }
        }
        
        return classes[maxIndex]
    }
    
    /**
     * Preprocess image: resize to 30x30 and convert to grayscale
     */
    private fun preprocessImage(bitmap: Bitmap): Bitmap {
        // Resize to model input size
        val resized = Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true)
        
        // Convert to grayscale if needed
        if (resized.config != Bitmap.Config.ALPHA_8) {
            val grayBitmap = Bitmap.createBitmap(inputImageWidth, inputImageHeight, Bitmap.Config.ALPHA_8)
            val canvas = android.graphics.Canvas(grayBitmap)
            val paint = android.graphics.Paint().apply {
                colorFilter = android.graphics.ColorMatrixColorFilter(
                    android.graphics.ColorMatrix().apply {
                        setSaturation(0f)
                    }
                )
            }
            canvas.drawBitmap(resized, 0f, 0f, paint)
            return grayBitmap
        }
        
        return resized
    }
    
    /**
     * Convert bitmap to ByteBuffer for TensorFlow Lite
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(inputImageWidth * inputImageHeight * 1 * 4) // 30*30*1*4 bytes
        buffer.order(ByteOrder.nativeOrder())
        
        val pixels = IntArray(inputImageWidth * inputImageHeight)
        bitmap.getPixels(pixels, 0, inputImageWidth, 0, 0, inputImageWidth, inputImageHeight)
        
        for (pixel in pixels) {
            // Normalize pixel value to [0, 1]
            val normalizedValue = (pixel and 0xFF) / 255.0f
            buffer.putFloat(normalizedValue)
        }
        
        return buffer
    }
    
    /**
     * Load TensorFlow Lite model from assets
     */
    @Throws(IOException::class)
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun close() {
        interpreter.close()
    }
}

