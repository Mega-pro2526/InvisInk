package com.invisink

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * Hand Tracker using MediaPipe
 * 
 * This class handles hand detection and gesture recognition.
 * It mirrors the functionality of the Python MediaPipe implementation.
 */
class HandTracker(
    context: Context,
    private val onResult: (String, List<HandLandmarkerResult>?) -> Unit
) {
    private val handLandmarker: HandLandmarker
    
    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .setDelegate(BaseOptions.Delegate.GPU)
            .build()
        
        val options = HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setMinHandDetectionConfidence(0.7f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result, image ->
                val gesture = detectGesture(result)
                onResult(gesture, if (result.landmarks().isNotEmpty()) listOf(result) else null)
            }
            .build()
        
        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }
    
    /**
     * Detect gesture from hand landmarks
     * This matches the Python get_gesture() function logic
     */
    private fun detectGesture(result: HandLandmarkerResult): String {
        if (result.landmarks().isEmpty()) {
            return "UNKNOWN"
        }
        
        val handLandmarks = result.landmarks()[0]
        if (handLandmarks.size < 21) {
            return "UNKNOWN"
        }
        
        // Landmark indices for fingertips (matching Python)
        val tipIds = intArrayOf(4, 8, 12, 16, 20) // Thumb, Index, Middle, Ring, Pinky
        
        val fingersExtended = mutableListOf<Int>()
        
        // Thumb (special case: check x-position relative to its base)
        val thumbTip = handLandmarks[tipIds[0]]
        val thumbIp = handLandmarks[tipIds[0] - 1]
        if (thumbTip.x() < thumbIp.x()) {
            fingersExtended.add(1)
        } else {
            fingersExtended.add(0)
        }
        
        // Other four fingers (check if tip is above PIP joint)
        for (i in 1..4) {
            val tip = handLandmarks[tipIds[i]]
            val pip = handLandmarks[tipIds[i] - 2]
            if (tip.y() < pip.y()) {
                fingersExtended.add(1)
            } else {
                fingersExtended.add(0)
            }
        }
        
        val totalFingers = fingersExtended.sum()
        
        // FINGERTIP: Only index finger is extended
        if (totalFingers == 1 && fingersExtended[1] == 1) {
            return "FINGERTIP"
        }
        
        // THUMBS UP: Only thumb is extended
        if (totalFingers == 1 && fingersExtended[0] == 1) {
            return "THUMBS_UP"
        }
        
        // FIST: No fingers are extended
        if (totalFingers == 0) {
            return "FIST"
        }
        
        // OPEN HAND: All five fingers are extended
        if (totalFingers == 5) {
            return "OPEN_HAND"
        }
        
        return "UNKNOWN"
    }
    
    /**
     * Process image asynchronously
     */
    fun detectAsync(bitmap: Bitmap, timestamp: Long) {
        val mpImage = com.google.mediapipe.tasks.core.Image.createFromBitmap(bitmap)
        handLandmarker.detectAsync(mpImage, timestamp)
    }
    
    fun close() {
        handLandmarker.close()
    }
}

