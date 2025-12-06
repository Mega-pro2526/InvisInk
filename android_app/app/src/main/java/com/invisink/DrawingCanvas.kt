package com.invisink

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * Custom View for drawing the fingertip path
 * 
 * This view overlays the camera preview and draws the path traced by the fingertip.
 */
class DrawingCanvas @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    private val drawingPath = Path()
    private val drawingPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 20f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        isAntiAlias = true
    }
    
    private val points = mutableListOf<Point>()
    
    /**
     * Add a point to the drawing path
     */
    fun addPoint(point: Point) {
        points.add(point)
        if (points.size == 1) {
            drawingPath.moveTo(point.x.toFloat(), point.y.toFloat())
        } else {
            drawingPath.lineTo(point.x.toFloat(), point.y.toFloat())
        }
        invalidate()
    }
    
    /**
     * Clear the drawing
     */
    fun clear() {
        points.clear()
        drawingPath.reset()
        invalidate()
    }
    
    /**
     * Get symbol bitmap from drawing points within bounding box
     */
    fun getSymbolBitmap(xMin: Int, yMin: Int, xMax: Int, yMax: Int): Bitmap? {
        if (points.isEmpty()) return null
        
        val width = xMax - xMin
        val height = yMax - yMin
        
        if (width <= 0 || height <= 0) return null
        
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        
        // Draw white background
        canvas.drawColor(Color.BLACK)
        
        // Draw the path in white
        val pathPaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 15f
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            isAntiAlias = true
        }
        
        val path = Path()
        var firstPoint = true
        for (point in points) {
            val x = point.x - xMin
            val y = point.y - yMin
            if (firstPoint) {
                path.moveTo(x.toFloat(), y.toFloat())
                firstPoint = false
            } else {
                path.lineTo(x.toFloat(), y.toFloat())
            }
        }
        
        canvas.drawPath(path, pathPaint)
        
        return bitmap
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawPath(drawingPath, drawingPaint)
    }
}

