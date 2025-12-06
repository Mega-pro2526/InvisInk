# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.

# Keep TensorFlow Lite classes
-keep class org.tensorflow.lite.** { *; }

# Keep MediaPipe classes
-keep class com.google.mediapipe.** { *; }

# Keep model classes
-keep class com.invisink.** { *; }

