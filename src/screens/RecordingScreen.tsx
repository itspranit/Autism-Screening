import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { Camera, useCameraDevice, useCameraFormat, VideoFile } from 'react-native-vision-camera';
import { useNavigation, useIsFocused } from '@react-navigation/native';

const SESSION_DURATION_SEC = 180; // 3 minutes

// FIX 1: Using named export to match App.tsx imports perfectly
export function RecordingScreen() {
  const navigation = useNavigation<any>();
  const isFocused = useIsFocused(); // FIX 2: Tracks if the screen is actually visible
  const cameraRef = useRef<Camera>(null);
  const device = useCameraDevice('front');
  
  // We want a stable format for the LSTM (30fps)
  const format = useCameraFormat(device, [
    { fps: 30 },
    { videoResolution: { width: 720, height: 1280 } }
  ]);

  const [isRecording, setIsRecording] = useState(false);
  const [timeLeft, setTimeLeft] = useState(SESSION_DURATION_SEC);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);

  // Timer Logic
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording && timeLeft > 0) {
      interval = setInterval(() => setTimeLeft((prev) => prev - 1), 1000);
    } else if (timeLeft === 0 && isRecording) {
      stopSession();
    }
    return () => clearInterval(interval);
  }, [isRecording, timeLeft]);

  const startSession = async () => {
    if (!cameraRef.current || !cameraReady) return;
    
    setIsRecording(true);
    
    // Start Recording (Audio disabled for privacy)
    cameraRef.current.startRecording({
      onRecordingFinished: (video) => handleVideoRecorded(video),
      onRecordingError: (error) => {
        console.error("Recording failed!", error);
        setIsRecording(false);
      }
    });
  };

  const stopSession = async () => {
    if (!cameraRef.current) return;
    await cameraRef.current.stopRecording(); // Triggers onRecordingFinished
  };

  const handleVideoRecorded = async (video: VideoFile) => {
    setIsRecording(false);
    setIsProcessing(true);

    try {
      console.log(`[Option B] Video securely saved to cache: ${video.path}`);
      
      // Navigate to your new Processing Screen, passing the video path!
      navigation.navigate('ProcessingScreen', { videoPath: video.path });

    } catch (error) {
      console.error("Pipeline Failed:", error);
      setIsProcessing(false);
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  if (!device) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#00ff00" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      
      {/* 1. The Stimulus Video Player (Taking up the whole screen) */}
      <View style={styles.stimulusPlaceholder}>
        <Text style={styles.stimulusText}>[ Stimulus Video Playing Here ]</Text>
      </View>

      {/* 2. The Front Camera (Hidden or PiP) */}
      <Camera
        ref={cameraRef}
        style={styles.pipCamera}
        device={device}
        format={format}
        isActive={isFocused} // FIX 3: Camera stays safely active until we leave the screen
        video={true}
        audio={false} // HIPPA compliance: no audio
        onInitialized={() => setCameraReady(true)}
      />

      {/* 3. The UI Overlay */}
      <View style={styles.overlay}>
        {!isRecording && !isProcessing && (
          <TouchableOpacity 
            style={[styles.btn, !cameraReady && styles.btnDisabled]} 
            onPress={startSession}
            disabled={!cameraReady}
          >
            <Text style={styles.btnText}>
              {cameraReady ? "START 3-MIN SESSION" : "WARMING UP CAMERA..."}
            </Text>
          </TouchableOpacity>
        )}

        {isRecording && (
          <View style={styles.recordingCard}>
            <View style={styles.redDot} />
            <Text style={styles.timerText}>Recording: {formatTime(timeLeft)}</Text>
            <TouchableOpacity style={styles.stopBtn} onPress={stopSession}>
              <Text style={styles.stopBtnText}>END EARLY</Text>
            </TouchableOpacity>
          </View>
        )}

        {isProcessing && (
          <View style={styles.processingCard}>
            <ActivityIndicator size="large" color="#00ff00" />
            <Text style={styles.processingText}>Saving Secure Session...</Text>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'black' },
  pipCamera: { 
    position: 'absolute', top: 60, right: 20, 
    width: 100, height: 150, borderRadius: 12, zIndex: 10,
    borderWidth: 2, borderColor: '#333'
  },
  stimulusPlaceholder: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#111',
    justifyContent: 'center',
    alignItems: 'center',
  },
  stimulusText: { color: '#666', fontSize: 18 },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'flex-end',
    padding: 30,
    paddingBottom: 60,
    zIndex: 20
  },
  btn: { backgroundColor: '#4F46E5', padding: 15, borderRadius: 10, alignItems: 'center' },
  btnDisabled: { backgroundColor: '#333' },
  btnText: { color: 'white', fontWeight: 'bold', fontSize: 16 },
  recordingCard: {
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    padding: 20, borderRadius: 20,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between'
  },
  redDot: { width: 15, height: 15, borderRadius: 8, backgroundColor: '#EF4444' },
  timerText: { color: 'white', fontSize: 20, fontWeight: 'bold', flex: 1, marginLeft: 15 },
  stopBtn: { backgroundColor: 'transparent', padding: 10 },
  stopBtnText: { color: '#EF4444', fontWeight: 'bold' },
  processingCard: {
    backgroundColor: '#1E293B', padding: 30, borderRadius: 20, alignItems: 'center'
  },
  processingText: { color: 'white', fontSize: 18, fontWeight: 'bold', marginTop: 20 }
});