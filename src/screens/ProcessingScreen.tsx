import React, { useEffect } from 'react';
import { View, Text, ActivityIndicator, StyleSheet, NativeModules } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';

export function ProcessingScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  
  const videoPath = route.params?.videoPath; 

  useEffect(() => {
    const processVideo = async () => {
      try {
        console.log(`[Processing Engine] Received Video at: ${videoPath}`);
        
        // 1. Grab our new Swift Module
        const { VideoProcessor } = NativeModules;
        
        if (!VideoProcessor) {
            console.error("[JS Log] VideoProcessor is undefined! The native module didn't link.");
        } else {
            // 2. Pass the video path to Swift and wait for it to respond!
            console.log("[JS Log] Calling Swift...");
            const swiftResponse = await VideoProcessor.processVideo(videoPath);
            console.log("\n=============================================");
            console.log("🍏 [JS Log] SWIFT REPLIED: ", swiftResponse);
            console.log("=============================================\n");
        }
        
        // Simulate the rest of the AI time for now
        await new Promise(resolve => setTimeout(resolve, 3500));

        // Simulated AI output
        const clinicalReport = {
          gazeRisk: 0.15,
          motorRisk: 0.85,
          faceRisk: 0.20,
          finalScore: (0.4 * 0.15) + (0.4 * 0.85) + (0.2 * 0.20),
          timestamp: new Date().toISOString()
        };

        navigation.navigate('Results', { report: clinicalReport }); 

      } catch (error) {
        console.error("Error processing video:", error);
        alert("Failed to analyze session.");
        navigation.goBack();
      }
    };

    processVideo();
  }, [videoPath, navigation]);

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#38bdf8" />
      <Text style={styles.text}>Analyzing Video Frames</Text>
      <Text style={styles.subtext}>Running LSTM Kinematics & Face Morphologies...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#020617', justifyContent: 'center', alignItems: 'center', padding: 20 },
  text: { color: 'white', fontSize: 22, marginTop: 30, fontWeight: 'bold' },
  subtext: { color: '#94a3b8', fontSize: 15, marginTop: 10, textAlign: 'center' }
});