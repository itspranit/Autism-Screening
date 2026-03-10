import React, { useEffect } from 'react';
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';

export function ProcessingScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  
  // This is the file path of the 3-minute video we just recorded!
  const videoPath = route.params?.videoPath; 

  useEffect(() => {
    const processVideo = async () => {
      try {
        console.log(`[Processing Engine] Received Video at: ${videoPath}`);
        
        // =================================================================
        // NEXT PHASE (SWIFT INTEGRATION) WILL GO HERE
        // 1. We will pass videoPath to our Swift module
        // 2. Swift will extract the frames
        // 3. TFLite models will run on those frames
        // =================================================================
        
        // For now, we simulate the 3-4 second AI processing time
        await new Promise(resolve => setTimeout(resolve, 3500));

        // Simulated AI output based on our Option B architecture
        const clinicalReport = {
          gazeRisk: 0.15,
          motorRisk: 0.85,
          faceRisk: 0.20,
          finalScore: (0.4 * 0.15) + (0.4 * 0.85) + (0.2 * 0.20),
          timestamp: new Date().toISOString()
        };

        // Navigate to the final Results Screen! 
        // (Make sure the name matches what's in your App.tsx)
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
  container: { 
    flex: 1, 
    backgroundColor: '#020617', 
    justifyContent: 'center', 
    alignItems: 'center',
    padding: 20
  },
  text: { 
    color: 'white', 
    fontSize: 22, 
    marginTop: 30, 
    fontWeight: 'bold' 
  },
  subtext: { 
    color: '#94a3b8', 
    fontSize: 15, 
    marginTop: 10,
    textAlign: 'center'
  }
});