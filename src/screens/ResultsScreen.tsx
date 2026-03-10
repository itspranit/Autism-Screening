import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';

export function ResultsScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  
  // Read 'report' instead of 'result' to match our Canvas types and ProcessingScreen!
  const { report } = route.params;

  // Determine risk level based on the final multimodal score
  const isHighRisk = report.finalScore > 0.5;
  const primaryColor = isHighRisk ? '#EF4444' : '#22C55E'; // Red for risk, Green for typical
  const bgColor = isHighRisk ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)';

  return (
    <View style={styles.container}>
      <Text style={styles.headerTitle}>Clinical Report</Text>

      <View style={[styles.card, { backgroundColor: bgColor, borderColor: primaryColor }]}>
        <Text style={styles.cardTitle}>
          {isHighRisk ? 'High Risk Detected' : 'Typical Development'}
        </Text>
        <Text style={[styles.scoreText, { color: primaryColor }]}>
          {(report.finalScore * 100).toFixed(1)}%
        </Text>
        <Text style={styles.subtext}>Integrated Multimodal Score</Text>

        <View style={styles.divider} />

        <View style={styles.row}>
          <Text style={styles.label}>Engine A (Gaze):</Text>
          <Text style={styles.value}>{(report.gazeRisk * 100).toFixed(1)}%</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Engine B (Motor):</Text>
          <Text style={styles.value}>{(report.motorRisk * 100).toFixed(1)}%</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Engine C (Face):</Text>
          <Text style={styles.value}>{(report.faceRisk * 100).toFixed(1)}%</Text>
        </View>
      </View>

      <TouchableOpacity 
        style={styles.btn} 
        onPress={() => navigation.navigate('Welcome')}
      >
        <Text style={styles.btnText}>START NEW SCREENING</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#020617', padding: 24, justifyContent: 'center' },
  headerTitle: { color: 'white', fontSize: 28, fontWeight: 'bold', textAlign: 'center', marginBottom: 40 },
  card: { 
    padding: 30, 
    borderRadius: 20, 
    borderWidth: 2,
    alignItems: 'center',
    marginBottom: 40
  },
  cardTitle: { color: 'white', fontSize: 20, fontWeight: 'bold', marginBottom: 10 },
  scoreText: { fontSize: 64, fontWeight: 'bold', marginVertical: 10 },
  subtext: { color: '#94a3b8', fontSize: 14, textTransform: 'uppercase', letterSpacing: 1 },
  divider: { height: 1, backgroundColor: 'rgba(255,255,255,0.1)', width: '100%', marginVertical: 20 },
  row: { flexDirection: 'row', justifyContent: 'space-between', width: '100%', paddingVertical: 8 },
  label: { color: '#cbd5e1', fontSize: 16 },
  value: { color: 'white', fontSize: 16, fontWeight: 'bold' },
  btn: { backgroundColor: '#38bdf8', padding: 18, borderRadius: 12, alignItems: 'center' },
  btnText: { color: '#020617', fontWeight: 'bold', fontSize: 16 }
});