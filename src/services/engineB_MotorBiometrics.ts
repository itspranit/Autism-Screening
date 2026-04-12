import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';
import RNFS from 'react-native-fs';
import { MotorAnalysis, FaceMeshLandmark, FRAME_BUFFER_SIZE, POSE_JOINT_COUNT, POSE_FEATURE_DIM } from '../utils/types';

const MOTION_ENERGY_VELOCITY_SCALE = 0.01;
let motorModel: TensorflowModel | null = null;

export async function loadMotorModel(): Promise<void> {
  try {
    console.log('[EngineB] Hunting for model file in iOS bundle...');
    
    const pathsToTry = [
      `${RNFS.MainBundlePath}/motor_risk_model.tflite`,
      `${RNFS.MainBundlePath}/models/motor_risk_model.tflite`,
      `${RNFS.MainBundlePath}/assets/models/motor_risk_model.tflite`
    ];

    let foundPath = null;
    for (const p of pathsToTry) {
      if (await RNFS.exists(p)) {
        foundPath = p;
        break;
      }
    }

    if (!foundPath) {
      throw new Error("XCODE BUNDLE ERROR: The file is missing. Target Membership must be checked in Xcode.");
    }

    console.log(`[EngineB] Found model at: ${foundPath}`);

    // Load natively on CPU to bypass LSTM crashes
    motorModel = await loadTensorflowModel(
      { url: `file://${foundPath}` },
      'default'
    );
    console.log('[EngineB] Motor model loaded natively.');
  } catch (err) {
    console.error('[EngineB] Failed to load motor model:', err);
    throw err;
  }
}

export function disposeMotorModel(): void {
  motorModel = null;
}

export interface MotorEngineState {
  frameBuffer: Float32Array;
  frameCount: number;
  writeIdx: number;
  motionEnergyAccumulator: number;
  motionEnergyFrames: number;
  previousJoints: Float32Array | null;
  inferencePending: boolean;
  lastMotorRisk: number;
}

export function createMotorEngine(): MotorEngineState {
  return { 
    frameBuffer: new Float32Array(FRAME_BUFFER_SIZE * POSE_FEATURE_DIM), 
    frameCount: 0, 
    writeIdx: 0, 
    motionEnergyAccumulator: 0, 
    motionEnergyFrames: 0, 
    previousJoints: null, 
    inferencePending: false, 
    lastMotorRisk: 0 
  };
}

// FIXED: Uses Shoulders instead of Hips to prevent divide-by-zero when lower body is off-camera
function normalisePoseLandmarks(landmarks: FaceMeshLandmark[]): Float32Array {
  const joints = new Float32Array(POSE_FEATURE_DIM);
  
  const leftShoulder = landmarks[11] ?? { x: 0, y: 0, z: 0 };
  const rightShoulder = landmarks[12] ?? { x: 0.1, y: 0, z: 0 }; // 0.1 prevents divide by zero
  const nose = landmarks[0] ?? { x: 0, y: 0, z: 0 };

  const shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
  const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
  const shoulderMidZ = (leftShoulder.z + rightShoulder.z) / 2;

  let scaleFactor = Math.sqrt(
    Math.pow(leftShoulder.x - rightShoulder.x, 2) + 
    Math.pow(leftShoulder.y - rightShoulder.y, 2) + 
    Math.pow(leftShoulder.z - rightShoulder.z, 2)
  );

  if (scaleFactor === 0 || isNaN(scaleFactor)) scaleFactor = 1.0;

  for (let i = 0; i < POSE_JOINT_COUNT; i++) {
    const lm = landmarks[i] ?? { x: 0, y: 0, z: 0 };
    joints[i * 3 + 0] = (lm.x - shoulderMidX) / scaleFactor;
    joints[i * 3 + 1] = (lm.y - shoulderMidY) / scaleFactor;
    joints[i * 3 + 2] = (lm.z - shoulderMidZ) / scaleFactor;
  }
  return joints;
}

function computeMotionEnergy(currentJoints: Float32Array, previousJoints: Float32Array): number {
  let totalVelocity = 0;
  for (let i = 0; i < POSE_FEATURE_DIM; i++) {
    totalVelocity += Math.abs(currentJoints[i] - previousJoints[i]);
  }
  return (totalVelocity / POSE_FEATURE_DIM) * MOTION_ENERGY_VELOCITY_SCALE;
}

export function addMotorFrame(state: MotorEngineState, landmarks: FaceMeshLandmark[]): { newState: MotorEngineState; bufferReady: boolean } {
  const joints = normalisePoseLandmarks(landmarks);
  const offset = state.writeIdx * POSE_FEATURE_DIM;
  state.frameBuffer.set(joints, offset);
  
  let motionEnergy = 0;
  if (state.previousJoints) motionEnergy = computeMotionEnergy(joints, state.previousJoints);
  
  const newState: MotorEngineState = { 
    ...state, 
    writeIdx: (state.writeIdx + 1) % FRAME_BUFFER_SIZE, 
    frameCount: Math.min(state.frameCount + 1, FRAME_BUFFER_SIZE), 
    motionEnergyAccumulator: state.motionEnergyAccumulator + motionEnergy, 
    motionEnergyFrames: state.motionEnergyFrames + 1, 
    previousJoints: joints 
  };
  
  const bufferReady = newState.frameCount >= FRAME_BUFFER_SIZE;
  return { newState, bufferReady };
}

export async function runMotorInference(state: MotorEngineState): Promise<{ motorRisk: number; stimmingDetected: boolean }> {
  if (!motorModel) return { motorRisk: state.lastMotorRisk, stimmingDetected: state.lastMotorRisk > 0.5 };
  
  // 🚨 THE FIX: Do not run the AI until we have a full 100 frames!
  if (state.frameCount < FRAME_BUFFER_SIZE) {
    console.log(`[EngineB] Buffering movement... ${state.frameCount}/${FRAME_BUFFER_SIZE} frames`);
    return { motorRisk: state.lastMotorRisk, stimmingDetected: false };
  }

  const orderedBuffer = new Float32Array(FRAME_BUFFER_SIZE * POSE_FEATURE_DIM);
  const startIdx = state.frameCount >= FRAME_BUFFER_SIZE ? state.writeIdx : 0;
  
  for (let i = 0; i < FRAME_BUFFER_SIZE; i++) {
    const srcIdx = (startIdx + i) % FRAME_BUFFER_SIZE;
    const srcOff = srcIdx * POSE_FEATURE_DIM;
    const dstOff = i * POSE_FEATURE_DIM;
    orderedBuffer.set(state.frameBuffer.subarray(srcOff, srcOff + POSE_FEATURE_DIM), dstOff);
  }
  
  // 🚨 DIAGNOSTIC CHECK: Are we feeding it zeros?
  console.log(`[EngineB] Buffer Full! First 3 joint coordinates: ${orderedBuffer[0].toFixed(3)}, ${orderedBuffer[1].toFixed(3)}, ${orderedBuffer[2].toFixed(3)}`);

  try {
    const outputs = motorModel.runSync([orderedBuffer]);
    const rawOutput = outputs[0] as Float32Array;
    const motorRisk = Math.min(1, Math.max(0, rawOutput[0]));
    
    console.log(`[EngineB] AI Output: ${motorRisk}`);
    return { motorRisk, stimmingDetected: motorRisk > 0.5 };
  } catch (err) {
    console.error('[EngineB] Inference error:', err);
    return { motorRisk: state.lastMotorRisk, stimmingDetected: false };
  }
}

export function finalizeMotorAnalysis(state: MotorEngineState, motorRiskScore: number, stimmingDetected: boolean): MotorAnalysis {
  const frames = Math.max(state.motionEnergyFrames, 1);
  const motionEnergy = state.motionEnergyAccumulator / frames;
  return { 
    motorRiskScore: Math.min(1, Math.max(0, motorRiskScore)), 
    stimmingDetected, 
    motionEnergy: Math.min(1, motionEnergy * 10), 
    bufferComplete: state.frameCount >= FRAME_BUFFER_SIZE 
  };
}