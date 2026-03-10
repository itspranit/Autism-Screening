import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';
import { MotorAnalysis, FaceMeshLandmark, FRAME_BUFFER_SIZE, POSE_JOINT_COUNT, POSE_FEATURE_DIM } from '../utils/types';

const MOTION_ENERGY_VELOCITY_SCALE = 0.01;
let motorModel: TensorflowModel | null = null;

import RNFS from 'react-native-fs';





export async function loadMotorModel(): Promise<void> {
  // if (motorModel) return;
  try {
    console.log('[EngineB] Hunting for model file in iOS bundle...');
    
    // Check all 3 possible ways Xcode might have bundled the file
    const pathsToTry = [
      `${RNFS.MainBundlePath}/motor_risk_model.tflite`,              // If files were dragged individually
      `${RNFS.MainBundlePath}/models/motor_risk_model.tflite`,       // If the 'models' folder was dragged
      `${RNFS.MainBundlePath}/assets/models/motor_risk_model.tflite` // If the 'assets' folder was dragged
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

    // Load it (Using 'as any' to bypass the strict TypeScript string error we had earlier)
    motorModel = await loadTensorflowModel(
      { url: `file://${foundPath}` },
      'default' as any 
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

// ... (KEEP ALL YOUR EXISTING BUFFER & MATH FUNCTIONS BELOW THIS) ...
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
  return { frameBuffer: new Float32Array(FRAME_BUFFER_SIZE * POSE_FEATURE_DIM), frameCount: 0, writeIdx: 0, motionEnergyAccumulator: 0, motionEnergyFrames: 0, previousJoints: null, inferencePending: false, lastMotorRisk: 0 };
}

function normalisePoseLandmarks(landmarks: FaceMeshLandmark[]): Float32Array {
  const joints = new Float32Array(POSE_FEATURE_DIM);
  const leftHip = landmarks[23] ?? { x: 0, y: 0, z: 0 };
  const rightHip = landmarks[24] ?? { x: 0, y: 0, z: 0 };
  const nose = landmarks[0] ?? { x: 0, y: 0, z: 0 };
  const hipMidX = (leftHip.x + rightHip.x) / 2;
  const hipMidY = (leftHip.y + rightHip.y) / 2;
  const hipMidZ = (leftHip.z + rightHip.z) / 2;
  const torsoHeight = Math.sqrt(Math.pow(nose.x - hipMidX, 2) + Math.pow(nose.y - hipMidY, 2) + Math.pow(nose.z - hipMidZ, 2)) || 1;

  for (let i = 0; i < POSE_JOINT_COUNT; i++) {
    const lm = landmarks[i] ?? { x: 0, y: 0, z: 0 };
    joints[i * 3 + 0] = (lm.x - hipMidX) / torsoHeight;
    joints[i * 3 + 1] = (lm.y - hipMidY) / torsoHeight;
    joints[i * 3 + 2] = (lm.z - hipMidZ) / torsoHeight;
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
  const newState: MotorEngineState = { ...state, writeIdx: (state.writeIdx + 1) % FRAME_BUFFER_SIZE, frameCount: Math.min(state.frameCount + 1, FRAME_BUFFER_SIZE), motionEnergyAccumulator: state.motionEnergyAccumulator + motionEnergy, motionEnergyFrames: state.motionEnergyFrames + 1, previousJoints: joints };
  const bufferReady = newState.frameCount >= FRAME_BUFFER_SIZE;
  return { newState, bufferReady };
}

export async function runMotorInference(state: MotorEngineState): Promise<{ motorRisk: number; stimmingDetected: boolean }> {
  if (!motorModel) return { motorRisk: state.lastMotorRisk, stimmingDetected: state.lastMotorRisk > 0.5 };
  const orderedBuffer = new Float32Array(FRAME_BUFFER_SIZE * POSE_FEATURE_DIM);
  const startIdx = state.frameCount >= FRAME_BUFFER_SIZE ? state.writeIdx : 0;
  for (let i = 0; i < FRAME_BUFFER_SIZE; i++) {
    const srcIdx = (startIdx + i) % FRAME_BUFFER_SIZE;
    const srcOff = srcIdx * POSE_FEATURE_DIM;
    const dstOff = i * POSE_FEATURE_DIM;
    orderedBuffer.set(state.frameBuffer.subarray(srcOff, srcOff + POSE_FEATURE_DIM), dstOff);
  }
  try {
    const outputs = motorModel.runSync([orderedBuffer]);
    const rawOutput = outputs[0] as Float32Array;
    const motorRisk = Math.min(1, Math.max(0, rawOutput[0]));
    return { motorRisk, stimmingDetected: motorRisk > 0.5 };
  } catch (err) {
    console.error('[EngineB] Inference error:', err);
    return { motorRisk: state.lastMotorRisk, stimmingDetected: false };
  }
}

export function finalizeMotorAnalysis(state: MotorEngineState, motorRiskScore: number, stimmingDetected: boolean): MotorAnalysis {
  const frames = Math.max(state.motionEnergyFrames, 1);
  const motionEnergy = state.motionEnergyAccumulator / frames;
  return { motorRiskScore: Math.min(1, Math.max(0, motorRiskScore)), stimmingDetected, motionEnergy: Math.min(1, motionEnergy * 10), bufferComplete: state.frameCount >= FRAME_BUFFER_SIZE };
}