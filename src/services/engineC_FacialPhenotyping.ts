import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';
import RNFS from 'react-native-fs';
import { FacialAnalysis, FaceMeshLandmark } from '../utils/types';

const INPUT_SIZE = 224 * 224 * 3;
const INFERENCE_INTERVAL_FRAMES = 30;
let faceModel: TensorflowModel | null = null;

export async function loadFaceModel(): Promise<void> {
  try {
    const pathsToTry = [
      `${RNFS.MainBundlePath}/face_risk_model.tflite`,
      `${RNFS.MainBundlePath}/models/face_risk_model.tflite`,
      `${RNFS.MainBundlePath}/assets/models/face_risk_model.tflite`
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

    faceModel = await loadTensorflowModel(
      { url: `file://${foundPath}` },
      'default' as any
    );
    console.log('[EngineC] Face model loaded natively.');
  } catch (err) {
    console.error('[EngineC] Failed to load face model:', err);
    throw err;
  }
}

export function disposeFaceModel(): void {
  faceModel = null;
}

export interface FaceEngineState { 
  frameCounter: number; 
  lastFaceRisk: number; 
  lastConfidence: number; 
  inferenceCount: number; 
  riskAccumulator: number; 
}

export function createFaceEngine(): FaceEngineState { 
  return { frameCounter: 0, lastFaceRisk: 0, lastConfidence: 0, inferenceCount: 0, riskAccumulator: 0 }; 
}

function mobilenetPreprocess(pixelData: Uint8ClampedArray | Float32Array): Float32Array {
  const tensor = new Float32Array(INPUT_SIZE);
  const inputLength = Math.min(pixelData.length, INPUT_SIZE);
  for (let i = 0; i < inputLength; i++) {
    // Normalizes pixels to 0.0-1.0, then shifts to -1.0 to 1.0 range usually required by MobileNet
    const pixel = pixelData[i] > 1 ? pixelData[i] / 255.0 : pixelData[i];
    tensor[i] = (pixel - 0.5) * 2.0;  
  }
  return tensor;
}

// FIXED: No longer rejects images that aren't perfectly 224x224
function prepareFaceTensor(faceImageData: Float32Array): Float32Array {
  return mobilenetPreprocess(faceImageData);
}

export function shouldRunFaceInference(state: FaceEngineState): boolean { 
  return state.frameCounter % INFERENCE_INTERVAL_FRAMES === 0; 
}

export function tickFaceEngine(state: FaceEngineState): FaceEngineState { 
  return { ...state, frameCounter: state.frameCounter + 1 }; 
}

export async function runFaceInference(state: FaceEngineState, faceImageData: Float32Array): Promise<FaceEngineState> {
  if (!faceModel) return state;
  const tensor = prepareFaceTensor(faceImageData);
  
  try {
    const outputs = faceModel.runSync([tensor]);
    const rawOutput = outputs[0] as Float32Array;
    let faceRisk: number;
    let confidence: number;

    if (rawOutput.length === 1) {
      faceRisk = rawOutput[0];
      confidence = Math.abs(faceRisk - 0.5) * 2;
    } else {
      faceRisk = rawOutput[1];                   
      confidence = Math.max(rawOutput[0], rawOutput[1]);
    }

    return { 
      ...state, 
      lastFaceRisk: Math.min(1, Math.max(0, faceRisk)), 
      lastConfidence: confidence, 
      inferenceCount: state.inferenceCount + 1, 
      riskAccumulator: state.riskAccumulator + Math.min(1, Math.max(0, faceRisk)) 
    };
  } catch (err) {
    console.error('[EngineC] Inference error:', err);
    return state;
  }
}

export function finalizeFacialAnalysis(state: FaceEngineState, faceDetected: boolean): FacialAnalysis {
  const avgRisk = state.inferenceCount > 0 ? state.riskAccumulator / state.inferenceCount : 0;
  return { 
    faceRiskScore: Math.min(1, Math.max(0, avgRisk)), 
    confidence: state.lastConfidence, 
    faceDetected 
  };
}