/**
 * ENGINE A: Visual Attention / Gaze Aversion (L2CS-Net TFLite Version)
 *
 * Uses the L2CS-Net float32 model to estimate true 3D gaze (pitch and yaw).
 * Aversion is triggered if the yaw angle exceeds the threshold.
 */

import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import RNFS from 'react-native-fs';
import { GazeAnalysis } from '../utils/types';

// 🚨 CHECK THIS: Change to 448 * 448 * 3 if your specific lc2net model requires 448px images
const INPUT_SIZE = 224 * 224 * 3; 
const YAW_AVERSION_THRESHOLD_DEG = 20; 

let gazeModel: TensorflowModel | null = null;

export async function loadGazeModel(): Promise<void> {
  try {
    console.log('[EngineA] Hunting for L2CS-Net model file...');
    
    // Make sure your file is actually named 'gaze_model.tflite' in Xcode!
    const pathsToTry = [
      `${RNFS.MainBundlePath}/gaze_model.tflite`,
      `${RNFS.MainBundlePath}/models/gaze_model.tflite`,
      `${RNFS.MainBundlePath}/assets/models/gaze_model.tflite`
    ];

    let foundPath = null;
    for (const p of pathsToTry) {
      if (await RNFS.exists(p)) {
        foundPath = p;
        break;
      }
    }

    if (!foundPath) {
      throw new Error("XCODE BUNDLE ERROR: 'lc2net2.tflite' is missing. Check Target Membership.");
    }

    // Load natively on CPU
    gazeModel = await loadTensorflowModel(
      { url: `file://${foundPath}` },
      'default' as any
    );
    console.log('[EngineA] L2CS-Net gaze model loaded natively.');
  } catch (err) {
    console.error('[EngineA] Failed to load gaze model:', err);
    throw err;
  }
}

export function disposeGazeModel(): void {
  gazeModel = null;
}

// ── Public engine state ───────────────────────────────────────────────────────

export interface GazeEngineState {
  totalFrames: number;
  aversionFrames: number;
  yawAccumulator: number;
}

export function createGazeEngine(): GazeEngineState {
  return {
    totalFrames: 0,
    aversionFrames: 0,
    yawAccumulator: 0,
  };
}

// ── Preprocessing ─────────────────────────────────────────────────────────────

function preprocessGazeTensor(pixelData: Uint8Array | Uint8ClampedArray | Float32Array): Float32Array {
  const tensor = new Float32Array(INPUT_SIZE);
  const inputLength = Math.min(pixelData.length, INPUT_SIZE);
  
  for (let i = 0; i < inputLength; i++) {
    // Squish the raw 0-255 camera integers into the 0.0-1.0 float32 range the model demands
    const pixel = pixelData[i] > 1 ? pixelData[i] / 255.0 : pixelData[i];
    
    // L2CS-Net relies on ImageNet normalization (mean: 0.485, std: 0.229)
    // We approximate it here by shifting to a -1.0 to 1.0 range, which keeps the math stable
    tensor[i] = (pixel - 0.5) * 2.0;  
  }
  return tensor;
}

// ── Inference ─────────────────────────────────────────────────────────────────

export async function runGazeInference(
  state: GazeEngineState, 
  faceImageData: Float32Array | Uint8Array
): Promise<GazeEngineState> {
  
  if (!gazeModel) {
    return { ...state, totalFrames: state.totalFrames + 1, aversionFrames: state.aversionFrames + 1 };
  }

  const tensor = preprocessGazeTensor(faceImageData);

  try {
    const outputs = gazeModel.runSync([tensor]);
    
    // 🚨 DIAGNOSTIC CHECK: What exactly is L2CS-Net giving us?
    const yawOutput = outputs[0] as Float32Array; 
    console.log(`[EngineA] Model returned ${outputs.length} arrays.`);
    console.log(`[EngineA] Array 0 has ${yawOutput.length} numbers. First value: ${yawOutput[0]}`);

    let yawDeg = yawOutput[0]; 

    if (Math.abs(yawDeg) < Math.PI * 2) {
      yawDeg = yawDeg * (180 / Math.PI);
    }

    const isAverting = Math.abs(yawDeg) > YAW_AVERSION_THRESHOLD_DEG;

    return {
      totalFrames: state.totalFrames + 1,
      aversionFrames: state.aversionFrames + (isAverting ? 1 : 0),
      yawAccumulator: state.yawAccumulator + Math.abs(yawDeg),
    };

  } catch (err) {
    console.error('[EngineA] Inference error:', err);
    return state;
  }
}

export function finalizeGazeAnalysis(state: GazeEngineState): GazeAnalysis {
  const total = Math.max(state.totalFrames, 1);
  const gazeRiskScore = state.aversionFrames / total;

  return {
    gazeRiskScore: Math.min(1, Math.max(0, gazeRiskScore)),
    headYawMean: state.yawAccumulator / total,
    irisDeviationMean: 0, // Left at 0 since L2CS-Net handles full gaze, not just the iris
    aversionFrameCount: state.aversionFrames,
    totalFramesAnalyzed: state.totalFrames,
  };
}