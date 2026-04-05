import Foundation
import AVFoundation
import TensorFlowLite
import Vision

@objc(VideoProcessor)
class VideoProcessor: NSObject {

  @objc(processVideo:withResolver:withRejecter:)
  func processVideo(_ videoPath: String, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
      
      // OPTIMIZATION 1: Use .utility to prevent UI freezing and Priority Inversions
      DispatchQueue.global(qos: .utility).async {
          print("\n---------------------------------------------")
          print("[Swift Native] 🚀 REAL MULTIMODAL AI INFERENCE STARTED")
          
          let cleanPath = videoPath.replacingOccurrences(of: "file://", with: "")
          let fileURL = URL(fileURLWithPath: cleanPath)
          
          // 1. BOOT TFLITE MODELS
          guard let facePath = Bundle.main.path(forResource: "face_risk_model", ofType: "tflite"),
                let motorPath = Bundle.main.path(forResource: "motor_risk_model", ofType: "tflite") else {
              reject("MODELS_MISSING", "Check Target Membership for .tflite files", nil)
              return
          }
          
          do {
              var options = Interpreter.Options()
              options.threadCount = 2
              let faceInterpreter = try Interpreter(modelPath: facePath, options: options)
              let motorInterpreter = try Interpreter(modelPath: motorPath, options: options)
              try faceInterpreter.allocateTensors()
              try motorInterpreter.allocateTensors()
              
              // OPTIMIZATION 2: Pre-allocate Vision requests outside the loop
              let faceRequest = VNDetectFaceLandmarksRequest()
              let bodyRequest = VNDetectHumanBodyPoseRequest()
              
              // 2. OPEN VIDEO
              let asset = AVAsset(url: fileURL)
              guard let track = asset.tracks(withMediaType: .video).first else {
                  reject("NO_VIDEO", "Video track missing", nil)
                  return
              }
              
              let reader = try AVAssetReader(asset: asset)
              let outputSettings: [String: Any] = [
                  kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
              ]
              let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
              reader.add(output)
              reader.startReading()
              
              var frameCount = 0
              var faceRiskAccumulator: Float = 0.0
              var faceInferenceCount = 0
              var gazeAwayCount = 0
              var gazeTotalCount = 0
              var motorSequence: [Float] = []
              let TARGET_MOTOR_FRAMES = 100
              
              print("[Swift Native] Extracting Frames and running Vision/TFLite...")

              // 3. THE MAIN INFERENCE LOOP
              while let sampleBuffer = output.copyNextSampleBuffer() {
                  // OPTIMIZATION 3: Autoreleasepool prevents RAM ballooning and thread hangs
                  autoreleasepool {
                      guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
                      frameCount += 1
                      
                      // --- ENGINE C: FACE PHENOTYPE (Every 30 frames) ---
                      if frameCount % 30 == 0 {
                          if let faceData = self.preprocessFace(pixelBuffer: pixelBuffer) {
                              try? faceInterpreter.copy(faceData, toInputAt: 0)
                              try? faceInterpreter.invoke()
                              if let outputTensor = try? faceInterpreter.output(at: 0) {
                                  let result = outputTensor.data.withUnsafeBytes { $0.load(as: Float.self) }
                                  faceRiskAccumulator += result
                                  faceInferenceCount += 1
                              }
                          }
                      }
                      
                      // --- ENGINE A: GAZE AVERSION (Every 10 frames) ---
                      if frameCount % 10 == 0 {
                          let isLookingAway = self.checkGazeAversion(pixelBuffer: pixelBuffer, request: faceRequest)
                          gazeTotalCount += 1
                          if isLookingAway { gazeAwayCount += 1 }
                      }
                      
                      // --- ENGINE B: MOTOR SKELETON (Collect up to 100 frames) ---
                      if motorSequence.count < (TARGET_MOTOR_FRAMES * 72) {
                          let frameSkeleton = self.extractSkeleton(pixelBuffer: pixelBuffer, request: bodyRequest)
                          motorSequence.append(contentsOf: frameSkeleton)
                      }
                  }
              }
              
              // 4. FINALIZE ENGINE B (MOTOR) LSTM INFERENCE
              var finalMotorRisk: Float = 0.15
              
              // 🚨 FIX 2: Pad with the last known frame, NOT zeros!
              let lastValidFrame = Array(motorSequence.suffix(72))
              
              while motorSequence.count < (TARGET_MOTOR_FRAMES * 72) {
                  if lastValidFrame.count == 72 {
                      motorSequence.append(contentsOf: lastValidFrame) // Child "stands still"
                  } else {
                      motorSequence.append(0.0) // Failsafe if video was completely empty
                  }
              }
              
              let motorData = Data(buffer: UnsafeBufferPointer(start: motorSequence, count: motorSequence.count))
              
              do {
                  try motorInterpreter.copy(motorData, toInputAt: 0)
                  try motorInterpreter.invoke()
                  let motorOutput = try motorInterpreter.output(at: 0)
                  finalMotorRisk = motorOutput.data.withUnsafeBytes { $0.load(as: Float.self) }
              } catch {
                  print("❌ [Swift Native] Motor LSTM Error: \(error)")
              }
              
              // 5. CALCULATE FINAL CLINICAL METRICS
              let avgFaceRisk = faceInferenceCount > 0 ? (faceRiskAccumulator / Float(faceInferenceCount)) : 0.15
              let gazeRisk = gazeTotalCount > 0 ? (Float(gazeAwayCount) / Float(gazeTotalCount)) : 0.15
              let normalizedGazeRisk = min(gazeRisk / 0.4, 1.0)
              
              print("✅ [Swift Native] PIPELINE COMPLETE. Frames processed: \(frameCount)")

              let results: [String: Any] = [
                  "faceRisk": Double(avgFaceRisk),
                  "motorRisk": Double(finalMotorRisk),
                  "gazeRisk": Double(normalizedGazeRisk),
                  "finalScore": (0.4 * Double(normalizedGazeRisk)) + (0.4 * Double(finalMotorRisk)) + (0.2 * Double(avgFaceRisk)),
                  "frameCount": frameCount,
                  "status": "success"
              ]
              
              DispatchQueue.main.async { resolve(results) }
              
          } catch {
              print("❌ [Swift Native] FATAL AI ERROR: \(error)")
              DispatchQueue.main.async { reject("AI_ERR", error.localizedDescription, nil) }
          }
      }
  }

  private func preprocessFace(pixelBuffer: CVPixelBuffer) -> Data? {
      let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
      let scaleX = 224.0 / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
      let scaleY = 224.0 / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
      let resizedImage = sourceImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
      
      let context = CIContext()
      guard let cgImage = context.createCGImage(resizedImage, from: CGRect(x: 0, y: 0, width: 224, height: 224)) else { return nil }
      
      var byteData = Data()
      guard let pixelData = cgImage.dataProvider?.data,
            let dataPtr = CFDataGetBytePtr(pixelData) else { return nil }
      
      for i in stride(from: 0, to: 224 * 224 * 4, by: 4) {
          let r = Float(dataPtr[i]) / 127.5 - 1.0
          let g = Float(dataPtr[i+1]) / 127.5 - 1.0
          let b = Float(dataPtr[i+2]) / 127.5 - 1.0
          var fr = r; var fg = g; var fb = b
          byteData.append(UnsafeBufferPointer(start: &fr, count: 1))
          byteData.append(UnsafeBufferPointer(start: &fg, count: 1))
          byteData.append(UnsafeBufferPointer(start: &fb, count: 1))
      }
      return byteData
  }

  private func checkGazeAversion(pixelBuffer: CVPixelBuffer, request: VNDetectFaceLandmarksRequest) -> Bool {
      let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
      try? handler.perform([request])
      
      guard let results = request.results as? [VNFaceObservation], let face = results.first, let landmarks = face.landmarks else {
          return true
      }
      
      if #available(iOS 15.0, *) {
          if let yaw = face.yaw?.doubleValue {
              if abs(yaw) > 0.35 { return true }
          }
      }
      
      if let leftEye = landmarks.leftEye?.normalizedPoints, let leftPupil = landmarks.leftPupil?.normalizedPoints {
          if !leftEye.isEmpty && !leftPupil.isEmpty {
              let eyeMinX = leftEye.map { $0.x }.min() ?? 0
              let eyeMaxX = leftEye.map { $0.x }.max() ?? 1
              let pupilX = leftPupil.first!.x
              
              let eyeWidth = eyeMaxX - eyeMinX
              let pupilPos = (pupilX - eyeMinX) / (eyeWidth + 0.0001)
              
              if pupilPos < 0.3 || pupilPos > 0.7 { return true }
          }
      }
      return false
  }

  private func extractSkeleton(pixelBuffer: CVPixelBuffer, request: VNDetectHumanBodyPoseRequest) -> [Float] {
      var features = [Float](repeating: 0.0, count: 72)
      let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
      try? handler.perform([request])
      
      guard let observation = request.results?.first as? VNHumanBodyPoseObservation else {
          return features
      }
      
      if let recognizedPoints = try? observation.recognizedPoints(.all) {
          func addPoint(key: VNHumanBodyPoseObservation.JointName, index: Int) {
              if let point = recognizedPoints[key], point.confidence > 0.1 {
                  features[index * 3] = Float(point.location.x)
                  // 🚨 FIX 1: Flip the Apple Y-Axis to match Python/MediaPipe
                  features[index * 3 + 1] = Float(1.0 - point.location.y)
                  features[index * 3 + 2] = 0.0
              }
          }
          
          addPoint(key: .nose, index: 0)
          addPoint(key: .leftEye, index: 1)
          addPoint(key: .rightEye, index: 2)
          addPoint(key: .leftEar, index: 3)
          addPoint(key: .rightEar, index: 4)
          addPoint(key: .leftShoulder, index: 11)
          addPoint(key: .rightShoulder, index: 12)
          addPoint(key: .leftElbow, index: 13)
          addPoint(key: .rightElbow, index: 14)
          addPoint(key: .leftWrist, index: 15)
          addPoint(key: .rightWrist, index: 16)
          
          let midShoulderX = (features[11 * 3] + features[12 * 3]) / 2.0
          let midShoulderY = (features[11 * 3 + 1] + features[12 * 3 + 1]) / 2.0
          
          if midShoulderX > 0 && midShoulderY > 0 {
              for i in 0..<24 {
                  if features[i*3] != 0.0 {
                      features[i*3] -= midShoulderX
                      features[i*3+1] -= midShoulderY
                  }
              }
          }
      }
      return features
  }

  @objc static func requiresMainQueueSetup() -> Bool { return false }
}
