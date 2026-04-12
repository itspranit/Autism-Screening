import Foundation
import AVFoundation
import TensorFlowLite
import Vision

@objc(VideoProcessor)
class VideoProcessor: NSObject {
  
  @objc(processVideo:withResolver:withRejecter:)
  func processVideo(_ videoPath: String, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
    
    DispatchQueue.global(qos: .utility).async {
      let cleanPath = videoPath.replacingOccurrences(of: "file://", with: "")
      let fileURL = URL(fileURLWithPath: cleanPath)
      
      guard let facePath = Bundle.main.path(forResource: "face_risk_model", ofType: "tflite"),
            let motorPath = Bundle.main.path(forResource: "motor_risk_model", ofType: "tflite"),
            let gazePath = Bundle.main.path(forResource: "gaze_model", ofType: "tflite") else {
        reject("MODELS_MISSING", "Check Target Membership", nil)
        return
      }
      
      do {
        var options = Interpreter.Options()
        options.threadCount = 2
        let faceInterpreter = try Interpreter(modelPath: facePath, options: options)
        let motorInterpreter = try Interpreter(modelPath: motorPath, options: options)
        let gazeInterpreter = try Interpreter(modelPath: gazePath, options: options)
        
        try faceInterpreter.allocateTensors()
        try motorInterpreter.allocateTensors()
        try gazeInterpreter.allocateTensors()
        
        let bodyRequest = VNDetectHumanBodyPoseRequest()
        let faceRequest = VNDetectFaceRectanglesRequest()
        
        let asset = AVAsset(url: fileURL)
        guard let track = asset.tracks(withMediaType: .video).first else { return }
        
        let reader = try AVAssetReader(asset: asset)
        let outputSettings: [String: Any] = [ kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA) ]
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
        var motorRiskAccumulator: Float = 0.0
        var motorChunkCount = 0
        var chunkScores: [Float] = []
        
        var lockedOrientation: CGImagePropertyOrientation? = nil
        // 🚨 FIX 1: Force .up (Portrait) FIRST to prevent sideways skeleton corruption
        let possibleOrientations: [CGImagePropertyOrientation] = [.up, .right, .left, .down, .upMirrored, .rightMirrored, .leftMirrored, .downMirrored]
        
        var lastValidSkeleton = [Float](repeating: 0.0, count: 72)
        
        while let sampleBuffer = output.copyNextSampleBuffer() {
          autoreleasepool {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            frameCount += 1
            
            // ENGINE C: Face Phenotype (1Hz)
            if frameCount % 30 == 0 {
              if let faceData = self.preprocessImage(pixelBuffer: pixelBuffer, targetSize: 224, faceRequest: faceRequest) {
                try? faceInterpreter.copy(faceData, toInputAt: 0)
                try? faceInterpreter.invoke()
                if let t = try? faceInterpreter.output(at: 0) {
                  let floatCount = t.data.count / MemoryLayout<Float>.stride
                  let vals = t.data.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(start: $0.baseAddress!.assumingMemoryBound(to: Float.self), count: floatCount)) }
                  faceRiskAccumulator += vals[0]
                  faceInferenceCount += 1
                }
              }
            }
            
            // ENGINE A: Gaze Aversion (10Hz)
            if frameCount % 10 == 0 {
              let isLookingAway = self.checkDeepGaze(pixelBuffer: pixelBuffer, interpreter: gazeInterpreter, faceRequest: faceRequest)
              gazeTotalCount += 1
              if isLookingAway { gazeAwayCount += 1 }
            }
            
            // ENGINE B: Motor Skeleton (30Hz)
            if lockedOrientation == nil {
              for orientation in possibleOrientations {
                let (skeleton, found) = self.extractSkeleton(pixelBuffer: pixelBuffer, request: bodyRequest, orientation: orientation)
                if found {
                  lockedOrientation = orientation
                  lastValidSkeleton = skeleton
                  motorSequence.append(contentsOf: skeleton)
                  break
                }
              }
            } else {
              let (skeleton, found) = self.extractSkeleton(pixelBuffer: pixelBuffer, request: bodyRequest, orientation: lockedOrientation!)
              if found {
                lastValidSkeleton = skeleton
                motorSequence.append(contentsOf: skeleton)
              } else {
                motorSequence.append(contentsOf: lastValidSkeleton)
              }
            }
            
            // ENGINE B INFERENCE
            if motorSequence.count == (TARGET_MOTOR_FRAMES * 72) {
              let motorData = motorSequence.withUnsafeBufferPointer { Data(buffer: $0) }
              do {
                try motorInterpreter.copy(motorData, toInputAt: 0)
                try motorInterpreter.invoke()
                if let t = try? motorInterpreter.output(at: 0) {
                  let floatCount = t.data.count / MemoryLayout<Float>.stride
                  let vals = t.data.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(start: $0.baseAddress!.assumingMemoryBound(to: Float.self), count: floatCount)) }
                  
                  // Handles both Keras Softmax [Normal, Risk] and Sigmoid [Normal]
                  let rawOutput = vals[0]
                  let chunkRisk = floatCount >= 2 ? vals[1] : (1.0 - rawOutput)
                  
                  chunkScores.append(chunkRisk)
                  motorRiskAccumulator += chunkRisk
                  motorChunkCount += 1
                  
                  print("🧠 [Engine B] Chunk \(motorChunkCount) - Raw: \(rawOutput) -> True Risk: \(chunkRisk)")
                }
              } catch { print("Motor LSTM Error: \(error)") }
              
              motorSequence.removeAll(keepingCapacity: true)
            }
          }
        }
        
        // Continuous Chunking Math (Drop the Stop Button Glitch)
        var finalMotorRisk: Float = 0.05
        if motorChunkCount > 0 {
          if motorChunkCount > 1 {
            chunkScores.removeLast()
            let safeSum = chunkScores.reduce(0, +)
            finalMotorRisk = safeSum / Float(chunkScores.count)
          } else {
            finalMotorRisk = chunkScores[0]
          }
        }
        
        if finalMotorRisk > 1.0 { finalMotorRisk = 1.0 }
        if finalMotorRisk < 0.01 { finalMotorRisk = 0.01 }
        
        let avgFaceRisk = faceInferenceCount > 0 ? (faceRiskAccumulator / Float(faceInferenceCount)) : 0.15
        let gazeRisk = gazeTotalCount > 0 ? (Float(gazeAwayCount) / Float(gazeTotalCount)) : 0.0
        
        print("✅ [Swift Native] PIPELINE COMPLETE. Frames: \(frameCount). Avg Motor Risk: \(finalMotorRisk)")
        
        let results: [String: Any] = [
          "faceRisk": Double(avgFaceRisk),
          "motorRisk": Double(finalMotorRisk),
          "gazeRisk": Double(gazeRisk),
          "finalScore": (0.4 * Double(gazeRisk)) + (0.4 * Double(finalMotorRisk)) + (0.2 * Double(avgFaceRisk))
        ]
        
        DispatchQueue.main.async { resolve(results) }
        
      } catch {
        DispatchQueue.main.async { reject("AI_ERR", error.localizedDescription, nil) }
      }
    }
  }
  
  // MARK: - AI Helpers
  
  private func preprocessImage(pixelBuffer: CVPixelBuffer, targetSize: Int, faceRequest: VNDetectFaceRectanglesRequest) -> Data? {
    let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
    try? handler.perform([faceRequest])
    
    var cropRect = sourceImage.extent
    if let firstFace = faceRequest.results?.first {
      let w = sourceImage.extent.width
      let h = sourceImage.extent.height
      let box = firstFace.boundingBox
      let centerX = box.origin.x + (box.width / 2.0)
      let centerY = box.origin.y + (box.height / 2.0)
      
      let size = max(box.width, box.height) * 1.5
      let x = max(0, (centerX - size / 2.0) * w)
      let y = max(0, (centerY - size / 2.0) * h)
      let finalSide = min(min(w - x, size * w), min(h - y, size * h))
      cropRect = CGRect(x: x, y: y, width: finalSide, height: finalSide)
    }
    
    let croppedImage = sourceImage.cropped(to: cropRect)
    let movedImage = croppedImage.transformed(by: CGAffineTransform(translationX: -cropRect.origin.x, y: -cropRect.origin.y))
    let scaleX = CGFloat(targetSize) / cropRect.width
    let scaleY = CGFloat(targetSize) / cropRect.height
    let resizedImage = movedImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    
    let context = CIContext()
    guard let cgImage = context.createCGImage(resizedImage, from: CGRect(x: 0, y: 0, width: targetSize, height: targetSize)) else { return nil }
    guard let pixelData = cgImage.dataProvider?.data, let dataPtr = CFDataGetBytePtr(pixelData) else { return nil }
    
    var floats = [Float]()
    floats.reserveCapacity(targetSize * targetSize * 3)
    for i in stride(from: 0, to: targetSize * targetSize * 4, by: 4) {
      floats.append(Float(dataPtr[i+2]) / 127.5 - 1.0)
      floats.append(Float(dataPtr[i+1]) / 127.5 - 1.0)
      floats.append(Float(dataPtr[i]) / 127.5 - 1.0)
    }
    return floats.withUnsafeBufferPointer { Data(buffer: $0) }
  }
  
  private func decodeL2CS28Bin(tensor: Tensor) -> Float {
    let floatCount = tensor.data.count / 4
    let logits = tensor.data.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(start: $0.baseAddress!.assumingMemoryBound(to: Float.self), count: floatCount)) }
    if floatCount == 1 { return logits[0] }
    
    let maxLogit = logits.max() ?? 0
    var expSum: Float = 0
    var exps = [Float](repeating: 0, count: 28)
    
    for i in 0..<min(28, floatCount) {
      exps[i] = exp(logits[i] - maxLogit)
      expSum += exps[i]
    }
    var expectedBin: Float = 0
    for i in 0..<min(28, floatCount) { expectedBin += (exps[i] / expSum) * Float(i) }
    return (expectedBin * 3.0 - 42.0) * (.pi / 180.0)
  }
  
  private func checkDeepGaze(pixelBuffer: CVPixelBuffer, interpreter: Interpreter, faceRequest: VNDetectFaceRectanglesRequest) -> Bool {
    guard let faceData = self.preprocessImage(pixelBuffer: pixelBuffer, targetSize: 448, faceRequest: faceRequest) else { return false }
    do {
      try interpreter.copy(faceData, toInputAt: 0)
      try interpreter.invoke()
      var yaw: Float = 0.0
      if interpreter.outputTensorCount >= 2 {
        yaw = decodeL2CS28Bin(tensor: try interpreter.output(at: 0))
      } else {
        let t = try interpreter.output(at: 0)
        let vals = t.data.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(start: $0.baseAddress!.assumingMemoryBound(to: Float.self), count: 2)) }
        yaw = vals[1]
      }
      if abs(yaw) > 0.15 { return true }
      return false
    } catch { return false }
  }
  
  private func extractSkeleton(pixelBuffer: CVPixelBuffer, request: VNDetectHumanBodyPoseRequest, orientation: CGImagePropertyOrientation) -> ([Float], Bool) {
    var features = [Float](repeating: 0.0, count: 72)
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
    try? handler.perform([request])
    
    guard let observation = request.results?.first as? VNHumanBodyPoseObservation,
          let points = try? observation.recognizedPoints(.all) else { return (features, false) }
    
    func addP(key: VNHumanBodyPoseObservation.JointName, idx: Int) {
      if let p = points[key], p.confidence > 0.1 {
        features[idx*3] = Float(p.location.x)
        features[idx*3+1] = Float(1.0 - p.location.y) // Vision to MediaPipe Y-flip
      }
    }
    
    addP(key: .nose, idx: 0); addP(key: .leftEye, idx: 1); addP(key: .rightEye, idx: 2)
    addP(key: .leftEar, idx: 3); addP(key: .rightEar, idx: 4)
    addP(key: .leftShoulder, idx: 11); addP(key: .rightShoulder, idx: 12)
    addP(key: .leftElbow, idx: 13); addP(key: .rightElbow, idx: 14)
    addP(key: .leftWrist, idx: 15); addP(key: .rightWrist, idx: 16)
    
    // 🚨 THE FIX: NOSE-ANCHORED RELATIVE COORDINATES
    // The LSTM cannot understand absolute screen positions. We must make the Nose (0,0).
    let noseX = features[0], noseY = features[1]
    let midShoulderX = (features[11*3] + features[12*3]) / 2.0
    let midShoulderY = (features[11*3+1] + features[12*3+1]) / 2.0
    
    // If nose isn't visible, fall back to the chest
    let anchorX = noseX > 0 ? noseX : midShoulderX
    let anchorY = noseY > 0 ? noseY : midShoulderY
    
    var foundBody = false
    if anchorX > 0 && anchorY > 0 {
      foundBody = true
      for i in 0..<24 {
        // If the joint exists, subtract the anchor to make it relative!
        if features[i*3] != 0 || features[i*3+1] != 0 {
          features[i*3] = features[i*3] - anchorX
          features[i*3+1] = features[i*3+1] - anchorY
        }
      }
    }
    
    return (features, foundBody)
  }
}
