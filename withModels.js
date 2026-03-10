// withModels.js
// This script automatically injects the .tflite files into the iOS 
// "Copy Bundle Resources" phase so you never have to open Xcode manually again.

const { withXcodeProject } = require('@expo/config-plugins');

module.exports = function withModels(config) {
  return withXcodeProject(config, (config) => {
    const project = config.modResults;
    const target = project.getFirstTarget().uuid;
    
    // Paths to your models relative to the /ios folder
    const motorModelPath = '../assets/models/motor_risk_model.tflite';
    const faceModelPath = '../assets/models/face_risk_model.tflite';

    // Add them to the Xcode build resources automatically!
    project.addResourceFile(motorModelPath, { target });
    project.addResourceFile(faceModelPath, { target });

    return config;
  });
};