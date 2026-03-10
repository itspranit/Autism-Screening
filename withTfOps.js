// withTfOps.js
const { withXcodeProject } = require('@expo/config-plugins');

module.exports = function withTfOps(config) {
  return withXcodeProject(config, async (config) => {
    const xcodeProject = config.modResults;
    const target = xcodeProject.getFirstTarget().uuid;
    
    // Add the linker flag
    xcodeProject.addBuildProperty('OTHER_LDFLAGS', '"-force_load $(PODS_ROOT)/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.xcframework/ios-arm64/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps"', target);
    
    return config;
  });
};