//
//  VideoProcessor.m
//  ASDScreening
//
//  Created by Pranit Rathkanthiwar on 10/03/26.
//

#import <React/RCTBridgeModule.h>

// This tells React Native to look for our Swift class
@interface RCT_EXTERN_MODULE(VideoProcessor, NSObject)

// This exposes our Swift processVideo function to JavaScript
RCT_EXTERN_METHOD(processVideo:(NSString *)videoPath
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)

@end
