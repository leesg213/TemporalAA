/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Header for renderer class which performs Metal setup and per frame rendering
*/

#import <MetalKit/MetalKit.h>

@interface AAPLRenderer : NSObject<MTKViewDelegate>

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView;
-(void)OnTAAEnableButton:(NSButton *)button;
-(void)OnAnimationButton:(NSButton *)button;
-(void)OnMagnifierEnableButton:(NSButton *)button;
-(void)OnStepButton:(NSButton *)button;
-(void)OnMouseDown:(NSEvent*)event;
@end
