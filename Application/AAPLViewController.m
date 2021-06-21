/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation of the cross-platform view controller
*/

#import "AAPLViewController.h"
#import "AAPLRenderer.h"

@implementation AAPLViewController
{
    MTKView *_view;

    AAPLRenderer *_renderer;
}
-(BOOL)acceptsFirstResponder
{
    return YES;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    // Set the view to use the default device
    _view = (MTKView *)self.view;
    
    _view.device = MTLCreateSystemDefaultDevice();
    
    NSAssert(_view.device, @"Metal is not supported on this device");

#if TARGET_IOS
    BOOL supportsLayerSelection = NO;

    supportsLayerSelection = [_view.device supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily5_v1];

    NSAssert(supportsLayerSelection, @"Sample requires iOS_GPUFamily5_v1 for Layer Selection");
#endif
    
    _renderer = [[AAPLRenderer alloc] initWithMetalKitView:_view];
    
    NSAssert(_renderer, @"Renderer failed initialization");
    
    [_renderer mtkView:_view drawableSizeWillChange:_view.drawableSize];

    _view.delegate = _renderer;
}

#if defined(TARGET_IOS)
- (BOOL)prefersHomeIndicatorAutoHidden
{
    return YES;
}
#endif

- (IBAction)OnAnimationButton:(NSButton *)sender {
    [_renderer OnAnimationButton:sender];
}
- (IBAction)OnTAAEnableButton:(NSButton *)sender {
    [_renderer OnTAAEnableButton:sender];
}
- (IBAction)OnMagnifierEnableButton:(NSButton *)sender {
    [_renderer OnMagnifierEnableButton:sender];
}
-(void)mouseDown:(NSEvent *)event
{
    [_renderer OnMouseDown:event];
}
- (IBAction)OnStepButton:(NSButton *)sender {
    [_renderer OnStepButton:sender];
}
@end
