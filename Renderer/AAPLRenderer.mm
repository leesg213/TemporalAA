/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation of renderer class which performs Metal setup and per frame rendering
*/

#import <ModelIO/ModelIO.h>
#import <MetalKit/MetalKit.h>
#import <vector>

#import "AAPLRenderer.h"
#import "AAPLMesh.h"
#import "AAPLMathUtilities.h"

// Include header shared between C code here, which executes Metal API commands,
// and .metal files
#import "AAPLShaderTypes.h"

#include "AAPLRendererUtils.h"

static const NSUInteger    MaxBuffersInFlight       = 3;  // Number of in-flight command buffers
static const NSUInteger    MaxActors                = 32; // Max possible actors
static const NSUInteger    MaxVisibleFaces          = 5;  // Number of faces an actor could be visible in
static const NSUInteger    CubemapResolution        = 256;
static const vector_float3 SceneCenter              = (vector_float3){0.f, -250.f, 1000.f};
static const vector_float3 CameraDistanceFromCenter = (vector_float3){0.f, 300.f, -550.f};
static const vector_float3 CameraRotationAxis       = (vector_float3){0,1,0};
static const float         CameraRotationSpeed      = 0;//0.0025f;//*8;

// Main class performing the rendering
@implementation AAPLRenderer
{
    dispatch_semaphore_t _inFlightSemaphore;
    id<MTLDevice>       _device;
    id<MTLCommandQueue> _commandQueue;

    // Current frame number modulo MaxBuffersInFlight.
    // Tells which buffer index is writable in buffer arrays.
    uint8_t _uniformBufferIndex;

    // CPU app-specific data
    Camera                           _cameraFinal;
    CameraProbe                      _cameraReflection;
    AAPLActorData *                  _reflectiveActor;
    NSMutableArray <AAPLActorData*>* _actorData;
    std::vector<matrix_float4x4> _actorPrevModelMatrix;

    // Dynamic GPU buffers
    id<MTLBuffer> _frameParamsBuffers                [MaxBuffersInFlight]; // frame-constant parameters
    id<MTLBuffer> _viewportsParamsBuffers      [MaxBuffersInFlight]; // frame-constant parameters, final viewport
    id<MTLBuffer> _actorsParamsBuffers               [MaxBuffersInFlight]; // per-actor parameters
    id<MTLBuffer> _magnifierParamsBuffers [MaxBuffersInFlight];
    
    id<MTLDepthStencilState> _depthState;
    
    id<MTLTexture> _historybuffers[2];
    id<MTLTexture> _framebuffer;
    id<MTLTexture> _framedepthbuffer;
    id<MTLTexture> _velocitybuffer;

    id<MTLRenderPipelineState> _pipeline_TAA_Resolve;
    id<MTLRenderPipelineState> _pipeline_blit;
    id<MTLRenderPipelineState> _pipeline_magnifier;

    vector_float2 _viewSize;
    bool _enableTAA;
    bool _enableAnimation;
    bool _enableMagnifier;
    bool _historyReset;
    bool _stepByStep;
    bool _advanceToNextFrame;
    matrix_float4x4 _viewProjMatrix;
    matrix_float4x4 _jitterMatrix;
    
    float _magnifier_pos_x, _magnifier_pos_y;
    float _magnifier_size_x, _magnifier_size_y;
}

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView
{
    self = [super init];
    if(self)
    {
        _device = mtkView.device;
        _inFlightSemaphore = dispatch_semaphore_create(MaxBuffersInFlight);
        
        _enableTAA = false;
        _enableAnimation = false;
        _enableMagnifier = false;
        _historyReset = false;
        _stepByStep = false;
        _advanceToNextFrame = false;
        
        _magnifier_pos_x = mtkView.drawableSize.width/2;
        _magnifier_pos_y = mtkView.drawableSize.height/2;
        _magnifier_size_x = 600;
        _magnifier_size_y = 600;
        
        [self loadMetalWithMetalKitView:mtkView];
        [self loadAssetsWithMetalKitView:mtkView];
    }

    return self;
}
-(void)OnTAAEnableButton:(NSButton *)button
{
    _enableTAA = [button state] == NSControlStateValueOn;
    _historyReset = true;
}
-(void)OnAnimationButton:(NSButton *)button
{
    _enableAnimation = [button state] == NSControlStateValueOn;
}
-(void)OnMagnifierEnableButton:(NSButton *)button
{
    _enableMagnifier = [button state] == NSControlStateValueOn;
}
-(void)OnStepButton:(NSButton *)button
{
    _stepByStep = true;
    _advanceToNextFrame = true;
}
-(float)generateHaltonSequence:(int)base :(int*)n_in :(int*)d_in
{
    int n = *n_in;
    int d = *d_in;
    
    int x = d - n;
    if (x==1)
    {
        n = 1;
        d *= base;
    }
    else
    {
        int y = d/base;
        while(x<=y)
        {
            y = y/base;
        }
        
        n = (base+1)*y-x;
    }
    
    *n_in = n;
    *d_in = d;
    
    float result = n/(float)d;
    return result;
}
-(vector_float2) generateJitterOffset
{
    static int n_2 = 0, d_2 = 1, n_3 = 0, d_3 = 1;
    static int count = 0;
    
    vector_float2 result;
    
    
    if(_enableTAA)
    {
        result.x = [self generateHaltonSequence:2 :&n_2 :&d_2];
        result.y = [self generateHaltonSequence:3 :&n_3 :&d_3];

        result.x = ((result.x-0.5f) / _viewSize.x) *2;
        result.y = ((result.y-0.5f) / _viewSize.y) *2;
    }
    else
    {
        result.x = 0;
        result.y = 0;
    }
    
    ++count;
    if(count == 16)
    {
        n_2 = 0;
        d_2 = 1;
        n_3 = 0;
        d_3 = 1;
        count = 0;
    }

    return result;
}
- (void)loadMetalWithMetalKitView:(nonnull MTKView *)mtkView
{
    // Create and load our basic Metal state objects

    // We allocate MaxBuffersInFlight instances of uniform buffers. This allows us to
    // update uniforms as a ring (i.e. triple buffer the uniform data) so that the GPU
    // reads from one slot in the ring while the CPU writes to another.

    for (int i = 0; i < MaxBuffersInFlight; i++)
    {
        id<MTLBuffer> frameParamsBuffer =
            [_device newBufferWithLength: sizeof(FrameParams)
                                 options: MTLResourceStorageModeShared];
        frameParamsBuffer.label = [NSString stringWithFormat:@"frameParams[%i]", i];
        _frameParamsBuffers[i] = frameParamsBuffer;
        
        id<MTLBuffer> magnifierParamsBuffer =
            [_device newBufferWithLength: sizeof(MagnifierParams)
                                 options: MTLResourceStorageModeShared];
        magnifierParamsBuffer.label = [NSString stringWithFormat:@"magnifierParams[%i]", i];
        _magnifierParamsBuffers[i] = magnifierParamsBuffer;

        id<MTLBuffer> viewportParamsBuffer =
            [_device newBufferWithLength: sizeof(ViewportParams)
                                 options: MTLResourceStorageModeShared];
        viewportParamsBuffer .label = [NSString stringWithFormat:@"viewportParams[%i]", i];
        _viewportsParamsBuffers[i] = viewportParamsBuffer;

        // This buffer will contain every actor's data required by shaders.
        //
        // When rendering a batch (aka an actor), the shader will access its actor data
        //   through a reference, without knowing about the actual offset of the data within
        //   the buffer.
        // This is done by, before each draw call, setting the buffer in the Metal framework with
        //   an explicit offset when setting the buffer
        //
        // As this offset _has_ to be 256 bytes aligned, that means we'll need to round up
        // the size of an ActorData to the next multiple of 256.
        id<MTLBuffer> actorParamsBuffer =
            [_device newBufferWithLength: Align<BufferOffsetAlign> (sizeof(ActorParams)) * MaxActors
                                 options: MTLResourceStorageModeShared];
        actorParamsBuffer.label = [NSString stringWithFormat:@"actorsParams[%i]", i];
        _actorsParamsBuffers[i] = actorParamsBuffer;
    }

    mtkView.sampleCount               = 1;
    mtkView.colorPixelFormat          = MTLPixelFormatBGRA8Unorm_sRGB;
    mtkView.depthStencilPixelFormat   = MTLPixelFormatDepth32Float;

    MTLDepthStencilDescriptor *depthStateDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthStateDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthStateDesc.depthWriteEnabled    = YES;

    _depthState = [_device newDepthStencilStateWithDescriptor:depthStateDesc];

    // Create the command queue
    _commandQueue = [_device newCommandQueue];
}

-(void)createTAAResource:(nonnull MTKView*)mtkView
{
    NSError *error = NULL;
    
    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
    
    {
        MTLTextureDescriptor* textureDesc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mtkView.colorPixelFormat width:mtkView.drawableSize.width height:mtkView.drawableSize.height mipmapped:NO];
        textureDesc.storageMode = MTLStorageModePrivate;
        textureDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

        _framebuffer = [_device newTextureWithDescriptor:textureDesc];
    }
    
    {
        MTLTextureDescriptor* textureDesc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mtkView.colorPixelFormat width:mtkView.drawableSize.width height:mtkView.drawableSize.height mipmapped:NO];
        textureDesc.storageMode = MTLStorageModePrivate;
        textureDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

        for(size_t i = 0;i<2;++i)
        {
            _historybuffers[i] = [_device newTextureWithDescriptor:textureDesc];
        }
    }
    
    {
        MTLTextureDescriptor* textureDesc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:mtkView.drawableSize.width height:mtkView.drawableSize.height mipmapped:NO];
        textureDesc.storageMode = MTLStorageModePrivate;
        textureDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

        _framedepthbuffer = [_device newTextureWithDescriptor:textureDesc];
    }
    {
        MTLTextureDescriptor* textureDesc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float width:mtkView.drawableSize.width height:mtkView.drawableSize.height mipmapped:NO];
        textureDesc.storageMode = MTLStorageModePrivate;
        textureDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

        _velocitybuffer = [_device newTextureWithDescriptor:textureDesc];
    }
    
    {
        MTLRenderPipelineDescriptor *renderDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        renderDescriptor.sampleCount = 1;
        renderDescriptor.vertexFunction = [defaultLibrary newFunctionWithName:@"fullscreenQuadVertex"];
        renderDescriptor.fragmentFunction = [defaultLibrary newFunctionWithName:@"TAA_ResolveFragment"];
        renderDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat;

        _pipeline_TAA_Resolve = [_device newRenderPipelineStateWithDescriptor:renderDescriptor error:&error];
    }
    {
        MTLRenderPipelineDescriptor *renderDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        renderDescriptor.sampleCount = 1;
        renderDescriptor.vertexFunction = [defaultLibrary newFunctionWithName:@"fullscreenQuadVertex"];
        renderDescriptor.fragmentFunction = [defaultLibrary newFunctionWithName:@"BlitFragment"];
        renderDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat;

        _pipeline_blit = [_device newRenderPipelineStateWithDescriptor:renderDescriptor error:&error];
    }
    {
        MTLRenderPipelineDescriptor *renderDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        renderDescriptor.sampleCount = 1;
        renderDescriptor.vertexFunction = [defaultLibrary newFunctionWithName:@"magnifierVertex"];
        renderDescriptor.fragmentFunction = [defaultLibrary newFunctionWithName:@"magnifierFragment"];
        renderDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat;

        _pipeline_magnifier = [_device newRenderPipelineStateWithDescriptor:renderDescriptor error:&error];
    }

}

- (void)loadAssetsWithMetalKitView:(nonnull MTKView*)mtkView
{
    //-------------------------------------------------------------------------------------------
    // Create a vertex descriptor for our Metal pipeline. Specifies the layout of vertices the
    //   pipeline should expect.  The layout below keeps attributes used to calculate vertex shader
    //   output (world position, skinning, tweening weights...) separate from other
    //   attributes (texture coordinates, normals).  This generally maximizes pipeline efficiency.

    MTLVertexDescriptor* mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init];

    // Positions.
    mtlVertexDescriptor.attributes[VertexAttributePosition].format       = MTLVertexFormatFloat3;
    mtlVertexDescriptor.attributes[VertexAttributePosition].offset       = 0;
    mtlVertexDescriptor.attributes[VertexAttributePosition].bufferIndex  = BufferIndexMeshPositions;

    // Texture coordinates.
    mtlVertexDescriptor.attributes[VertexAttributeTexcoord].format       = MTLVertexFormatFloat2;
    mtlVertexDescriptor.attributes[VertexAttributeTexcoord].offset       = 0;
    mtlVertexDescriptor.attributes[VertexAttributeTexcoord].bufferIndex  = BufferIndexMeshGenerics;

    // Normals.
    mtlVertexDescriptor.attributes[VertexAttributeNormal].format         = MTLVertexFormatHalf4;
    mtlVertexDescriptor.attributes[VertexAttributeNormal].offset         = 8;
    mtlVertexDescriptor.attributes[VertexAttributeNormal].bufferIndex    = BufferIndexMeshGenerics;

    // Tangents
    mtlVertexDescriptor.attributes[VertexAttributeTangent].format        = MTLVertexFormatHalf4;
    mtlVertexDescriptor.attributes[VertexAttributeTangent].offset        = 16;
    mtlVertexDescriptor.attributes[VertexAttributeTangent].bufferIndex   = BufferIndexMeshGenerics;

    // Bitangents
    mtlVertexDescriptor.attributes[VertexAttributeBitangent].format      = MTLVertexFormatHalf4;
    mtlVertexDescriptor.attributes[VertexAttributeBitangent].offset      = 24;
    mtlVertexDescriptor.attributes[VertexAttributeBitangent].bufferIndex = BufferIndexMeshGenerics;

    // Position Buffer Layout
    mtlVertexDescriptor.layouts[BufferIndexMeshPositions].stride         = 12;
    mtlVertexDescriptor.layouts[BufferIndexMeshPositions].stepRate       = 1;
    mtlVertexDescriptor.layouts[BufferIndexMeshPositions].stepFunction   = MTLVertexStepFunctionPerVertex;

    // Generic Attribute Buffer Layout
    mtlVertexDescriptor.layouts[BufferIndexMeshGenerics].stride          = 32;
    mtlVertexDescriptor.layouts[BufferIndexMeshGenerics].stepRate        = 1;
    mtlVertexDescriptor.layouts[BufferIndexMeshGenerics].stepFunction    = MTLVertexStepFunctionPerVertex;

    //-------------------------------------------------------------------------------------------

    NSError *error = NULL;

    // Load all the shader files with a metal file extension in the project
    id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];

    // Create a reusable pipeline state
    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineStateDescriptor.vertexDescriptor                = mtlVertexDescriptor;
    pipelineStateDescriptor.inputPrimitiveTopology          = MTLPrimitiveTopologyClassTriangle;
    pipelineStateDescriptor.vertexFunction =
        [defaultLibrary newFunctionWithName:@"vertexTransform"];
    pipelineStateDescriptor.fragmentFunction =
        [defaultLibrary newFunctionWithName:@"fragmentLighting"];
    pipelineStateDescriptor.sampleCount                     = mtkView.sampleCount;
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat;
    pipelineStateDescriptor.colorAttachments[1].pixelFormat = MTLPixelFormatRG32Float;
    pipelineStateDescriptor.depthAttachmentPixelFormat      = mtkView.depthStencilPixelFormat;

    pipelineStateDescriptor.label = @"TemplePipeline";
    id<MTLRenderPipelineState> templePipelineState =
        [_device newRenderPipelineStateWithDescriptor: pipelineStateDescriptor error:&error];
    
    NSAssert(templePipelineState, @"Failed to create pipeline state: %@", error);

    pipelineStateDescriptor.label = @"GroundPipeline";
    pipelineStateDescriptor.fragmentFunction =
        [defaultLibrary newFunctionWithName:@"fragmentGround"];
    id<MTLRenderPipelineState> groundPipelineState  =
        [_device newRenderPipelineStateWithDescriptor: pipelineStateDescriptor error:&error];
    
    NSAssert(groundPipelineState, @"Failed to create pipeline state: %@", error);
    
    pipelineStateDescriptor.label = @"ChromePipeline";
    pipelineStateDescriptor.sampleCount = 1;
    pipelineStateDescriptor.fragmentFunction =
        [defaultLibrary newFunctionWithName:@"fragmentChromeLighting"];
    id<MTLRenderPipelineState> chromePipelineState  =
        [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];

    NSAssert(chromePipelineState, @"Failed to create pipeline state: %@", error);

    _cameraReflection.distanceNear = 50.f;
    _cameraReflection.distanceFar  = 3000.f;
    _cameraReflection.position     = SceneCenter;

    _cameraFinal.rotation = 0;

    //-------------------------------------------------------------------------------------------
    // Create and load our assets into Metal objects including meshes and textures

    // Create a Model I/O vertexDescriptor so that we format/layout our Model I/O mesh vertices to
    //   fit our Metal render pipeline's vertex descriptor layout
    MDLVertexDescriptor *modelIOVertexDescriptor =
    MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor);

    // Indicate how each Metal vertex descriptor attribute maps to each Model I/O  attribute
    modelIOVertexDescriptor.attributes[VertexAttributePosition].name  = MDLVertexAttributePosition;
    modelIOVertexDescriptor.attributes[VertexAttributeTexcoord].name  = MDLVertexAttributeTextureCoordinate;
    modelIOVertexDescriptor.attributes[VertexAttributeNormal].name    = MDLVertexAttributeNormal;
    modelIOVertexDescriptor.attributes[VertexAttributeTangent].name   = MDLVertexAttributeTangent;
    modelIOVertexDescriptor.attributes[VertexAttributeBitangent].name = MDLVertexAttributeBitangent;

    NSURL *modelFileURL = [[NSBundle mainBundle] URLForResource: @"Models/Temple.obj"
                                                  withExtension:  nil];

    NSAssert(modelFileURL,
             @"Could not find model file (%@) in bundle",
             modelFileURL.absoluteString);

    MDLAxisAlignedBoundingBox templeAabb;
    NSArray <AAPLMesh*>* templeMeshes = [AAPLMesh newMeshesFromUrl: modelFileURL
                                           modelIOVertexDescriptor: modelIOVertexDescriptor
                                                       metalDevice: _device
                                                             error: &error
                                                              aabb: templeAabb];
    
    NSAssert(templeMeshes, @"Could not create meshes from model file: %@", modelFileURL.absoluteString);
    
    vector_float4 templeBSphere;
    templeBSphere.xyz = (templeAabb.maxBounds + templeAabb.minBounds)*0.5;
    templeBSphere.w = vector_length ((templeAabb.maxBounds - templeAabb.minBounds)*0.5);


    MTKMeshBufferAllocator *meshBufferAllocator =
        [[MTKMeshBufferAllocator alloc] initWithDevice:_device];

    MDLMesh* mdlSphere = [MDLMesh newEllipsoidWithRadii: 200.0
                                         radialSegments: 30
                                       verticalSegments: 20
                                           geometryType: MDLGeometryTypeTriangles
                                          inwardNormals: false
                                             hemisphere: false
                                              allocator: meshBufferAllocator];

    vector_float4 sphereBSphere;
    sphereBSphere.xyz = (vector_float3){0,0,0};
    sphereBSphere.w = 200.f;

    NSArray <AAPLMesh*>* sphereMeshes = [AAPLMesh newMeshesFromObject: mdlSphere
                                              modelIOVertexDescriptor: modelIOVertexDescriptor
                                                metalKitTextureLoader: NULL
                                                          metalDevice: _device
                                                                error: &error ];

    NSAssert(sphereMeshes, @"Could not create sphere meshes: %@", error);

    MDLMesh* mdlGround = [MDLMesh newPlaneWithDimensions: {100000.f, 100000.f}
                                                segments: {1,1}
                                            geometryType: MDLGeometryTypeTriangles
                                               allocator: meshBufferAllocator];

    vector_float4 groundBSphere;
    groundBSphere.xyz = (vector_float3){0,0,0};
    groundBSphere.w = 1415.f;

    NSArray <AAPLMesh*>* groundMeshes = [AAPLMesh newMeshesFromObject: mdlGround
                                              modelIOVertexDescriptor: modelIOVertexDescriptor
                                                metalKitTextureLoader: NULL
                                                          metalDevice: _device
                                                                error: &error ];
    
    NSAssert(groundMeshes, @"Could not create ground meshes: %@", error);

    // Finally, we create the actor list :
    _actorData = [NSMutableArray new];
    /*
    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3) {-1000, -150.f, 1000.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 1.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {1.f, 1.f, 1.f};
    _actorData.lastObject.bSphere           = templeBSphere;
    _actorData.lastObject.gpuProg           = templePipelineState;
    _actorData.lastObject.meshes            = templeMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;

    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3) {1000.f, -150.f, 1000.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 2.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {0.6f, 1.f, 0.6f};
    _actorData.lastObject.bSphere           = templeBSphere;
    _actorData.lastObject.gpuProg           = templePipelineState;
    _actorData.lastObject.meshes            = templeMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;

    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3) {1150.f, -150.f, -400.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 3.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {0.45f, 0.45f, 1.f};
    _actorData.lastObject.bSphere           = templeBSphere;
    _actorData.lastObject.gpuProg           = templePipelineState;
    _actorData.lastObject.meshes            = templeMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;

    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3) {-1200.f, -150.f, -300.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 4.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {1.f, 0.6f, 0.6f};
    _actorData.lastObject.bSphere           = templeBSphere;
    _actorData.lastObject.gpuProg           = templePipelineState;
    _actorData.lastObject.meshes            = templeMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;
     */
/*
    [_actorData addObject:[AAPLActorData new]];
    _actorData.lastObject.translation       = (vector_float3) {0.f, 0.f, 0.f};
    _actorData.lastObject.rotationPoint     = SceneCenter + (vector_float3){0.f, -200.f, 0.f};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 0.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {1.f, 1.f, 1.f};
    _actorData.lastObject.bSphere           = groundBSphere;
    _actorData.lastObject.gpuProg           = groundPipelineState;
    _actorData.lastObject.meshes            = groundMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::ALL_PASS;
*/
    _reflectiveActor = [AAPLActorData new];
    [_actorData addObject:_reflectiveActor];
    _actorData.lastObject.rotationPoint     = (vector_float3) {0, -600.f, 1500.f};
    _actorData.lastObject.translation       = (vector_float3) {0,0,0};
    _actorData.lastObject.rotationAmount    = 0.f;
    _actorData.lastObject.rotationSpeed     = 6.f;
    _actorData.lastObject.rotationAxis      = (vector_float3) {0.0f, 1.f, 0.f};
    _actorData.lastObject.diffuseMultiplier = (vector_float3) {1.f, 1.f, 1.f};
    _actorData.lastObject.bSphere           = templeBSphere;
    _actorData.lastObject.gpuProg           = templePipelineState;
    _actorData.lastObject.meshes            = templeMeshes;
    _actorData.lastObject.passFlags         = EPassFlags::Final;
    
    _actorPrevModelMatrix.resize(_actorData.count);
    
    [self createTAAResource:mtkView];
}
-(void)OnMouseDown:(NSEvent *)event
{
    CGPoint mousePos = [event locationInWindow];
    _magnifier_pos_x = mousePos.x*2;
    _magnifier_pos_y = mousePos.y*2;
}
- (void)updateGameState
{
    FrustumCuller culler_final;
    FrustumCuller culler_probe [6];
    
    vector_float2 jitter = [self generateJitterOffset];
    
    matrix_float4x4 oldJitterMatrix = _jitterMatrix;
    _jitterMatrix = matrix4x4_translation(jitter.x, jitter.y,0 );

    // Update each actor's position and parameter buffer
    {
        ActorParams *actorParams  =
            (ActorParams *)_actorsParamsBuffers[_uniformBufferIndex].contents;

        for (int i = 0; i < _actorData.count; i++)
        {
            const matrix_float4x4 modelTransMatrix    = matrix4x4_translation(_actorData[i].translation);
            const matrix_float4x4 modelRotationMatrix = matrix4x4_rotation (_actorData[i].rotationAmount, _actorData[i].rotationAxis);
            const matrix_float4x4 modelPositionMatrix = matrix4x4_translation(_actorData[i].rotationPoint);

            matrix_float4x4 modelMatrix;
            modelMatrix = matrix_multiply(modelRotationMatrix, modelTransMatrix);
            modelMatrix = matrix_multiply(modelPositionMatrix, modelMatrix);

            _actorData[i].modelPosition = matrix_multiply(modelMatrix, (vector_float4) {0, 0, 0, 1});

            // we update the actor's rotation for next frame (cpu side) :
            if(_enableAnimation)
            {
                _actorData[i].rotationAmount += 0.004 * _actorData[i].rotationSpeed / 2.0f;
            }
            // we update the actor's shader parameters :
            actorParams[i].prevModelMatrix = _actorPrevModelMatrix[i];
            _actorPrevModelMatrix[i] = modelMatrix;
            actorParams[i].modelMatrix = modelMatrix;
            actorParams[i].diffuseMultiplier = _actorData[i].diffuseMultiplier;
            actorParams[i].materialShininess = 4;
        }
    }
    
    // We update the viewport (shader parameter buffer + culling utility) :
    {
        _cameraFinal.target   = SceneCenter;

        if(_enableAnimation)
        {
            _cameraFinal.rotation = fmod ((_cameraFinal.rotation + CameraRotationSpeed), M_PI*2.f);
        }
        matrix_float3x3 rotationMatrix = matrix3x3_rotation (_cameraFinal.rotation,  CameraRotationAxis);

        _cameraFinal.position = SceneCenter;
        _cameraFinal.position += matrix_multiply (rotationMatrix, CameraDistanceFromCenter);

        const matrix_float4x4 viewMatrix       = _cameraFinal.GetViewMatrix();
        const matrix_float4x4 projectionMatrix = _cameraFinal.GetProjectionMatrix_LH();

        culler_final.Reset_LH (viewMatrix, _cameraFinal);

        ViewportParams *viewportBuffer = (ViewportParams *)_viewportsParamsBuffers[_uniformBufferIndex].contents;
        viewportBuffer[0].cameraPos            = _cameraFinal.position;
        viewportBuffer[0].viewSize = _viewSize;
        viewportBuffer[0].prevViewProjMatrix =  _viewProjMatrix;
        _viewProjMatrix = matrix_multiply (projectionMatrix, viewMatrix);
        viewportBuffer[0].jitter = jitter;
        viewportBuffer[0].viewProjectionMatrix = _viewProjMatrix;
        viewportBuffer[0].invViewProjMatrix = matrix_invert(_viewProjMatrix);
    }
    // We update the shader parameters - frame constants :
    {
        const vector_float3 ambientLightColor         = {0.2, 0.2, 0.2};
        const vector_float3 directionalLightColor     = {.75, .75, .75};
        const vector_float3 directionalLightDirection = vector_normalize((vector_float3){1.0, -1.0, 1.0});

        FrameParams *frameParams =
            (FrameParams *) _frameParamsBuffers[_uniformBufferIndex].contents;
        frameParams[0].ambientLightColor            = ambientLightColor;
        frameParams[0].directionalLightInvDirection = -directionalLightDirection;
        frameParams[0].directionalLightColor        = directionalLightColor;
    }
    {
        MagnifierParams *magnifierParams =
            (MagnifierParams *) _magnifierParamsBuffers[_uniformBufferIndex].contents;
        magnifierParams[0].viewSize = _viewSize;
        magnifierParams[0].position.x = _magnifier_pos_x;
        magnifierParams[0].position.y = _magnifier_pos_y;
        magnifierParams[0].size.x = _magnifier_size_x;
        magnifierParams[0].size.y = _magnifier_size_y;
        
    }
    //  Perform culling and determine how many instances we need to draw
    {
        for (int actorIdx = 0; actorIdx < _actorData.count; actorIdx++)
        {
            if (_actorData[actorIdx].passFlags & EPassFlags::Final)
            {
                if (culler_final.Intersects (_actorData[actorIdx].modelPosition.xyz, _actorData[actorIdx].bSphere))
                {
                    _actorData[actorIdx].visibleInFinal = YES;
                }
                else
                {
                    _actorData[actorIdx].visibleInFinal = NO;
                }
            }
        }
    }
}

- (void)drawActors:(id<MTLRenderCommandEncoder>) renderEncoder
              pass:(EPassFlags)pass
{
    id<MTLBuffer> viewportBuffer;

    viewportBuffer        = _viewportsParamsBuffers [_uniformBufferIndex];

    // Adds contextual info into the GPU Frame Capture tool
    [renderEncoder pushDebugGroup:[NSString stringWithFormat:@"DrawActors %d", pass]];

    [renderEncoder setCullMode:MTLCullModeBack];
    [renderEncoder setDepthStencilState:_depthState];

    // Set any buffers fed into our render pipeline

    [renderEncoder setFragmentBuffer: _frameParamsBuffers[_uniformBufferIndex]
                              offset: 0
                             atIndex: BufferIndexFrameParams];

    [renderEncoder setVertexBuffer: viewportBuffer
                            offset: 0
                           atIndex: BufferIndexViewportParams];

    [renderEncoder setFragmentBuffer: viewportBuffer
                              offset: 0
                             atIndex: BufferIndexViewportParams];

    for (int actorIdx = 0; actorIdx < _actorData.count; actorIdx++)
    {
        AAPLActorData* lActor = _actorData[actorIdx];

        if ((lActor.passFlags & pass) == 0) continue;

        uint32_t visibleVpCount;

        visibleVpCount = lActor.visibleInFinal;

        if (visibleVpCount == 0) continue;

        // per-actor parameters
        [renderEncoder setVertexBuffer: _actorsParamsBuffers[_uniformBufferIndex]
                                offset: actorIdx * Align<BufferOffsetAlign> (sizeof(ActorParams))
                               atIndex: BufferIndexActorParams];

        [renderEncoder setFragmentBuffer: _actorsParamsBuffers[_uniformBufferIndex]
                                  offset: actorIdx * Align<BufferOffsetAlign> (sizeof(ActorParams))
                                 atIndex: BufferIndexActorParams];

        [renderEncoder setRenderPipelineState:lActor.gpuProg];

        for (AAPLMesh *mesh in lActor.meshes)
        {
            MTKMesh *metalKitMesh = mesh.metalKitMesh;

            // Set mesh's vertex buffers
            for (NSUInteger bufferIndex = 0; bufferIndex < metalKitMesh.vertexBuffers.count; bufferIndex++)
            {
                MTKMeshBuffer *vertexBuffer = metalKitMesh.vertexBuffers[bufferIndex];
                if((NSNull*)vertexBuffer != [NSNull null])
                {
                    [renderEncoder setVertexBuffer: vertexBuffer.buffer
                                            offset: vertexBuffer.offset
                                           atIndex: bufferIndex];
                }
            }

            // Draw each submesh of our mesh
            for(AAPLSubmesh *submesh in mesh.submeshes)
            {
                // Set any textures read/sampled from our render pipeline
                id<MTLTexture> tex;

                tex = submesh.textures [TextureIndexBaseColor];
                if ((NSNull*)tex != [NSNull null])
                {
                    [renderEncoder setFragmentTexture:tex atIndex:TextureIndexBaseColor];
                }

                tex = submesh.textures [TextureIndexNormal];
                if ((NSNull*)tex != [NSNull null])
                {
                    [renderEncoder setFragmentTexture:tex atIndex:TextureIndexNormal];
                }

                tex = submesh.textures[TextureIndexSpecular];
                if ((NSNull*)tex != [NSNull null])
                {
                    [renderEncoder setFragmentTexture:tex atIndex:TextureIndexSpecular];
                }

                MTKSubmesh *metalKitSubmesh = submesh.metalKitSubmmesh;

                [renderEncoder drawIndexedPrimitives: metalKitSubmesh.primitiveType
                                          indexCount: metalKitSubmesh.indexCount
                                           indexType: metalKitSubmesh.indexType
                                         indexBuffer: metalKitSubmesh.indexBuffer.buffer
                                   indexBufferOffset: metalKitSubmesh.indexBuffer.offset
                                       instanceCount: visibleVpCount
                                          baseVertex: 0
                                        baseInstance: actorIdx * MaxVisibleFaces];
            }
        }
    }

    [renderEncoder popDebugGroup];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    // We set the overall vertical field of view to 65 degrees (converted to radians).
    // We store it divided by two.
    static const float fovY_Half = radians_from_degrees(65.0 *.5);
    const float        aspect    = size.width / (float)size.height;

    _cameraFinal.aspectRatio  = aspect;
    _cameraFinal.fovVert_Half = fovY_Half;
    _cameraFinal.distanceNear = 50.f;
    _cameraFinal.distanceFar  = 5000.f;
    
    _viewSize.x = size.width;
    _viewSize.y = size.height;
}

- (void)drawInMTKView:(nonnull MTKView *)view
{
    if(_stepByStep)
    {
        if(_advanceToNextFrame == false)
        {
            return;
        }
        _advanceToNextFrame = false;
    }
    //------------------------------------------------------------------------------------
    // Update game state and shader parameters

    // Wait to ensure only MaxBuffersInFlight are getting processed by any stage in the Metal
    //   pipeline (App, Metal, Drivers, GPU, etc)
    dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);

    // Update the location(s) to which we'll write to in our dynamically changing Metal buffers for
    //   the current frame (i.e. update our slot in the ring buffer used for the current frame)
    _uniformBufferIndex = (_uniformBufferIndex + 1) % MaxBuffersInFlight;

    [self updateGameState];

    //------------------------------------------------------------------------------------
    // Render

    // Create a new command buffer for each render pass to the current drawable
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"Drawable Rendering";
    
    // Render to frame buffer
    {
        MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
        passDesc.colorAttachments[0].clearColor = MTLClearColorMake (1.0, 1.0, 1.0, 1.0);
        passDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
        passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
        passDesc.colorAttachments[1].clearColor = MTLClearColorMake (0.0, 0.0, 0.0, 1.0);
        passDesc.colorAttachments[1].loadAction = MTLLoadActionClear;
        passDesc.colorAttachments[1].storeAction = MTLStoreActionStore;
        passDesc.depthAttachment.clearDepth     = 1.0;
        passDesc.depthAttachment.loadAction     = MTLLoadActionClear;
        passDesc.depthAttachment.storeAction = MTLStoreActionStore;
        
        passDesc.colorAttachments[0].texture    = _framebuffer;
        passDesc.colorAttachments[1].texture = _velocitybuffer;
        passDesc.depthAttachment.texture        = _framedepthbuffer;
        passDesc.renderTargetArrayLength        = 1;
        
        id<MTLRenderCommandEncoder> renderEncoder =
            [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
        renderEncoder.label = @"Frame Render";

        [self drawActors: renderEncoder pass: EPassFlags::Final];

        [renderEncoder endEncoding];
    }
    
    if(_enableTAA)
    {
        // Resolve
        {
            MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
            passDesc.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            passDesc.colorAttachments[0].texture    = _historybuffers[1];
            passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
            
            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
            renderEncoder.label = @"Resolve";

            [renderEncoder setFragmentTexture:_framebuffer atIndex:0];
            [renderEncoder setFragmentTexture:_historyReset ? _framebuffer : _historybuffers[0] atIndex:1];
            [renderEncoder setFragmentTexture:_velocitybuffer atIndex:2];
            [renderEncoder setRenderPipelineState:_pipeline_TAA_Resolve];
            
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];

            [renderEncoder endEncoding];
            
            _historyReset = false;
        }
        {
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            
            [blitEncoder copyFromTexture:_historybuffers[1] toTexture:_historybuffers[0]];
            [blitEncoder endEncoding];
        }
    }
    
    // Add a completed handler which signals _inFlightSemaphore when Metal and the GPU has fully
    //   finished processing the commands encoded this frame.  This indicates when the
    //   dynamic buffers written to this frame will no longer be needed by Metal and the GPU.
    __block dispatch_semaphore_t block_sema = _inFlightSemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer)
     {
         dispatch_semaphore_signal(block_sema);
     }];
    
    if(_enableTAA)
    {
        // Blit to Drawable
        {
            MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
            passDesc.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

            passDesc.colorAttachments[0].texture    = view.currentRenderPassDescriptor.colorAttachments[0].texture;

            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
            renderEncoder.label = @"Blit to Drawable";

            [renderEncoder setFragmentTexture:_historybuffers[0] atIndex:0];
            [renderEncoder setRenderPipelineState:_pipeline_blit];
            
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];

            [renderEncoder endEncoding];
        }
    }
    else
    {
        // Blit to Drawable
        {
            MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
            passDesc.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

            passDesc.colorAttachments[0].texture    = view.currentRenderPassDescriptor.colorAttachments[0].texture;

            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
            renderEncoder.label = @"Blit to Drawable";

            [renderEncoder setFragmentTexture:_framebuffer atIndex:0];
            [renderEncoder setRenderPipelineState:_pipeline_blit];
            
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];

            [renderEncoder endEncoding];
        }

    }
    
    // Render the magnifier
    if(_enableMagnifier)
    {
        {
            MTLRenderPassDescriptor* passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
            passDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
            passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

            passDesc.colorAttachments[0].texture    = view.currentRenderPassDescriptor.colorAttachments[0].texture;

            id<MTLRenderCommandEncoder> renderEncoder =
                [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
            renderEncoder.label = @"Render Magnifier";

            [renderEncoder setVertexBuffer:_magnifierParamsBuffers[_uniformBufferIndex] offset:0 atIndex:0];
            [renderEncoder setFragmentTexture:_enableTAA ? _historybuffers[0] : _framebuffer atIndex:0];
            [renderEncoder setRenderPipelineState:_pipeline_magnifier];
            
            [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];

            [renderEncoder endEncoding];
        }
    }
    if(view.currentDrawable)
    {
        // Schedule a present once the framebuffer is complete using the current drawable
        [commandBuffer presentDrawable:view.currentDrawable];
    }

    // Finalize rendering here & push the command buffer to the GPU
    [commandBuffer commit];
}

@end
