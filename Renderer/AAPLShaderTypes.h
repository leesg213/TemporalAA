/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Header containing types and enum constants shared between Metal shaders and C/ObjC source
*/

#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

// [MTLRenderCommandEncoder setVertexBuffer:offset:atIndex] requires that if you the offset
//   be 256 byte aligned.  Thus we'll use this constant for any buffer or structure which we use
//   the 'offset' parameter to index into
#define BufferOffsetAlign 256

// Buffer index values shared between shader and C code to ensure Metal shader buffer inputs match
//   Metal API buffer set calls
typedef enum BufferIndex
{
    BufferIndexMeshPositions,
    BufferIndexMeshGenerics,
    BufferIndexFrameParams,
    BufferIndexViewportParams,
    BufferIndexActorParams,
    BufferIndexInstanceParams
} AAPLBufferIndex;

// Attribute index values shared between shader and C code to ensure Metal shader vertex
//   attribute indices match the Metal API vertex descriptor attribute indices
typedef enum VertexAttribute
{
    VertexAttributePosition  = 0,
    VertexAttributeTexcoord  = 1,
    VertexAttributeNormal    = 2,
    VertexAttributeTangent   = 3,
    VertexAttributeBitangent = 4
} AAPLVertexAttribute;

// Texture index values shared between shader and C code to ensure Metal shader texture indices
//   match indices of Metal API texture set calls
typedef enum TextureIndex
{
    TextureIndexBaseColor = 0,
    TextureIndexSpecular  = 1,
    TextureIndexNormal    = 2,
    TextureIndexCubeMap   = 3
} AAPLTextureIndex;

// Structure shared between shader and C code to ensure the layout of uniform data accessed in
//    Metal shaders matches the layout of uniform data set in C code
typedef struct
{
    // Per Light Properties
    vector_float3 ambientLightColor;
    vector_float3 directionalLightInvDirection;
    vector_float3 directionalLightColor;
} FrameParams;

typedef struct
{
    vector_float3 cameraPos;
    vector_float2 viewSize;
    vector_float2 jitter;
    matrix_float4x4 viewProjectionMatrix;
    matrix_float4x4 invViewProjMatrix;
    matrix_float4x4 prevViewProjMatrix;
} ViewportParams;

typedef struct
{
    vector_float2 viewSize;
    vector_float2 position;
    vector_float2 size;
} MagnifierParams;

// Structure shared between shader and C code to ensure the layout of uniform data accessed in
//    Metal shaders matches the layout of uniform data set in C code
typedef struct __attribute__((aligned(BufferOffsetAlign)))
{
    // Per Mesh Uniforms
    matrix_float4x4 modelMatrix;
    matrix_float4x4 prevModelMatrix;
    vector_float3 diffuseMultiplier;
    float materialShininess;
    uint viewportIndex;
} ActorParams;

#endif /* ShaderTypes_h */

