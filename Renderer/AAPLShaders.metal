/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Metal shaders used for this sample.
*/

#include <metal_stdlib>

using namespace metal;

#include "AAPLShaderTypes.h"

struct Vertex
{
    float3 position  [[attribute(VertexAttributePosition)]];
    float2 texCoord  [[attribute(VertexAttributeTexcoord)]];
    half3  normal    [[attribute(VertexAttributeNormal)]];
    half3  tangent   [[attribute(VertexAttributeTangent)]];
    half3  bitangent [[attribute(VertexAttributeBitangent)]];
};

struct ColorInOut
{
    float4 position [[position]];
    float4 currentFramePosition;
    float4 prevFramePosition;
    float2 texCoord;

    half3  worldPos;
    half3  tangent;
    half3  bitangent;
    half3  normal;
};

struct FragmentOut
{
    float4 color [[color(0)]];
    float4 velocity [[color(1)]];
};

// Vertex function
vertex ColorInOut vertexTransform (const Vertex in                               [[ stage_in ]],
                                   const uint   instanceId                       [[ instance_id ]],
                                   const device ActorParams&    actorParams      [[ buffer (BufferIndexActorParams)    ]],
                                   constant     ViewportParams& viewportParams   [[ buffer (BufferIndexViewportParams) ]] )
{
    ColorInOut out;
    out.texCoord = in.texCoord;

    float4x4 currentFrame_modelMatrix = actorParams.modelMatrix;
    float4 currentFrame_worldPos  = currentFrame_modelMatrix * float4(in.position, 1.0);
    float4 currentFrame_clipPos = viewportParams.viewProjectionMatrix * currentFrame_worldPos;
    
    out.currentFramePosition = currentFrame_clipPos;
    
    float4 currentFrame_clipPos_jittered =
        currentFrame_clipPos + float4(viewportParams.jitter*currentFrame_clipPos.w,0,0);
    
    out.position = currentFrame_clipPos_jittered;
    
    float4x4 prevFrame_modelMatrix = actorParams.prevModelMatrix;
    float4 prevFrame_worldPos  = prevFrame_modelMatrix * float4(in.position, 1.0);
    float4 prevFrame_clipPos = viewportParams.prevViewProjMatrix * prevFrame_worldPos;
    
    out.prevFramePosition = prevFrame_clipPos;

    half3x3 normalMatrix = half3x3((half3)currentFrame_modelMatrix[0].xyz,
                                   (half3)currentFrame_modelMatrix[1].xyz,
                                   (half3)currentFrame_modelMatrix[2].xyz);

    out.tangent   = normalMatrix * in.tangent;
    out.bitangent = normalMatrix * in.bitangent;
    out.normal    = normalMatrix * in.normal;

    return out;
}

float2 CalcVelocity(float4 newPos, float4 oldPos, float2 viewSize)
{
    oldPos /= oldPos.w;
    oldPos.xy = (oldPos.xy+1)/2.0f;
    oldPos.y = 1 - oldPos.y;
    
    newPos /= newPos.w;
    newPos.xy = (newPos.xy+1)/2.0f;
    newPos.y = 1 - newPos.y;
    
    return (newPos - oldPos).xy;
}
// Fragment function used to render the temple object in both the
//   reflection pass and the final pass
fragment FragmentOut fragmentLighting (         ColorInOut      in             [[ stage_in ]],
                                  device   ActorParams&    actorParams    [[ buffer (BufferIndexActorParams)    ]],
                                  constant FrameParams &   frameParams    [[ buffer (BufferIndexFrameParams)    ]],
                                  constant ViewportParams& viewportParams [[ buffer (BufferIndexViewportParams) ]],
                                           texture2d<half> baseColorMap   [[ texture (TextureIndexBaseColor)    ]],
                                           texture2d<half> normalMap      [[ texture (TextureIndexNormal)       ]],
                                           texture2d<half> specularMap    [[ texture (TextureIndexSpecular)     ]] )
{
    constexpr sampler linearSampler (mip_filter::linear,
                                     mag_filter::linear,
                                     min_filter::linear);
    FragmentOut out;

    const half4 baseColorSample = baseColorMap.sample (linearSampler, in.texCoord.xy);
    half3 normalSampleRaw = normalMap.sample (linearSampler, in.texCoord.xy).xyz;
    // The x and y coordinates in a normal map (red and green channels) are mapped from [-1;1] to [0;255].
    // As the sampler returns a value in [0 ; 1], we need to do :
    normalSampleRaw.xy = normalSampleRaw.xy * 2.0 - 1.0;
    const half3 normalSample = normalize(normalSampleRaw);

    const half  specularSample  = specularMap.sample  (linearSampler, in.texCoord.xy).x*0.5;

    // The per-vertex vectors have been interpolated, thus we need to normalize them again :
    in.tangent   = normalize (in.tangent);
    in.bitangent = normalize (in.bitangent);
    in.normal    = normalize (in.normal);

    half3x3 tangentMatrix = half3x3(in.tangent, in.bitangent, in.normal);

    float3 normal = (float3) (tangentMatrix * normalSample);

    float3 directionalContribution = float3(0);
    float3 specularTerm = float3(0);
    {
        float nDotL = saturate (dot(normal, frameParams.directionalLightInvDirection));

        // The diffuse term is the product of the light color, the surface material
        // reflectance, and the falloff
        float3 diffuseTerm = frameParams.directionalLightColor * nDotL;

        // Apply specular lighting...

        // 1) Calculate the halfway vector between the light direction and the direction they eye is looking
        float3 eyeDir = normalize (viewportParams.cameraPos - float3(in.worldPos));
        float3 halfwayVector = normalize(frameParams.directionalLightInvDirection + eyeDir);

        // 2) Calculate the reflection amount by evaluating how the halfway vector matches the surface normal
        float reflectionAmount = saturate(dot(normal, halfwayVector));

        // 3) Calculate the specular intensity by powering our reflection amount to our object's
        //    shininess
        float specularIntensity = powr(reflectionAmount*1.025, actorParams.materialShininess*4);

        // 4) Obtain the specular term by multiplying the intensity by our light's color
        specularTerm = frameParams.directionalLightColor * specularIntensity * float(specularSample);

        // The base color sample is actually the diffuse color of the material
        float3 baseColor = float3(baseColorSample.xyz) * actorParams.diffuseMultiplier;

        // The ambient contribution is an approximation for global, indirect lighting, and simply added
        //   to the calculated lit color value below

        // Calculate diffuse contribution from this light : the sum of the diffuse and ambient * albedo
        directionalContribution = baseColor * (diffuseTerm + frameParams.ambientLightColor);
    }

    // Now that we have the contributions our light sources in the scene, we sum them together
    //   to get the fragment's lit color value
    float3 color = specularTerm + directionalContribution;

    // We return the color we just computed and the alpha channel of our baseColorMap for this
    //   fragment's alpha value
    
    out.color = float4(color, baseColorSample.w);
    out.velocity = float4(CalcVelocity(in.currentFramePosition, in.prevFramePosition, viewportParams.viewSize),0,0);
    
    return out;
}

fragment FragmentOut fragmentGround (         ColorInOut      in             [[ stage_in ]],
                                constant ViewportParams& viewportParams [[ buffer (BufferIndexViewportParams) ]] )
{
    float onEdge;
    {
        float2 onEdge2d = fract(float2(in.worldPos.xz)/500.f);
        // If onEdge2d is negative, we want 1. Otherwise, we want zero (independent for each axis).
        float2 offset2d = sign(onEdge2d) * -0.5 + 0.5;
        onEdge2d += offset2d;
        onEdge2d = step (0.03, onEdge2d);

        onEdge = min(onEdge2d.x, onEdge2d.y);
    }

    float3 neutralColor = float3 (0.9, 0.9, 0.9);
    float3 edgeColor = neutralColor * 0.2;
    float3 groundColor = mix (edgeColor, neutralColor, 1);

    FragmentOut out;
    
    out.color = float4(groundColor, 1);

    out.velocity = float4(CalcVelocity(in.currentFramePosition, in.prevFramePosition, viewportParams.viewSize),0,0);
    
    return out;
}


// Screen filling quad in normalized device coordinates
constant float2 quadVertices[] = {
    float2(-1, -1),
    float2(-1,  1),
    float2( 1,  1),
    float2(-1, -1),
    float2( 1,  1),
    float2( 1, -1)
};

struct quadVertexOut {
    float4 position [[position]];
    float2 uv;
};

// Simple vertex shader which passes through NDC quad positions
vertex quadVertexOut fullscreenQuadVertex(unsigned short vid [[vertex_id]]) {
    float2 position = quadVertices[vid];
    
    quadVertexOut out;
    
    out.position = float4(position, 0, 1);
    out.uv = position * 0.5f + 0.5f;
    out.uv.y = 1 - out.uv.y;
    return out;
}

float2 FindClosestDepthSamplePos(float2 uv, texture2d<float> depthbuffer, float2 viewSize)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float2 closest = uv;
    float closest_depth = 1;
    
    float depth = 0;
    
    for(int y = -1;y<=1;++y)
    {
        for(int x=-1;x<=1;++x)
        {
            float2 uv_offset = float2(x,y) / viewSize;
            depth = depthbuffer.sample(sam, uv + uv_offset).x;
            if(depth < closest_depth)
            {
                closest_depth = depth;
                closest = uv + uv_offset;
            }
        }
    }
    
    return closest;
}
// Simple fragment shader which copies a texture and applies a simple tonemapping function
fragment float4 TAA_ResolveFragment(quadVertexOut in [[stage_in]],
                             texture2d<float> currentFrameColorBuffer [[texture(0)]],
                             texture2d<float> historyBuffer [[texture(1)]],
                            texture2d<float> velocityBuffer [[texture(2)]])
{
    constexpr sampler sam_point(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    constexpr sampler sam_linear(min_filter::linear, mag_filter::linear, mip_filter::none);

    float2 velocity_sample_pos = in.uv;
    float2 velocity = velocityBuffer.sample(sam_point, velocity_sample_pos).xy;
    float2 prevousPixelPos = in.uv - velocity;
    
    float3 currentColor = currentFrameColorBuffer.sample(sam_point, in.uv).xyz;
    float3 historyColor = historyBuffer.sample(sam_linear, prevousPixelPos).xyz;

    // Apply clamping on the history color.
    float3 NearColor0 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(1, 0)).xyz;
    float3 NearColor1 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(0, 1)).xyz;
    float3 NearColor2 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(-1, 0)).xyz;
    float3 NearColor3 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(0, -1)).xyz;
    
    float3 BoxMin = min(currentColor, min(NearColor0, min(NearColor1, min(NearColor2, NearColor3))));
    float3 BoxMax = max(currentColor, max(NearColor0, max(NearColor1, max(NearColor2, NearColor3))));;
    
    historyColor = clamp(historyColor, BoxMin, BoxMax);
    
    float modulationFactor = 0.9;
    
    float3 color = mix(currentColor, historyColor, modulationFactor);

    return float4(color, 1.0f);
}

fragment float4 BlitFragment(quadVertexOut in [[stage_in]],
                             texture2d<float> tex)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float3 color = tex.sample(sam, in.uv).xyz;
    
    return float4(color, 1);
}

vertex quadVertexOut magnifierVertex(unsigned short vid [[vertex_id]],
                                     constant MagnifierParams& magnifierParam [[buffer(0)]]) {
    float2 position = quadVertices[vid];
    float2 uv = position;
    
    quadVertexOut out;
    
    float2 scale = magnifierParam.size / (magnifierParam.viewSize);
    
    float2 windowPos = magnifierParam.viewSize - magnifierParam.size/2;
    float2 translate = (windowPos / (magnifierParam.viewSize))*2 + float2(-1,-1);
    
    position.xy *= scale;
    position.xy += translate;
    
    uv *= scale;
    float2 uv_translate = (magnifierParam.position / magnifierParam.viewSize)*2 + float2(-1,-1);
    uv += uv_translate;
    uv = uv*0.5f + 0.5f;
    float magnifier_scale = 6;
    float2 tex_center = uv_translate * 0.5f + 0.5f;
    uv = uv - tex_center;
    uv /= magnifier_scale;
    uv = uv + tex_center;
    uv.y = 1 - uv.y;
    out.uv = uv;
    out.position = float4(position, 0, 1);
    return out;
}

fragment float4 magnifierFragment(quadVertexOut in [[stage_in]],
                             texture2d<float> tex)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float3 color = tex.sample(sam, in.uv).xyz;
    
    return float4(color, 1);
}

