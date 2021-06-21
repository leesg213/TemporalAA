/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Utilities for renderer class
*/

#import "AAPLMathUtilities.h"

//----------------------------------------------------------------------------------------

struct Camera
{
    vector_float3 position;
    vector_float3 target;
    float rotation;
    float aspectRatio;      // width/height
    float fovVert_Half;     // half of vertical field of view, in radians
    float distanceNear;
    float distanceFar;

    matrix_float4x4 GetViewMatrix () const
    {
        return matrix_look_at_left_hand(position, target, (vector_float3){0,1,0});
    }

    matrix_float4x4 GetProjectionMatrix_LH () const
    {
        return matrix_perspective_left_hand (
            fovVert_Half * 2.f,
            aspectRatio,
            distanceNear,
            distanceFar);
    }
};

//----------------------------------------------------------------------------------------

struct CameraProbe
{
    vector_float3 position;
    float distanceNear;
    float distanceFar;

    // Fills in the view matrices, for the following axis : +X -X +Y -Y +Z -Z
    matrix_float4x4 GetViewMatrixForFace_LH (int faceIdx) const
    {
        static const vector_float3 directions [6] =
        {
            { 1,  0,  0}, // Right
            {-1,  0,  0}, // Left
            { 0,  1,  0}, // Top
            { 0, -1,  0}, // Down
            { 0,  0,  1}, // Front
            { 0,  0, -1}  // Back
        };

        static const vector_float3 ups [6] =
        {
            {0, 1,  0},
            {0, 1,  0},
            {0, 0, -1},
            {0, 0,  1},
            {0, 1,  0},
            {0, 1,  0}
        };

        return matrix_look_at_left_hand(position, position + directions[faceIdx], ups[faceIdx]);
    }

    matrix_float4x4 GetProjectionMatrix_LH () const
    {
        return matrix_perspective_left_hand (
            M_PI_2,
            1.f,
            distanceNear,
            distanceFar);
    }
};

//----------------------------------------------------------------------------------------

// Utility structure to test intersection between a frustum and parametric shapes
struct FrustumCuller
{
    // frustum origin location
    vector_float3 position;

    // planes normals :
    vector_float3 norm_NearPlane;
    vector_float3 norm_LeftPlane;
    vector_float3 norm_RightPlane;
    vector_float3 norm_BottomPlane;
    vector_float3 norm_TopPlane;

    // near / far distances from the frustum's origin
    float         dist_Near;
    float         dist_Far;

    // Initializes data so we can call Intersection predicates.
    // Made for a left-handed coordinate system.
    void Reset_LH ( const matrix_float4x4 viewMatrix,
                    const vector_float3   viewPosition,
                    const float           aspect,
                    const float           halfAngleApertureHeight, // in radians
                    const float           nearPlaneDistance,
                    const float           farPlaneDistance )
    {
        position  = viewPosition;
        dist_Near = nearPlaneDistance;
        dist_Far  = farPlaneDistance;

        const float halfAngleApertureWidth = halfAngleApertureHeight * aspect;
        const matrix_float3x3 cameraRotationMatrix = matrix_invert (matrix3x3_upper_left (viewMatrix));

        norm_NearPlane = matrix_multiply (
            cameraRotationMatrix,
            (vector_float3) {0.0, 0.0, 1.0} );

        norm_LeftPlane = matrix_multiply (
            cameraRotationMatrix,
            (vector_float3) {cosf(halfAngleApertureWidth), 0.f, sinf(halfAngleApertureWidth)} );

        norm_BottomPlane = matrix_multiply (
            cameraRotationMatrix,
            (vector_float3) {0.f, cosf(halfAngleApertureHeight), sinf(halfAngleApertureHeight)} );

        // we reflect the left plane normal along the view direction (norm_NearPlane) to get the right plane normal :
        norm_RightPlane  = (- norm_LeftPlane)   + norm_NearPlane * (vector_dot(norm_NearPlane, norm_LeftPlane)   * 2.f);
        // we do the same, to get the top plane normal, from the bottom plane :
        norm_TopPlane    = (- norm_BottomPlane) + norm_NearPlane * (vector_dot(norm_NearPlane, norm_BottomPlane) * 2.f);
    }

    // `cachedViewMatrix` must be the view matrix retrieved from `camera`.
    // It is given as an argument (instead of directly getting it from the camera) since
    // the caller is requiring it anyway, and probably already asked for it.
    void Reset_LH (const matrix_float4x4 cachedViewMatrix, const Camera camera)
    {
        Reset_LH (cachedViewMatrix, camera.position, camera.aspectRatio, camera.fovVert_Half, camera.distanceNear, camera.distanceFar);
    }

    void Reset_LH (const matrix_float4x4 cachedViewMatrix, const CameraProbe camera)
    {
        Reset_LH (cachedViewMatrix, camera.position, 1.f, M_PI_4, camera.distanceNear, camera.distanceFar);
    }

    // To test the intersection between a frustum and a sphere, we "inflate" the frustum by
    // the bounding sphere radius ; then test if the sphere center is inside this extended frustum.
    bool Intersects (const vector_float3 actorPosition, vector_float4 bSphere) const
    {
        const vector_float4 position_f4 = (vector_float4) {actorPosition.x, actorPosition.y, actorPosition.z, 0.f};
        bSphere += position_f4;

        const float         bSphereRadius    = bSphere.w;
        const vector_float3 camToSphere      = bSphere.xyz - position;

        if (vector_dot (camToSphere + norm_NearPlane * (bSphereRadius-dist_Near), norm_NearPlane)   < 0) { return false; }
        if (vector_dot (camToSphere - norm_NearPlane * (bSphereRadius+dist_Far),  -norm_NearPlane)  < 0) { return false; }

        if (vector_dot (camToSphere + norm_LeftPlane  * bSphereRadius,            norm_LeftPlane)   < 0) { return false; }
        if (vector_dot (camToSphere + norm_RightPlane * bSphereRadius,            norm_RightPlane)  < 0) { return false; }

        if (vector_dot (camToSphere + norm_BottomPlane * bSphereRadius,           norm_BottomPlane) < 0) { return false; }
        if (vector_dot (camToSphere + norm_TopPlane    * bSphereRadius,           norm_TopPlane)    < 0) { return false; }

        return true;
    }
};

//----------------------------------------------------------------------------------------

// List of all passes the renderer will go through. They are defined as a bit field,
// to allow actors selectively 'subscribe' to 0-n of them.
enum EPassFlags : uint8_t
{
    Reflection = 1 << 0,
    Final      = 1 << 1,
    // ...
    ALL_PASS = (uint8_t) ~(uint8_t(0))
};

//----------------------------------------------------------------------------------------

// Data describing each 'object' the world will contain.
@interface AAPLActorData : NSObject

    // Metal pipeline used to render this actor
    @property (nonatomic) id<MTLRenderPipelineState>  gpuProg;

    // pointer to meshes used by this actor
    @property (nonatomic, copy)  NSArray <AAPLMesh*>*  meshes;

    // bounding sphere. position is stored in xyz, radius is stored in w.
    @property (nonatomic)  vector_float4               bSphere;

    // multiplier used in shading to color actors using the same mesh differently
    @property (nonatomic)  vector_float3               diffuseMultiplier;

    // translation away from rotation point
    @property (nonatomic)  vector_float3               translation;

    // Position around which we rotate the object
    @property (nonatomic)  vector_float3               rotationPoint;

    // current rotation angle (in radians) around rotationAxis at rotationPoint
    @property (nonatomic)  float                       rotationAmount;

    // per-actor multiplier for rotation
    @property (nonatomic)  float                       rotationSpeed;

    // per-actor axis for rotation
    @property (nonatomic)  vector_float3               rotationAxis;

    // actor's position in the scene
    @property (nonatomic)  vector_float4               modelPosition;

    // passes this actor must be rendered to
    @property (nonatomic)  EPassFlags                  passFlags;

    // number of instances with which we must draw this actor in the reflection pass
    @property (nonatomic)  uint8_t                     instanceCountInReflection;

    // Whether this actor is visible in the final pass
    @property (nonatomic)  BOOL                        visibleInFinal;
@end
@implementation AAPLActorData
@end

//----------------------------------------------------------------------------------------

// Utility function to align a memory address
template <size_t align>
constexpr size_t Align (size_t value)
{
    static_assert (
        align == 0 || (align & (align-1)) == 0,
        "align must 0 or a power of two" );

    if (align == 0)
    {
        return value;
    }
    else if ((value & (align-1)) == 0)
    {
        return value;
    }
    else
    {
        return (value+align) & ~(align-1);
    }
}
