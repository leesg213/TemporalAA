/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation for Mesh and Submesh objects
*/

#import <ModelIO/ModelIO.h>
#import <MetalKit/MetalKit.h>

#import "AAPLMesh.h"
#import "AAPLShaderTypes.h"

@implementation AAPLSubmesh
{
    NSArray<id<MTLTexture>>* _textures;
}

@synthesize textures = _textures;

+ (nonnull id<MTLTexture>)createMetalTextureFromMaterial:(nonnull MDLMaterial *) material
                                 modelIOMaterialSemantic:(MDLMaterialSemantic ) materialSemantic
                                   metalKitTextureLoader:(nonnull MTKTextureLoader *) textureLoader
{
    id<MTLTexture> texture;

    NSArray<MDLMaterialProperty *> *propertiesWithSemantic =
        [material propertiesWithSemantic:materialSemantic];

    for (MDLMaterialProperty *property in propertiesWithSemantic)
    {
        assert(property.semantic == materialSemantic);

        NSURL * textureURL = nil;

        if(property.type == MDLMaterialPropertyTypeURL)
        {
            textureURL = property.URLValue;
        }
        else if(property.type == MDLMaterialPropertyTypeString)
        {
            // First, we'll interpret the string as a file path and attempt to load it with
            //    -[MTKTextureLoader newTextureWithContentsOfURL:options:error:]
            NSMutableString *URLString = [[NSMutableString alloc] initWithString:@"file://"];
            [URLString appendString:property.stringValue];
            textureURL = [NSURL URLWithString:URLString];
        }
        else
        {
            continue;
        }

        // Load our textures with TextureUsageShaderRead and StorageModePrivate
        NSDictionary *textureLoaderOptions =
        @{
          MTKTextureLoaderOptionTextureUsage       : @(MTLTextureUsageShaderRead),
          MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModePrivate)
          };

        // Attempt to load the texture from the file system
        texture = [textureLoader newTextureWithContentsOfURL:textureURL
                                                     options:textureLoaderOptions
                                                       error:nil];

        // If we found a texture using the string as a file path name...
        if(texture)
        {
            // ...return it
            return texture;
        }

        // If we did not find a texture by interpreting the URL as a path, we'll interpret
        //   the last component of the URL as an asset catalog name and attempt to load it
        //   with -[MTKTextureLoader newTextureWithName:scaleFactor:bundle:options::error:]

        NSString *lastComponent =
            [[textureURL.relativeString componentsSeparatedByString:@"/"] lastObject];

        texture = [textureLoader newTextureWithName:lastComponent
                                        scaleFactor:1.0
                                             bundle:nil
                                            options:textureLoaderOptions
                                              error:nil];

        // If we found a texture with the string in our asset catalog...
        if(texture)
        {
            // ...return it
            return texture;
        }

        // If we did not find the texture by interpreting the string as a file path or as an
        //   asset name in our asset catalog, something went wrong (Perhaps the file was missing
        //   or  misnamed in the asset catalog, model/material file, or file system)

        // Depending on how the Metal render pipeline used with this submesh is implemented,
        //   this condition could be handled more gracefully.  The app could load a dummy texture
        //   that will look okay when set with the pipeline or ensure that the pipeline rendering
        //   this submesh does not require a material with this property.

        [NSException raise:@"Texture data for material property not found"
                    format:@"Requested material property semantic: %lu string: %@",
         (unsigned long)materialSemantic, property.stringValue];
    }

    [NSException raise:@"No appropriate material property from which to create texture"
                format:@"Requested material property semantic: %lu", (unsigned long)materialSemantic];

    return nil;
}

- (nonnull instancetype)initWithModelIOSubmesh:(nonnull MDLSubmesh *)modelIOSubmesh
                               metalKitSubmesh:(nonnull MTKSubmesh *)metalKitSubmesh
                         metalKitTextureLoader:(nonnull MTKTextureLoader *)textureLoader
{
    self = [super init];
    if(self)
    {
        _metalKitSubmmesh = metalKitSubmesh;

        NSMutableArray <id<MTLTexture>>* mutableTextures =
            [[NSMutableArray alloc] initWithCapacity:3];

        if (textureLoader != NULL)
        {
            // Set each index in our array with the appropriate material semantic specified in the
            //   submesh's material property

            [mutableTextures addObject:
                [AAPLSubmesh createMetalTextureFromMaterial: modelIOSubmesh.material
                                    modelIOMaterialSemantic: MDLMaterialSemanticBaseColor
                                      metalKitTextureLoader: textureLoader ]];

            [mutableTextures addObject:
                [AAPLSubmesh createMetalTextureFromMaterial: modelIOSubmesh.material
                                    modelIOMaterialSemantic: MDLMaterialSemanticSpecular
                                      metalKitTextureLoader: textureLoader ]];

            [mutableTextures addObject:
                [AAPLSubmesh createMetalTextureFromMaterial: modelIOSubmesh.material
                                    modelIOMaterialSemantic: MDLMaterialSemanticTangentSpaceNormal
                                      metalKitTextureLoader: textureLoader ]];
        }
        else
        {
            [mutableTextures addObject:(id<MTLTexture>)[NSNull null]];
            [mutableTextures addObject:(id<MTLTexture>)[NSNull null]];
            [mutableTextures addObject:(id<MTLTexture>)[NSNull null]];
        }

        // As we'll access the `_textures` array by predefined indices, we must ensure
        //   they match the order in which we created them.
        static_assert (TextureIndexBaseColor == 0, "");
        static_assert (TextureIndexSpecular == 1, "");
        static_assert (TextureIndexNormal == 2, "");

        _textures = mutableTextures;
    }
    return self;
}

@end

@implementation AAPLMesh
{
    NSArray<AAPLSubmesh *> *_submeshes;
}

@synthesize submeshes = _submeshes;

/// Load the Model I/O mesh, including vertex data and submesh data which have index buffers and
///   textures.  Also generate tangent and bitangent vertex attributes
- (nonnull instancetype)initWithModelIOMesh:(nonnull MDLMesh *)modelIOMesh
                    modelIOVertexDescriptor:(nonnull MDLVertexDescriptor *)vertexDescriptor
                      metalKitTextureLoader:(nonnull MTKTextureLoader *)textureLoader
                                metalDevice:(nonnull id<MTLDevice>)device
                                      error:(NSError * __nullable * __nullable)error
{
    self = [super init];
    if(!self)
    {
        return nil;
    }

    // Have Model I/O create the tangents from mesh texture coordinates and normals
    [modelIOMesh addTangentBasisForTextureCoordinateAttributeNamed:MDLVertexAttributeTextureCoordinate
                                              normalAttributeNamed:MDLVertexAttributeNormal
                                             tangentAttributeNamed:MDLVertexAttributeTangent];

    // Have Model I/O create bitangents from mesh texture coordinates and the newly created tangents
    [modelIOMesh addTangentBasisForTextureCoordinateAttributeNamed:MDLVertexAttributeTextureCoordinate
                                             tangentAttributeNamed:MDLVertexAttributeTangent
                                           bitangentAttributeNamed:MDLVertexAttributeBitangent];

    // Apply the Model I/O vertex descriptor we created to match the Metal vertex descriptor.
    // Assigning a new vertex descriptor to a Model I/O mesh performs a re-layout of the vertex
    //   data.  In this case we created the Model I/O vertex descriptor so that the layout of the
    //   vertices in the Model I/O mesh match the layout of vertices our Metal render pipeline
    //   expects as input into its vertex shader
    // Note that we can only perform this re-layout operation after we have created tangents and
    //   bitangents (as we did above).  This is because Model IO's addTangentBasis methods only work
    //   with vertex data is all in 32-bit floating-point.  The vertex descriptor we're applying
    //   changes some 32-bit floats into 16-bit floats or other types from which Model I/O cannot
    //   produce tangents

    modelIOMesh.vertexDescriptor = vertexDescriptor;

    // Create the MetalKit mesh which will contain the Metal buffer(s) with the mesh's vertex data
    //   and submeshes with info to draw the mesh
    MTKMesh* metalKitMesh = [[MTKMesh alloc] initWithMesh:modelIOMesh
                                                   device:device
                                                    error:error];

    _metalKitMesh = metalKitMesh;

    // There should always be the same number of MetalKit submeshes in the MetalKit mesh as there
    //   are Model I/O submeshes in the Model I/O mesh
    assert(metalKitMesh.submeshes.count == modelIOMesh.submeshes.count);

    // Create an array to hold this AAPLMesh object's AAPLSubmesh objects
    NSMutableArray<AAPLSubmesh*>* mutableSubmeshes =
        [[NSMutableArray alloc] initWithCapacity:metalKitMesh.submeshes.count];

    // Create an AAPLSubmesh object for each submesh and a add it to our submeshes array
    for(NSUInteger index = 0; index < metalKitMesh.submeshes.count; index++)
    {
        // Create our own app specific submesh to hold the MetalKit submesh
        AAPLSubmesh *submesh =
        [[AAPLSubmesh alloc] initWithModelIOSubmesh:modelIOMesh.submeshes[index]
                                    metalKitSubmesh:metalKitMesh.submeshes[index]
                              metalKitTextureLoader:textureLoader];

        [mutableSubmeshes addObject:submesh];
    }

    _submeshes = mutableSubmeshes;

    return self;
}

/// Traverses the Model I/O object hierarchy picking out Model I/O mesh objects and creating Metal
///   vertex buffers, index buffers, and textures from them
+ (nullable NSArray<AAPLMesh*> *)newMeshesFromObject:(nonnull MDLObject*)object
                             modelIOVertexDescriptor:(nonnull MDLVertexDescriptor*)vertexDescriptor
                               metalKitTextureLoader:(MTKTextureLoader *)textureLoader
                                         metalDevice:(nonnull id<MTLDevice>)device
                                               error:(NSError * __nullable * __nullable)error
{

    NSMutableArray<AAPLMesh *> *newMeshes = [[NSMutableArray alloc] init];

    // If this Model I/O  object is a mesh object (not a camera, light, or something else)...
    if ([object isKindOfClass:[MDLMesh class]])
    {
        //...create an app-specific AAPLMesh object from it
        MDLMesh* mesh = (MDLMesh*) object;

        AAPLMesh *newMesh = [[AAPLMesh alloc] initWithModelIOMesh:mesh
                                          modelIOVertexDescriptor:vertexDescriptor
                                            metalKitTextureLoader:textureLoader
                                                      metalDevice:device
                                                            error:error];

        [newMeshes addObject:newMesh];
    }

    // Recursively traverse the Model I/O  asset hierarchy to find Model I/O  meshes that are children
    //   of this Model I/O  object and create app-specific AAPLMesh objects from those Model I/O meshes
    for (MDLObject *child in object.children)
    {
        NSArray<AAPLMesh*> *childMeshes;

        childMeshes = [AAPLMesh newMeshesFromObject:child
                            modelIOVertexDescriptor:vertexDescriptor
                              metalKitTextureLoader:textureLoader
                                        metalDevice:device
                                              error:error];

        [newMeshes addObjectsFromArray:childMeshes];
    }

    return newMeshes;
}

/// Uses Model I/O to load a model file at the given URL, create Model I/O vertex buffers, index buffers
///   and textures, applying the given Model I/O vertex descriptor to re-layout vertex attribute data
///   in the way that our Metal vertex shaders expect
+ (nullable NSArray<AAPLMesh *> *)newMeshesFromUrl:(nonnull NSURL *)url
                           modelIOVertexDescriptor:(nonnull MDLVertexDescriptor *)vertexDescriptor
                                       metalDevice:(nonnull id<MTLDevice>)device
                                             error:(NSError * __nullable * __nullable)error
                                              aabb:(MDLAxisAlignedBoundingBox&)aabb
{

    // Create a MetalKit mesh buffer allocator so that Model I/O  will load mesh data directly into
    //   Metal buffers accessible by the GPU
    MTKMeshBufferAllocator *bufferAllocator =
        [[MTKMeshBufferAllocator alloc] initWithDevice:device];

    // Use Model I/O to load the model file at the URL.  This returns a Model I/O asset object, which
    //   contains a hierarchy of Model I/O objects composing a "scene" described by the model file.
    //   This hierarchy may include lights, cameras, but, most importantly, mesh and submesh data
    //   that we'll render with Metal
    MDLAsset *asset = [[MDLAsset alloc] initWithURL:url
                                   vertexDescriptor:nil
                                    bufferAllocator:bufferAllocator];

    NSAssert(asset, @"Failed to open model file with given URL: %@", url.absoluteString);
    
    aabb = [asset boundingBox];

    // Create a MetalKit texture loader to load material textures from files or the asset catalog
    //   into Metal textures
    MTKTextureLoader *textureLoader = [[MTKTextureLoader alloc] initWithDevice:device];

    NSMutableArray<AAPLMesh *> *newMeshes = [[NSMutableArray alloc] init];

    // Traverse the Model I/O asset hierarchy to find Model I/O meshes and create app-specific
    //   AAPLMesh objects from those Model I/O meshes
    for(MDLObject* object in asset)
    {
        NSArray<AAPLMesh *> *assetMeshes;

        assetMeshes = [AAPLMesh newMeshesFromObject:object
                            modelIOVertexDescriptor:vertexDescriptor
                              metalKitTextureLoader:textureLoader
                                        metalDevice:device
                                              error:error];

        [newMeshes addObjectsFromArray:assetMeshes];
    }

    return newMeshes;
}

@end
