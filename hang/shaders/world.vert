#version 450
// shock2quest's world vertex: position, texture, lightmap and atlas coordinates, normal (56 bytes).
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inTex;
layout(location = 2) in vec2 inLightMapTex;
layout(location = 3) in vec4 inAtlas;
layout(location = 4) in vec3 inNormal;

layout(set = 0, binding = 0) uniform Vertex {
	mat4 transform;
} v;

layout(location = 0) out vec2 texCoord;
layout(location = 1) out vec2 lightMapTexCoord;
layout(location = 2) out vec4 atlasCoord;
layout(location = 3) out vec3 worldPos;
layout(location = 4) out vec3 worldNormal;

void main()
{
	texCoord = inTex;
	lightMapTexCoord = inLightMapTex;
	atlasCoord = inAtlas;
	worldPos = inPos;
	worldNormal = normalize(inNormal);
	gl_Position = v.transform * vec4(inPos, 1.0);
}
