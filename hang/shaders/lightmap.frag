#version 450
// Phase 3: shock2quest's world fragment shader cut to its lightmap path: the atlas at a wrapped coordinate, the
// diffuse, an alpha test, the ambient floor and distance fog.
layout(set = 0, binding = 1) uniform Fragment {
	vec4 ambient;
	// x: lightmap intensity, y: alpha test (> 0.5), z: fog density.
	vec4 params;
} f;
layout(set = 0, binding = 2) uniform sampler2D lightmap;
layout(set = 0, binding = 3) uniform sampler2D diffuse;

layout(location = 0) in vec2 texCoord;
layout(location = 1) in vec2 lightMapTexCoord;
layout(location = 2) in vec4 atlasCoord;
layout(location = 3) in vec3 worldPos;
layout(location = 4) in vec3 worldNormal;
layout(location = 0) out vec4 fragColor;

void main()
{
	float half_pixel = 0.5 / 4096.0;
	float full_pixel = half_pixel * 2.0;
	float width = atlasCoord.z - full_pixel;
	float height = atlasCoord.w - full_pixel;
	vec2 wrapped = vec2(mod(lightMapTexCoord.x * width, width) + atlasCoord.x + half_pixel,
		mod(lightMapTexCoord.y * height, height) + atlasCoord.y + half_pixel);
	vec4 light = texture(lightmap, wrapped) * f.params.x;
	vec4 texel = texture(diffuse, texCoord);
	if (f.params.y > 0.5 && texel.a < 0.1)
		discard;
	vec3 colour = max(texel.rgb * light.rgb, f.ambient.rgb);
	colour *= exp(-f.params.z * length(worldPos));
	fragColor = vec4(colour + 1e-6 * worldNormal, 1.0);
}
