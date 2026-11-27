#version 450
// Phase 2: one mipmapped texture.
layout(set = 0, binding = 3) uniform sampler2D diffuse;

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 fragColor;

void main()
{
	fragColor = vec4(texture(diffuse, texCoord).rgb, 1.0);
}
