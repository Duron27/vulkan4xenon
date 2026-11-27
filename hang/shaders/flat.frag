#version 450
// Phase 1: a colour from the uniform block, nothing sampled.
layout(set = 0, binding = 1) uniform Fragment {
	vec4 ambient;
	vec4 params;
} f;

layout(location = 0) out vec4 fragColor;

void main()
{
	fragColor = vec4(f.ambient.rgb, 1.0);
}
