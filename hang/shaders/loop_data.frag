#version 450
// The game's spotlight loop cut to its shape: six passes, counted by a uniform, as SHOCK_COUNT(6) is (6 * shockLoopScale), so xost runs it as a 255-pass hardware loop left on its condition.
layout(location = 0) in vec2 texCoord;
layout(location = 3) in vec3 worldPos;
layout(set = 0, binding = 1) uniform Fragment {
	vec4 ambient;
	vec4 params;
};
layout(location = 0) out vec4 fragColor;
void main()
{
	vec3 sum = ambient.rgb;
	for (int i = 0; i < int(6.0 * params.w); i++)
		sum += 0.01 * max(dot(normalize(worldPos + vec3(float(i))), vec3(0.0, 0.0, 1.0)), 0.0);
	fragColor = vec4(sum, 1.0);
}
