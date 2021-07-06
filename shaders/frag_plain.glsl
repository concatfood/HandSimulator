#version 330 core

in vec2 UV;
in vec3 Normal_cameraspace;

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 segmentation;
layout(location = 2) out vec4 depth;
layout(location = 3) out vec4 normal;

uniform float far;
uniform int id;
uniform float near;
uniform sampler2D texture_sampler;

float linearize_depth(float depth)
{
    float z = depth * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main()
{
	color = texture(texture_sampler, UV);

	if (id == 0)
	{
		segmentation = vec4(1.0, 0.0, 0.0, 1.0);
	}
	else if (id == 1)
	{
		segmentation = vec4(0.0, 0.0, 1.0, 1.0);
	}

//	depth = vec4(vec3(linearize_depth(gl_FragCoord.z) / far), 1.0);
	depth = vec4(vec3(gl_FragCoord.z), 1.0);
	normal = vec4((normalize(Normal_cameraspace) + vec3(1.0)) / 2.0, 1.0);
}
