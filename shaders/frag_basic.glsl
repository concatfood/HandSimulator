#version 330 core

in vec2 UV;
in vec3 Position_worldspace;
in vec3 Normal_cameraspace;
in vec3 EyeDirection_cameraspace;
in vec3 LightDirection_cameraspace;

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 segmentation;
layout(location = 2) out vec4 depth;
layout(location = 3) out vec4 normal;

uniform float far;
uniform int id;
uniform vec3 LightPosition_worldspace;
uniform float near;
uniform sampler2D texture_sampler;

float linearize_depth(float depth)
{
    float z = depth * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main()
{
	vec3 light_color = vec3(1.0, 1.0, 1.0);
	float light_power = 50.0;

	vec3 material_diffuse_color = texture(texture_sampler, UV).rgb;
	vec3 material_ambient_color = vec3(0.2, 0.2, 0.2) * material_diffuse_color;	// set to 0.2 later
//	vec3 material_specular_color = vec3(0.3, 0.3, 0.3);
//
//	float distance = length(LightPosition_worldspace - Position_worldspace);

	vec3 n = normalize(Normal_cameraspace);
	vec3 l = normalize(LightDirection_cameraspace);
	float cos_theta = clamp(dot(n, l), 0.0, 1.0);

//	vec3 E = normalize(EyeDirection_cameraspace);
//	vec3 R = reflect(-l, n);
//	float cosAlpha = clamp(dot(E, R), 0.0, 1.0);
//
//	color = vec4(material_ambient_color +
//			material_diffuse_color * light_color * light_power * cos_theta / (distance*distance) +
//			material_specular_color * light_color * light_power * pow(cosAlpha, 5.0) / (distance*distance), 1.0);
//
//	color = vec4(material_diffuse_color * light_color * light_power * cos_theta / (distance * distance), 1.0);

	color = vec4(material_ambient_color +
			material_diffuse_color * light_color * cos_theta, 1.0);

	if (id == 0)
	{
		segmentation = vec4(1.0, 0.0, 0.0, 1.0);
	}
	else if (id == 1)
	{
		segmentation = vec4(0.0, 0.0, 1.0, 1.0);
	}

	depth = vec4(vec3(linearize_depth(gl_FragCoord.z) / far), 1.0);
	normal = vec4((normalize(Normal_cameraspace) + vec3(1.0)) / 2.0, 1.0);
}
