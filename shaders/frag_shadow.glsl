#version 330 core

in vec2 UV;
in vec3 Position_worldspace;
in vec3 Normal_cameraspace;
in vec3 EyeDirection_cameraspace;
in vec3 LightDirection_cameraspace;
in vec4 ShadowCoord;

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 segmentation;
layout(location = 2) out vec4 depth;
layout(location = 3) out vec4 normal;

uniform float far;
uniform int id;
uniform float near;
uniform sampler2D texture_sampler;
uniform sampler2DShadow shadowMap;

vec2 poisson_disk[16] = vec2[]
(
   vec2(-0.94201624, -0.39906216),
   vec2(0.94558609, -0.76890725),
   vec2(-0.094184101, -0.92938870),
   vec2(0.34495938, 0.29387760),
   vec2(-0.91588581, 0.45771432),
   vec2(-0.81544232, -0.87912464),
   vec2(-0.38277543, 0.27676845),
   vec2(0.97484398, 0.75648379),
   vec2(0.44323325, -0.97511554),
   vec2(0.53742981, -0.47373420),
   vec2(-0.26496911, -0.41893023),
   vec2(0.79197514, 0.19090188),
   vec2(-0.24188840, 0.99706507),
   vec2(-0.81409955, 0.91437590),
   vec2(0.19984126, 0.78641367),
   vec2(0.14383161, -0.14100790)
);

float linearize_depth(float depth)
{
    float z = depth * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

float random(vec3 seed, int i)
{
	vec4 seed4 = vec4(seed, i);
	float dot_product = dot(seed4, vec4(12.9898, 78.233, 45.164, 94.673));
	return fract(sin(dot_product) * 43758.5453);
}

void main()
{
	vec3 LightColor = vec3(1.0, 1.0, 1.0);
	float LightPower = 1.0;

	vec3 material_diffuse_color = texture(texture_sampler, UV).rgb;
	vec3 material_ambient_color = vec3(0.2, 0.2, 0.2) * material_diffuse_color;
	vec3 material_specular_color = vec3(0.0, 0.0, 0.0);

	vec3 n = normalize(Normal_cameraspace);
	vec3 l = normalize(LightDirection_cameraspace);
	float cos_theta = clamp(dot(n, l), 0.0, 1.0);

	vec3 E = normalize(EyeDirection_cameraspace);
	vec3 R = reflect(-l, n);
	float cos_alpha = clamp(dot(E, R), 0.0, 1.0);
	
	float visibility = 1.0;

	float bias = 0.005;

	for (int i = 0; i < 4; i++)
	{
		int index = i;

		visibility -= 0.2 * (1.0 - texture(shadowMap, vec3(ShadowCoord.xy + poisson_disk[index] / 700.0, (ShadowCoord.z - bias) / ShadowCoord.w)));
	}
	
	color = vec4(material_ambient_color
			+ visibility * material_diffuse_color * LightColor * LightPower * cos_theta
			+ visibility * material_specular_color * LightColor * LightPower * pow(cos_alpha, 5.0), 1.0);

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