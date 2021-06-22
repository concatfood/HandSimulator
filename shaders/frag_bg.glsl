#version 330 core

in vec2 UV;

layout(location = 0) out vec4 color_0;
layout(location = 1) out vec4 color_1;
layout(location = 2) out vec4 color_2;
layout(location = 3) out vec4 color_3;

uniform sampler2D texture_sampler;

void main()
{
	color_0 = texture(texture_sampler, UV);
	color_1 = vec4(0.0, 1.0, 0.0, 1.0);
	color_2 = vec4(1.0, 1.0, 1.0, 1.0);
	color_3 = vec4(0.0, 0.0, 0.0, 1.0);
}
