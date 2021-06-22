#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 vertexUV;
layout(location = 2) in vec3 vertexNormal_modelspace;

out vec2 UV;
out vec3 Normal_cameraspace;

uniform mat4 MVP;
uniform mat4 V;
uniform mat4 M;

void main()
{
	gl_Position = MVP * vec4(vertexPosition_modelspace, 1.0);

	UV = vertexUV;
	Normal_cameraspace = (V * M * vec4(vertexNormal_modelspace, 0.0)).xyz;
}
