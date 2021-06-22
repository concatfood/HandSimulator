#version 330 core

layout(location = 0) in vec2 vertexPosition;
layout(location = 1) in vec2 vertexUV;

out vec2 UV;

void main()
{
	vec2 vertexPosition_h = vertexPosition - vec2(400, 300);
	vertexPosition_h /= vec2(400, 300);
	gl_Position =  vec4(vertexPosition_h, 0, 1);
	
	UV = vertexUV;
}

