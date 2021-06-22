from OpenGL.GL import *


# load and compile a pair of vertex and fragment shaders
def load_shaders(vertex_file_path, fragment_file_path):
    vertex_file_path = 'shaders/' + vertex_file_path
    fragment_file_path = 'shaders/' + fragment_file_path

    vertex_shader_id = glCreateShader(GL_VERTEX_SHADER)
    fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER)

    with open(vertex_file_path, 'r') as content_file:
        vertex_shader_code = content_file.read()

    with open(fragment_file_path, 'r') as content_file:
        fragment_shader_code = content_file.read()

    print('Compiling shader: {path}'.format(path=vertex_file_path))

    glShaderSource(vertex_shader_id, vertex_shader_code)
    glCompileShader(vertex_shader_id)

    glGetShaderiv(vertex_shader_id, GL_COMPILE_STATUS)
    info_log_length = glGetShaderiv(vertex_shader_id, GL_INFO_LOG_LENGTH)

    if info_log_length > 0:
        vertex_shader_error_message = glGetShaderInfoLog(vertex_shader_id)

        print(vertex_shader_error_message)

    print('Compiling shader: {path}'.format(path=fragment_file_path))

    glShaderSource(fragment_shader_id, fragment_shader_code)
    glCompileShader(fragment_shader_id)

    glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS)
    info_log_length = glGetShaderiv(fragment_shader_id, GL_INFO_LOG_LENGTH)

    if info_log_length > 0:
        fragment_shader_error_message = glGetShaderInfoLog(fragment_shader_id)

        print(fragment_shader_error_message)

    print('Linking program')

    program_id = glCreateProgram()

    glAttachShader(program_id, vertex_shader_id)
    glAttachShader(program_id, fragment_shader_id)
    glLinkProgram(program_id)

    glGetProgramiv(program_id, GL_LINK_STATUS)
    info_log_length = glGetProgramiv(program_id, GL_INFO_LOG_LENGTH)

    if info_log_length > 0:
        program_error_message = glGetProgramInfoLog(program_id)

        print(program_error_message)

    glDetachShader(program_id, vertex_shader_id)
    glDetachShader(program_id, fragment_shader_id)

    glDeleteShader(vertex_shader_id)
    glDeleteShader(fragment_shader_id)

    return program_id
