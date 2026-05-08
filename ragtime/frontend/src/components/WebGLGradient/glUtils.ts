export const FULLSCREEN_QUAD_VERTICES = new Float32Array([
  -1, -1,
   1, -1,
  -1,  1,
   1,  1,
]);

interface ProgramLogLabels {
  shaderCompile?: string;
  programLink?: string;
}

export function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string,
  errorLabel = 'Shader compilation error:',
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(errorLabel, gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

export function createProgram(
  gl: WebGLRenderingContext,
  vertexShaderSource: string,
  fragmentShaderSource: string,
  labels: ProgramLogLabels = {},
): WebGLProgram | null {
  const vertexShader = compileShader(
    gl,
    gl.VERTEX_SHADER,
    vertexShaderSource,
    labels.shaderCompile,
  );
  const fragmentShader = compileShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource,
    labels.shaderCompile,
  );
  if (!vertexShader || !fragmentShader) return null;

  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error(labels.programLink ?? 'Program linking error:', gl.getProgramInfoLog(program));
    return null;
  }

  return program;
}

export function createFullscreenQuadBuffer(gl: WebGLRenderingContext): WebGLBuffer | null {
  const buffer = gl.createBuffer();
  if (!buffer) return null;

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, FULLSCREEN_QUAD_VERTICES, gl.STATIC_DRAW);

  return buffer;
}

export function bindFullscreenQuad(
  gl: WebGLRenderingContext,
  program: WebGLProgram,
  buffer: WebGLBuffer,
) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  const positionLocation = gl.getAttribLocation(program, 'a_position');
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
}
