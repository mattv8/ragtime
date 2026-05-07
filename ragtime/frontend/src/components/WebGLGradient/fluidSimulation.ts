/**
 * GPU fluid simulation for the WebGL gradient.
 *
 * Implements a subset of Jos Stam's "Stable Fluids" on the GPU using
 * ping-pong framebuffers: each frame we run a semi-Lagrangian advection
 * with a small dissipation factor, and pointer motion injects velocity
 * via a Gaussian splat. Pressure projection is intentionally omitted -
 * the fluid is purely advective which produces the slow, smoky, drifting
 * perturbations called for by the design (similar in spirit to the x.ai
 * homepage), at a fraction of the cost of a full Navier-Stokes solver.
 *
 * The resulting velocity texture is sampled in the main gradient shader
 * to displace the noise sampling coordinates, so dragging across the
 * canvas warps the gradient as if pushing through a viscous fluid.
 */

const SIM_RESOLUTION = 256;

const VERTEX_SHADER = /* glsl */ `
  attribute vec2 a_position;
  varying vec2 v_uv;
  void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

const ADVECT_SHADER = /* glsl */ `
  precision highp float;
  uniform sampler2D u_velocity;
  uniform float u_dt;
  uniform float u_dissipation;
  varying vec2 v_uv;
  void main() {
    vec2 vel = texture2D(u_velocity, v_uv).xy;
    // Backwards-trace in uv-space. Velocity is stored in uv-units/second.
    vec2 coord = v_uv - vel * u_dt;
    vec2 newVel = texture2D(u_velocity, coord).xy * u_dissipation;
    gl_FragColor = vec4(newVel, 0.0, 1.0);
  }
`;

const SPLAT_SHADER = /* glsl */ `
  precision highp float;
  uniform sampler2D u_target;
  uniform vec2 u_point;
  uniform vec2 u_force;
  uniform float u_radius;
  uniform float u_aspect;
  varying vec2 v_uv;
  void main() {
    vec2 p = v_uv - u_point;
    p.x *= u_aspect;
    float gauss = exp(-dot(p, p) / u_radius);
    vec2 base = texture2D(u_target, v_uv).xy;
    gl_FragColor = vec4(base + u_force * gauss, 0.0, 1.0);
  }
`;

interface FBO {
  texture: WebGLTexture;
  framebuffer: WebGLFramebuffer;
}

interface DoubleFBO {
  read: FBO;
  write: FBO;
  swap: () => void;
}

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Fluid shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl: WebGLRenderingContext, vsSource: string, fsSource: string): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
  if (!vs || !fs) return null;
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Fluid program link error:', gl.getProgramInfoLog(program));
    return null;
  }
  return program;
}

function createFBO(gl: WebGLRenderingContext, width: number, height: number, internalType: number): FBO | null {
  const texture = gl.createTexture();
  if (!texture) return null;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, internalType, null);

  const framebuffer = gl.createFramebuffer();
  if (!framebuffer) return null;
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    console.warn('Fluid FBO incomplete:', status);
    return null;
  }

  // Clear to zero velocity.
  gl.viewport(0, 0, width, height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  return { texture, framebuffer };
}

function createDoubleFBO(gl: WebGLRenderingContext, width: number, height: number, type: number): DoubleFBO | null {
  const a = createFBO(gl, width, height, type);
  const b = createFBO(gl, width, height, type);
  if (!a || !b) return null;
  let read = a;
  let write = b;
  return {
    get read() { return read; },
    get write() { return write; },
    swap() { const t = read; read = write; write = t; },
  };
}

export interface PointerSplat {
  /** Position in normalized [0,1] coordinates (origin bottom-left to match WebGL uv). */
  x: number;
  y: number;
  /** Velocity vector in uv-units per second. */
  fx: number;
  fy: number;
}

export class FluidSimulation {
  private gl: WebGLRenderingContext;
  private velocity: DoubleFBO;
  private advectProgram: WebGLProgram;
  private splatProgram: WebGLProgram;
  private quadBuffer: WebGLBuffer;
  private uniforms: {
    advect: { velocity: WebGLUniformLocation | null; dt: WebGLUniformLocation | null; dissipation: WebGLUniformLocation | null };
    splat: { target: WebGLUniformLocation | null; point: WebGLUniformLocation | null; force: WebGLUniformLocation | null; radius: WebGLUniformLocation | null; aspect: WebGLUniformLocation | null };
  };
  private pendingSplats: PointerSplat[] = [];
  /** Aspect ratio (width / height) used to keep splats circular on screen. */
  aspect: number = 1;

  private constructor(
    gl: WebGLRenderingContext,
    velocity: DoubleFBO,
    advectProgram: WebGLProgram,
    splatProgram: WebGLProgram,
    quadBuffer: WebGLBuffer,
  ) {
    this.gl = gl;
    this.velocity = velocity;
    this.advectProgram = advectProgram;
    this.splatProgram = splatProgram;
    this.quadBuffer = quadBuffer;
    this.uniforms = {
      advect: {
        velocity: gl.getUniformLocation(advectProgram, 'u_velocity'),
        dt: gl.getUniformLocation(advectProgram, 'u_dt'),
        dissipation: gl.getUniformLocation(advectProgram, 'u_dissipation'),
      },
      splat: {
        target: gl.getUniformLocation(splatProgram, 'u_target'),
        point: gl.getUniformLocation(splatProgram, 'u_point'),
        force: gl.getUniformLocation(splatProgram, 'u_force'),
        radius: gl.getUniformLocation(splatProgram, 'u_radius'),
        aspect: gl.getUniformLocation(splatProgram, 'u_aspect'),
      },
    };
  }

  static create(gl: WebGLRenderingContext): FluidSimulation | null {
    // Need float-render-target support. Prefer half-float (cheaper, almost
    // universally available); fall back to full float; otherwise bail and
    // the caller will simply skip the fluid effect.
    const halfFloatExt = gl.getExtension('OES_texture_half_float');
    const halfFloatLinearExt = gl.getExtension('OES_texture_half_float_linear');
    const floatExt = gl.getExtension('OES_texture_float');
    const floatLinearExt = gl.getExtension('OES_texture_float_linear');

    let textureType: number | null = null;
    if (halfFloatExt) {
      textureType = halfFloatExt.HALF_FLOAT_OES;
    } else if (floatExt) {
      textureType = gl.FLOAT;
    } else {
      return null;
    }

    // Some drivers expose the extension but won't actually render to the
    // format. Verify by trying to create one FBO before committing.
    const probe = createFBO(gl, 4, 4, textureType);
    if (!probe) {
      // Try the other type as a fallback.
      if (textureType !== gl.FLOAT && floatExt) {
        const probe2 = createFBO(gl, 4, 4, gl.FLOAT);
        if (!probe2) return null;
        gl.deleteTexture(probe2.texture);
        gl.deleteFramebuffer(probe2.framebuffer);
        textureType = gl.FLOAT;
      } else {
        return null;
      }
    } else {
      gl.deleteTexture(probe.texture);
      gl.deleteFramebuffer(probe.framebuffer);
    }

    // Linear filtering on float textures isn't always available; if missing
    // the texture sampling falls back to nearest, which still works fine for
    // the visual result though it produces slightly blockier swirls.
    void halfFloatLinearExt;
    void floatLinearExt;

    const velocity = createDoubleFBO(gl, SIM_RESOLUTION, SIM_RESOLUTION, textureType);
    if (!velocity) return null;

    const advectProgram = createProgram(gl, VERTEX_SHADER, ADVECT_SHADER);
    const splatProgram = createProgram(gl, VERTEX_SHADER, SPLAT_SHADER);
    if (!advectProgram || !splatProgram) return null;

    // Quad covering clip space, shared by all fluid passes.
    const quadBuffer = gl.createBuffer();
    if (!quadBuffer) return null;
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );

    return new FluidSimulation(gl, velocity, advectProgram, splatProgram, quadBuffer);
  }

  /** Queue a velocity injection at the given uv-space point. */
  addSplat(splat: PointerSplat) {
    this.pendingSplats.push(splat);
  }

  /** Returns the current velocity texture for the main shader to sample. */
  get velocityTexture(): WebGLTexture {
    return this.velocity.read.texture;
  }

  /**
   * Step the simulation forward by `dt` seconds.
   * The caller is responsible for restoring its own GL state (viewport,
   * program, bound framebuffer, vertex attributes) after this returns.
   */
  step(dt: number) {
    const gl = this.gl;
    const clampedDt = Math.min(Math.max(dt, 0), 1 / 30);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.viewport(0, 0, SIM_RESOLUTION, SIM_RESOLUTION);
    gl.disable(gl.BLEND);

    // 1. Advect the velocity field through itself.
    gl.useProgram(this.advectProgram);
    const aPosAdvect = gl.getAttribLocation(this.advectProgram, 'a_position');
    gl.enableVertexAttribArray(aPosAdvect);
    gl.vertexAttribPointer(aPosAdvect, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
    gl.uniform1i(this.uniforms.advect.velocity, 1);
    gl.uniform1f(this.uniforms.advect.dt, clampedDt);
    // Slow viscosity-like decay so the swirls linger but eventually fade.
    gl.uniform1f(this.uniforms.advect.dissipation, 0.996);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.write.framebuffer);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    this.velocity.swap();

    // 2. Apply any pending pointer splats on top of the advected field.
    if (this.pendingSplats.length) {
      gl.useProgram(this.splatProgram);
      const aPosSplat = gl.getAttribLocation(this.splatProgram, 'a_position');
      gl.enableVertexAttribArray(aPosSplat);
      gl.vertexAttribPointer(aPosSplat, 2, gl.FLOAT, false, 0, 0);
      gl.uniform1f(this.uniforms.splat.aspect, this.aspect);
      // Splat radius in (aspect-corrected) uv squared. Tuned for a soft,
      // roughly thumb-sized influence regardless of canvas size.
      gl.uniform1f(this.uniforms.splat.radius, 0.00115);

      for (const splat of this.pendingSplats) {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        gl.uniform1i(this.uniforms.splat.target, 1);
        gl.uniform2f(this.uniforms.splat.point, splat.x, splat.y);
        gl.uniform2f(this.uniforms.splat.force, splat.fx, splat.fy);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.write.framebuffer);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        this.velocity.swap();
      }
      this.pendingSplats.length = 0;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  dispose() {
    const gl = this.gl;
    gl.deleteTexture(this.velocity.read.texture);
    gl.deleteTexture(this.velocity.write.texture);
    gl.deleteFramebuffer(this.velocity.read.framebuffer);
    gl.deleteFramebuffer(this.velocity.write.framebuffer);
    gl.deleteProgram(this.advectProgram);
    gl.deleteProgram(this.splatProgram);
    gl.deleteBuffer(this.quadBuffer);
  }
}
