/**
 * GPU fluid simulation for the WebGL gradient.
 *
 * Implements a subset of Jos Stam's "Stable Fluids" on the GPU using
 * ping-pong framebuffers: each frame we run a semi-Lagrangian advection
 * with a small dissipation factor, pointer motion injects velocity via a
 * Gaussian splat, and a subtle curl-confinement pass preserves slow rolling
 * wakes. Pressure projection is intentionally omitted, keeping the effect
 * light enough for a background gradient while still allowing gentle vortex
 * shedding after pointer movement.
 *
 * The resulting velocity texture is sampled in the main gradient shader
 * to displace the noise sampling coordinates, so dragging across the
 * canvas warps the gradient as if pushing through a viscous fluid.
 *
 * References:
 * - Jos Stam, "Stable Fluids", 1999: https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
 * - Alex Harri, "WebGL Gradients", 2025: https://alexharri.com/blog/webgl-gradients
 */

import { bindFullscreenQuad, createFullscreenQuadBuffer, createProgram } from './glUtils';

const SIM_RESOLUTION = 256;
const CURL_RADIUS_TEXELS = 7.00;
const CURL_STRENGTH = 30.0;
const CURL_FORCE_CLAMP = 0.20;
const FLUID_PROGRAM_LOG_LABELS = {
  shaderCompile: 'Fluid shader compile error:',
  programLink: 'Fluid program link error:',
} as const;

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

    vec2 aspectForce = vec2(u_force.x * u_aspect, u_force.y);
    float forceLength = length(aspectForce);
    vec2 direction = aspectForce / max(forceLength, 0.0001);
    vec2 tangent = vec2(-direction.y, direction.x);
    float along = dot(p, direction);
    float side = dot(p, tangent);
    float sideNorm = side / sqrt(max(u_radius, 0.00001));
    float tail = 1.0 - smoothstep(-0.02, 0.035, along);
    float wakeMask = gauss * tail * clamp(abs(sideNorm) * 1.75, 0.0, 1.0);
    vec2 wakeAspect = tangent * sign(side) * forceLength * wakeMask * 0.055;
    vec2 wake = vec2(wakeAspect.x / max(u_aspect, 0.0001), wakeAspect.y);

    gl_FragColor = vec4(base + u_force * gauss + wake, 0.0, 1.0);
  }
`;

const CURL_SHADER = /* glsl */ `
  precision highp float;
  uniform sampler2D u_velocity;
  uniform vec2 u_texelSize;
  uniform float u_dt;
  uniform float u_strength;
  uniform float u_forceClamp;
  varying vec2 v_uv;

  float curlAt(vec2 uv) {
    vec2 leftVelocity = texture2D(u_velocity, uv - vec2(u_texelSize.x, 0.0)).xy;
    vec2 rightVelocity = texture2D(u_velocity, uv + vec2(u_texelSize.x, 0.0)).xy;
    vec2 bottomVelocity = texture2D(u_velocity, uv - vec2(0.0, u_texelSize.y)).xy;
    vec2 topVelocity = texture2D(u_velocity, uv + vec2(0.0, u_texelSize.y)).xy;
    return 0.5 * ((rightVelocity.y - leftVelocity.y) - (topVelocity.x - bottomVelocity.x));
  }

  void main() {
    vec2 velocity = texture2D(u_velocity, v_uv).xy;
    float curl = curlAt(v_uv);
    float leftCurl = abs(curlAt(v_uv - vec2(u_texelSize.x, 0.0)));
    float rightCurl = abs(curlAt(v_uv + vec2(u_texelSize.x, 0.0)));
    float bottomCurl = abs(curlAt(v_uv - vec2(0.0, u_texelSize.y)));
    float topCurl = abs(curlAt(v_uv + vec2(0.0, u_texelSize.y)));

    vec2 gradient = 0.5 * vec2(rightCurl - leftCurl, topCurl - bottomCurl);
    vec2 normal = gradient / (length(gradient) + 0.0001);
    vec2 force = vec2(normal.y * curl, -normal.x * curl) * u_strength;
    force = clamp(force, vec2(-u_forceClamp), vec2(u_forceClamp));

    gl_FragColor = vec4(velocity + force * u_dt, 0.0, 1.0);
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
  private curlProgram: WebGLProgram;
  private quadBuffer: WebGLBuffer;
  private uniforms: {
    advect: { velocity: WebGLUniformLocation | null; dt: WebGLUniformLocation | null; dissipation: WebGLUniformLocation | null };
    splat: { target: WebGLUniformLocation | null; point: WebGLUniformLocation | null; force: WebGLUniformLocation | null; radius: WebGLUniformLocation | null; aspect: WebGLUniformLocation | null };
    curl: { velocity: WebGLUniformLocation | null; texelSize: WebGLUniformLocation | null; dt: WebGLUniformLocation | null; strength: WebGLUniformLocation | null; forceClamp: WebGLUniformLocation | null };
  };
  private pendingSplats: PointerSplat[] = [];
  /** Aspect ratio (width / height) used to keep splats circular on screen. */
  aspect: number = 1;

  private constructor(
    gl: WebGLRenderingContext,
    velocity: DoubleFBO,
    advectProgram: WebGLProgram,
    splatProgram: WebGLProgram,
    curlProgram: WebGLProgram,
    quadBuffer: WebGLBuffer,
  ) {
    this.gl = gl;
    this.velocity = velocity;
    this.advectProgram = advectProgram;
    this.splatProgram = splatProgram;
    this.curlProgram = curlProgram;
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
      curl: {
        velocity: gl.getUniformLocation(curlProgram, 'u_velocity'),
        texelSize: gl.getUniformLocation(curlProgram, 'u_texelSize'),
        dt: gl.getUniformLocation(curlProgram, 'u_dt'),
        strength: gl.getUniformLocation(curlProgram, 'u_strength'),
        forceClamp: gl.getUniformLocation(curlProgram, 'u_forceClamp'),
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

    const advectProgram = createProgram(gl, VERTEX_SHADER, ADVECT_SHADER, FLUID_PROGRAM_LOG_LABELS);
    const splatProgram = createProgram(gl, VERTEX_SHADER, SPLAT_SHADER, FLUID_PROGRAM_LOG_LABELS);
    const curlProgram = createProgram(gl, VERTEX_SHADER, CURL_SHADER, FLUID_PROGRAM_LOG_LABELS);
    if (!advectProgram || !splatProgram || !curlProgram) return null;

    // Quad covering clip space, shared by all fluid passes.
    const quadBuffer = createFullscreenQuadBuffer(gl);
    if (!quadBuffer) return null;

    return new FluidSimulation(gl, velocity, advectProgram, splatProgram, curlProgram, quadBuffer);
  }

  private bindPassProgram(program: WebGLProgram) {
    const gl = this.gl;
    gl.useProgram(program);
    bindFullscreenQuad(gl, program, this.quadBuffer);
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

    gl.viewport(0, 0, SIM_RESOLUTION, SIM_RESOLUTION);
    gl.disable(gl.BLEND);

    // 1. Advect the velocity field through itself.
    this.bindPassProgram(this.advectProgram);
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
      this.bindPassProgram(this.splatProgram);
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

    // 3. Preserve small rolling wakes with a restrained curl-confinement pass.
    this.bindPassProgram(this.curlProgram);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
    gl.uniform1i(this.uniforms.curl.velocity, 1);
    gl.uniform2f(
      this.uniforms.curl.texelSize,
      CURL_RADIUS_TEXELS / SIM_RESOLUTION,
      CURL_RADIUS_TEXELS / SIM_RESOLUTION,
    );
    gl.uniform1f(this.uniforms.curl.dt, clampedDt);
    gl.uniform1f(this.uniforms.curl.strength, CURL_STRENGTH);
    gl.uniform1f(this.uniforms.curl.forceClamp, CURL_FORCE_CLAMP);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.write.framebuffer);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    this.velocity.swap();

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
    gl.deleteProgram(this.curlProgram);
    gl.deleteBuffer(this.quadBuffer);
  }
}
