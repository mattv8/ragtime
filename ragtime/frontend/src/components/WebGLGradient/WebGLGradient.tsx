import React, { useCallback, useEffect, useRef, useState } from 'react';
import createFragmentShader from './fragmentShader';
import { FluidSimulation } from './fluidSimulation';

const TEXTURE_WIDTH = 1024;
const DEFAULT_COLOR_VARIABLES = [
  '--login-gradient-start',
  '--login-gradient-middle',
  '--login-gradient-end',
] as const;
const FALLBACK_BLUE_COLORS = ['#0a1220', '#0f182a', '#354b61'] as const;

interface WebGLGradientProps {
  className?: string;
  colors?: readonly string[];
  colorVariables?: readonly string[];
  fullscreen?: boolean;
  ignorePointerSelector?: string;
}

const isMobileDevice = () => /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

const WebGLGradient: React.FC<WebGLGradientProps> = ({
  className = '',
  colors,
  colorVariables = DEFAULT_COLOR_VARIABLES,
  fullscreen = false,
  ignorePointerSelector,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const uniformsRef = useRef<{ [key: string]: WebGLUniformLocation | null }>({});
  const startTimeRef = useRef<number>(Date.now());
  const lastFrameTimeRef = useRef<number>(performance.now());
  const fluidRef = useRef<FluidSimulation | null>(null);
  const fluidStrengthRef = useRef<number>(0);
  const pointerStateRef = useRef<{
    active: boolean;
    lastX: number;
    lastY: number;
    lastT: number;
  }>({ active: false, lastX: 0, lastY: 0, lastT: 0 });
  const [themeRevision, setThemeRevision] = useState(0);

  const resolveGradientColors = useCallback(() => {
    if (colors && colors.length >= 2) {
      return [...colors];
    }

    const rootStyles = getComputedStyle(document.documentElement);
    const resolvedColors = colorVariables
      .map((variableName) => rootStyles.getPropertyValue(variableName).trim())
      .filter(Boolean);

    return resolvedColors.length >= 2 ? resolvedColors : [...FALLBACK_BLUE_COLORS];
  }, [colors, colorVariables]);

  const createGradientTexture = useCallback((gl: WebGLRenderingContext) => {
    const canvas = document.createElement('canvas');
    canvas.width = TEXTURE_WIDTH;
    canvas.height = 1;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const gradientColors = resolveGradientColors();
    const lastColorIndex = gradientColors.length - 1;
    const gradient = ctx.createLinearGradient(0, 0, TEXTURE_WIDTH, 0);

    gradientColors.forEach((color, index) => {
      gradient.addColorStop(lastColorIndex === 0 ? 0 : index / lastColorIndex, color);
    });

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, TEXTURE_WIDTH, 1);

    const texture = gl.createTexture();
    if (!texture) return null;

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    return texture;
  }, [resolveGradientColors]);

  const createShader = useCallback((gl: WebGLRenderingContext, type: number, source: string) => {
    const shader = gl.createShader(type);
    if (!shader) return null;

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }

    return shader;
  }, []);

  const updateGradientTexture = useCallback(() => {
    const gl = glRef.current;
    if (!gl) return false;

    const gradientTexture = createGradientTexture(gl);
    if (!gradientTexture) return false;

    if (textureRef.current) {
      gl.deleteTexture(textureRef.current);
    }

    textureRef.current = gradientTexture;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, gradientTexture);
    gl.uniform1i(uniformsRef.current.u_gradient, 0);

    return true;
  }, [createGradientTexture]);

  const initWebGL = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return false;

    const gl = canvas.getContext('webgl', {
      antialias: false,
      alpha: true,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      powerPreference: 'default'
    }) as WebGLRenderingContext | null;
    if (!gl) {
      console.error('WebGL not supported');
      return false;
    }

    glRef.current = gl;

    // Vertex shader (simple quad)
    const vertexShaderSource = `
      attribute vec2 a_position;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;

    const isMobile = isMobileDevice();

    // Check if high precision is supported
    const precisionFormat = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
    const supportsHighPrecision = precisionFormat && precisionFormat.precision > 0;

    const { shader: fragmentShaderSource } = createFragmentShader({
      blurAmount: isMobile ? 50 : 150,
      blurQuality: isMobile ? 3 : 6,
      blurExponentRange: [0.8, 1.0]
    });

    // If high precision isn't supported, fallback to mediump
    let finalFragmentShaderSource = fragmentShaderSource;
    if (!supportsHighPrecision) {
      finalFragmentShaderSource = fragmentShaderSource.replace('precision highp float;', 'precision mediump float;');
    }

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, finalFragmentShaderSource);

    if (!vertexShader || !fragmentShader) return false;

    const program = gl.createProgram();
    if (!program) return false;

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program linking error:', gl.getProgramInfoLog(program));
      return false;
    }

    programRef.current = program;
    gl.useProgram(program);

    // Create quad vertices
    const vertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1,
    ]);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const positionLocation = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    // Get uniform locations
    uniformsRef.current = {
      u_time: gl.getUniformLocation(program, 'u_time'),
      u_w: gl.getUniformLocation(program, 'u_w'),
      u_h: gl.getUniformLocation(program, 'u_h'),
      u_gradient: gl.getUniformLocation(program, 'u_gradient'),
      u_velocity: gl.getUniformLocation(program, 'u_velocity'),
      u_fluidStrength: gl.getUniformLocation(program, 'u_fluidStrength'),
    };

    if (!updateGradientTexture()) return false;

    // Try to spin up the fluid simulation. If the platform lacks float
    // render targets we silently fall back to the original quiescent
    // animation by leaving u_velocity bound to a 1x1 zero texture and
    // u_fluidStrength at 0.
    fluidRef.current = FluidSimulation.create(gl);
    if (!fluidRef.current) {
      // Bind a tiny zero-velocity texture to TEXTURE1 so the sampler in
      // the main shader is always valid.
      const zeroTex = gl.createTexture();
      if (zeroTex) {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, zeroTex);
        gl.texImage2D(
          gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
          new Uint8Array([0, 0, 0, 0]),
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      }
    }
    // The fluid sim runs its own programs/buffers; restore the main
    // program/buffer state so subsequent rendering is unaffected.
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(uniformsRef.current.u_velocity, 1);
    gl.uniform1f(uniformsRef.current.u_fluidStrength, 0);

    return true;
  }, [createShader, updateGradientTexture]);

  const resize = useCallback(() => {
    const canvas = canvasRef.current;
    const gl = glRef.current;
    if (!canvas || !gl) return;

    const rect = canvas.getBoundingClientRect();
    const isMobile = isMobileDevice();
    const devicePixelRatio = isMobile ? Math.min(window.devicePixelRatio || 1, 1.5) : (window.devicePixelRatio || 1);

    // Use visual viewport dimensions if available (better for mobile)
    const viewportHeight = window.visualViewport ? window.visualViewport.height : window.innerHeight;
    const viewportWidth = window.visualViewport ? window.visualViewport.width : window.innerWidth;

    // Ensure fullscreen backgrounds use visual viewport dimensions. This
    // avoids stale inline canvas sizes after desktop resizes and mobile URL
    // bar show/hide changes.
    let targetWidth = rect.width;
    let targetHeight = rect.height;

    const isFullscreen = fullscreen || (canvas.parentElement &&
      (canvas.parentElement.classList.contains('h-screen') ||
       canvas.parentElement.classList.contains('min-h-screen') ||
       canvas.parentElement.classList.contains('inset-0') ||
       getComputedStyle(canvas.parentElement).height === '100vh'));

    if (isFullscreen) {
      targetHeight = Math.max(rect.height, viewportHeight);
      targetWidth = Math.max(rect.width, viewportWidth);
    }

    canvas.width = targetWidth * devicePixelRatio;
    canvas.height = targetHeight * devicePixelRatio;

    // Set CSS size to maintain proper display
    canvas.style.width = targetWidth + 'px';
    canvas.style.height = targetHeight + 'px';

    gl.viewport(0, 0, canvas.width, canvas.height);

    // Update size uniforms
    if (uniformsRef.current.u_w) {
      gl.uniform1f(uniformsRef.current.u_w, canvas.width);
    }
    if (uniformsRef.current.u_h) {
      gl.uniform1f(uniformsRef.current.u_h, canvas.height);
    }
    if (fluidRef.current && canvas.height > 0) {
      fluidRef.current.aspect = canvas.width / canvas.height;
    }
  }, [fullscreen]);

  const render = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const canvas = canvasRef.current;
    if (!gl || !program || !canvas) return;

    const now = performance.now();
    const dt = Math.min((now - lastFrameTimeRef.current) / 1000, 1 / 30);
    lastFrameTimeRef.current = now;

    // Step the fluid simulation, then rebind the main program and the
    // viewport that was clobbered by the off-screen passes.
    const fluid = fluidRef.current;
    if (fluid) {
      fluid.step(dt);
      gl.useProgram(program);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, textureRef.current);
      gl.uniform1i(uniformsRef.current.u_gradient, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, fluid.velocityTexture);
      gl.uniform1i(uniformsRef.current.u_velocity, 1);

      // Ease the fluid contribution in once we have the simulation, so the
      // first frame doesn't pop. Capped well below 1 so motion stays subtle.
      fluidStrengthRef.current = Math.min(
        fluidStrengthRef.current + dt * 0.6,
        0.045,
      );
      gl.uniform1f(uniformsRef.current.u_fluidStrength, fluidStrengthRef.current);
    }

    const currentTime = (Date.now() - startTimeRef.current) / 1000;

    // Update time uniform
    if (uniformsRef.current.u_time) {
      gl.uniform1f(uniformsRef.current.u_time, currentTime * 0.5); // Slow down animation
    }

    // Clear with transparent background
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    animationFrameRef.current = requestAnimationFrame(render);
  }, []);

  useEffect(() => {
    if (glRef.current) {
      updateGradientTexture();
    }
  }, [themeRevision, updateGradientTexture]);

  useEffect(() => {
    const handleThemeChange = () => setThemeRevision((revision) => revision + 1);
    const rootElement = document.documentElement;
    const observer = new MutationObserver(handleThemeChange);
    observer.observe(rootElement, { attributes: true, attributeFilter: ['data-theme', 'style'] });

    const colorSchemeQuery = window.matchMedia('(prefers-color-scheme: light)');
    colorSchemeQuery.addEventListener('change', handleThemeChange);

    return () => {
      observer.disconnect();
      colorSchemeQuery.removeEventListener('change', handleThemeChange);
    };
  }, []);

  useEffect(() => {
    if (initWebGL()) {
      resize();
      render();
    }

    const handleResize = () => resize();
    let orientationTimer: ReturnType<typeof window.setTimeout> | null = null;
    const handleOrientationChange = () => {
      if (orientationTimer) window.clearTimeout(orientationTimer);
      orientationTimer = window.setTimeout(resize, 100);
    };

    let viewportTimer: ReturnType<typeof window.setTimeout> | null = null;
    const handleViewportChange = () => {
      if (viewportTimer) window.clearTimeout(viewportTimer);
      viewportTimer = window.setTimeout(resize, 50);
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleOrientationChange);

    // Listen for visual viewport changes (modern mobile browsers)
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', handleViewportChange);
    }

    let scrollTimer: ReturnType<typeof window.setTimeout> | null = null;
    const handleScroll = () => {
      if (scrollTimer) window.clearTimeout(scrollTimer);
      scrollTimer = window.setTimeout(resize, 100);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });

    // Pointer interaction: convert pointer motion into a velocity splat in
    // uv-space and queue it on the fluid sim. The canvas is styled as a
    // passive background (`pointer-events: none`), so listen at the window
    // level and use the canvas bounds to decide whether to react.
    const canvas = canvasRef.current;

    const pointerPos = (event: PointerEvent) => {
      if (!canvas) return null;
      if (
        ignorePointerSelector
        && event.target instanceof Element
        && event.target.closest(ignorePointerSelector)
      ) {
        return null;
      }
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return null;
      const x = (event.clientX - rect.left) / rect.width;
      // WebGL uv has its origin at the bottom-left, DOM has it at the top-left.
      const y = 1 - (event.clientY - rect.top) / rect.height;
      if (x < 0 || x > 1 || y < 0 || y > 1) return null;
      return { x, y };
    };

    const handlePointerDown = (event: PointerEvent) => {
      const pos = pointerPos(event);
      if (!pos) return;
      pointerStateRef.current = {
        active: true,
        lastX: pos.x,
        lastY: pos.y,
        lastT: performance.now(),
      };
    };

    const handlePointerMove = (event: PointerEvent) => {
      const fluid = fluidRef.current;
      if (!fluid) return;
      const pos = pointerPos(event);
      if (!pos) {
        pointerStateRef.current.active = false;
        return;
      }
      const state = pointerStateRef.current;
      const now = performance.now();

      // Hovering (no button) still produces a gentle nudge, but only after
      // we've seen a previous sample to compute a delta from. A press-drag
      // injects a stronger force.
      if (!state.active) {
        state.active = true;
        state.lastX = pos.x;
        state.lastY = pos.y;
        state.lastT = now;
        return;
      }

      const dt = Math.max((now - state.lastT) / 1000, 1 / 240);
      const dx = pos.x - state.lastX;
      const dy = pos.y - state.lastY;
      // Velocity in uv-units / second.
      const vx = dx / dt;
      const vy = dy / dt;
      const speed = Math.hypot(vx, vy);
      const inertia = Math.min(speed * 0.12, 0.55);
      // Scale with pointer speed so fast strokes carry momentum through the
      // velocity field while slow movement stays in the gentle, smoky regime.
      const isPressed = (event.buttons & 1) !== 0 || event.pointerType === 'touch';
      const force = (isPressed ? 0.48 : 0.16) + inertia;

      fluid.addSplat({
        x: pos.x,
        y: pos.y,
        fx: vx * force,
        fy: vy * force,
      });

      state.lastX = pos.x;
      state.lastY = pos.y;
      state.lastT = now;
    };

    const handlePointerLeave = () => {
      pointerStateRef.current.active = false;
    };

    window.addEventListener('pointerdown', handlePointerDown, { passive: true });
    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('pointerup', handlePointerLeave, { passive: true });
    window.addEventListener('pointercancel', handlePointerLeave, { passive: true });
    window.addEventListener('blur', handlePointerLeave);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleOrientationChange);
      if (window.visualViewport) {
        window.visualViewport.removeEventListener('resize', handleViewportChange);
      }
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerLeave);
      window.removeEventListener('pointercancel', handlePointerLeave);
      window.removeEventListener('blur', handlePointerLeave);
      if (orientationTimer) window.clearTimeout(orientationTimer);
      if (viewportTimer) window.clearTimeout(viewportTimer);
      if (scrollTimer) window.clearTimeout(scrollTimer);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (fluidRef.current) {
        fluidRef.current.dispose();
        fluidRef.current = null;
      }
      if (textureRef.current) {
        glRef.current?.deleteTexture(textureRef.current);
        textureRef.current = null;
      }
    };
  }, [ignorePointerSelector, initWebGL, resize, render]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={className}
      style={{ display: 'block' }}
    />
  );
};

export default WebGLGradient;
