import { simplex_noise } from './utils/simplexNoise';
import type { CreateFragmentShader, FragmentShaderUniforms } from './types';

const createFragmentShader: CreateFragmentShader = (options) => {
  const {
    blurAmount = 345,
    blurQuality = 7,
    blurExponentRange = [0.9, 1.2],
  } = options;

  const uniforms: FragmentShaderUniforms = {};

  const shader = /* glsl */ `
    #ifdef GL_ES
    precision highp float;
    #endif

    uniform float u_time;
    uniform float u_h;
    uniform float u_w;
    uniform sampler2D u_gradient;
    uniform sampler2D u_velocity;
    uniform float u_fluidStrength;

    const float PI = 3.14159;

    ${simplex_noise}

    // Per-fragment displacement (in pixels) sourced from the fluid sim.
    // Set in main() before calling any of the noise helpers below so they
    // pick up the warped coordinate consistently.
    vec2 g_disp;

    float get_x() {
      return 900.0 + gl_FragCoord.x + g_disp.x - u_w / 2.0;
    }

    float get_y() {
      return gl_FragCoord.y + g_disp.y;
    }

    float smoothstep_custom(float t) {
      return t * t * t * (t * (6.0 * t - 15.0) + 10.0);
    }

    float lerp(float a, float b, float t) {
      return a * (1.0 - t) + b * t;
    }

    float ease_in(float x) {
      return 1.0 - cos((x * PI) * 0.5);
    }

    float wave_alpha_part(float dist, float blur_fac, float t) {
      float exp = mix(${blurExponentRange[0].toFixed(5)}, ${blurExponentRange[1].toFixed(5)}, t);
      float v = pow(blur_fac, exp);
      v = ease_in(v);
      v = smoothstep_custom(v);
      v = clamp(v, 0.008, 1.0);
      v *= ${blurAmount.toFixed(1)};
      float alpha = clamp(0.5 + dist / v, 0.0, 1.0);
      alpha = smoothstep_custom(alpha);
      return alpha;
    }

    float background_noise(float offset) {
      const float S = 0.064;
      const float L = 0.00085;
      const float L1 = 1.5, L2 = 0.9, L3 = 0.6;
      const float LY1 = 1.00, LY2 = 0.85, LY3 = 0.70;
      const float F = 0.04;
      const float Y_SCALE = 1.0 / 0.27;

      float x = get_x() * L;
      float y = get_y() * L * Y_SCALE;
      float time = u_time + offset;
      float x_shift = time * F;
      float sum = 0.5;
      sum += simplex_noise(vec3(x * L1 +  x_shift * 1.1, y * L1 * LY1, time * S)) * 0.30;
      sum += simplex_noise(vec3(x * L2 + -x_shift * 0.6, y * L2 * LY2, time * S)) * 0.25;
      sum += simplex_noise(vec3(x * L3 +  x_shift * 0.8, y * L3 * LY3, time * S)) * 0.20;
      return sum;
    }

    float wave_y_noise(float offset) {
      const float L = 0.000845;
      const float S = 0.075;
      const float F = 0.026;

      float time = u_time + offset;
      float x = get_x() * 0.000845;
      float y = time * S;
      float x_shift = time * 0.026;

      float sum = 0.0;
      sum += simplex_noise(vec2(x * 1.30 + x_shift, y * 0.54)) * 0.85;
      sum += simplex_noise(vec2(x * 1.00 + x_shift, y * 0.68)) * 1.15;
      sum += simplex_noise(vec2(x * 0.70 + x_shift, y * 0.59)) * 0.60;
      sum += simplex_noise(vec2(x * 0.40 + x_shift, y * 0.48)) * 0.40;
      return sum;
    }

    float calc_blur_bias() {
      const float S = 0.261;
      float bias_t = (sin(u_time * S) + 1.0) * 0.5;
      return lerp(-0.17, -0.04, bias_t);
    }

    float calc_blur(float offset) {
      const float L = 0.0011;
      const float S = 0.07;
      const float F = 0.03;

      float time = u_time + offset;

      float x = get_x() * L;
      float blur_fac = calc_blur_bias();
      blur_fac += simplex_noise(vec2(x * 0.60 + time * F *  1.0, time * S * 0.7)) * 0.5;
      blur_fac += simplex_noise(vec2(x * 1.30 + time * F * -0.8, time * S * 1.0)) * 0.4;
      blur_fac = (blur_fac + 1.0) * 0.5;
      blur_fac = clamp(blur_fac, 0.0, 1.0);
      return blur_fac;
    }

    float wave_alpha(float Y, float wave_height, float offset) {
      float wave_y = Y + wave_y_noise(offset) * wave_height;
      float dist = wave_y - get_y();
      float blur_fac = calc_blur(offset);

      const float PART = 1.0 / float(${blurQuality});
      float sum = 0.0;
      for (int i = 0; i < ${blurQuality}; i++) {
        float t = ${blurQuality} == 1 ? 0.5 : PART * float(i);
        sum += wave_alpha_part(dist, blur_fac, t) * PART;
      }
      return sum;
    }

    vec3 calc_color(float lightness) {
      lightness = clamp(lightness, 0.0, 1.0);
      return vec3(texture2D(u_gradient, vec2(lightness, 0.5)));
    }

    float edge_mask(float alpha) {
      // Peaks where alpha is around 0.5, which is the high-contrast boundary
      // between the blended sine-wave layers.
      float edge = alpha * (1.0 - alpha) * 4.0;
      return smoothstep(0.18, 0.92, edge);
    }

    void main() {
      // Sample the fluid velocity field and convert to a pixel-space
      // displacement. The strength uniform lets the JS side fade the effect
      // in/out (e.g. when no pointer interaction has happened yet).
      vec2 uv = gl_FragCoord.xy / vec2(u_w, u_h);
      vec2 vel = texture2D(u_velocity, uv).xy;
      g_disp = -vel * vec2(u_w, u_h) * u_fluidStrength;

      float WAVE1_Y = 0.45 * u_h;
      float WAVE2_Y = 0.9 * u_h;
      float WAVE1_HEIGHT = 0.195 * u_h;
      float WAVE2_HEIGHT = 0.144 * u_h;

      float bg_lightness = background_noise(-192.4);
      float w1_lightness = background_noise( 273.3);
      float w2_lightness = background_noise( 623.1);

      float w1_alpha = wave_alpha(WAVE1_Y, WAVE1_HEIGHT, 112.5 * 48.75);
      float w2_alpha = wave_alpha(WAVE2_Y, WAVE2_HEIGHT, 225.0 * 36.00);

      float boundary_strength = max(edge_mask(w1_alpha), edge_mask(w2_alpha));
      float velocity_strength = smoothstep(0.001, 0.075, length(vel));
      float smear_strength = boundary_strength * velocity_strength;

      // Apply a second, boundary-only displacement pass. This leaves quiet
      // areas mostly untouched while making the visible wave transition bands
      // stretch and drag with pointer inertia.
      g_disp -= vel * vec2(u_w, u_h) * u_fluidStrength * smear_strength * 2.6;
      if (smear_strength > 0.001) {
        w1_alpha = wave_alpha(WAVE1_Y, WAVE1_HEIGHT, 112.5 * 48.75);
        w2_alpha = wave_alpha(WAVE2_Y, WAVE2_HEIGHT, 225.0 * 36.00);
      }

      float lightness = bg_lightness;
      lightness = lerp(lightness, w2_lightness, w2_alpha);
      lightness = lerp(lightness, w1_lightness, w1_alpha);

      // Add dither to eliminate banding
      float d = (simplex_noise(vec2(gl_FragCoord.xy)) - 0.5) * (1.0/1024.0);
      lightness = clamp(lightness + d, 0.0, 1.0);

      gl_FragColor = vec4(calc_color(lightness), 1.0);
    }
  `;
  return { shader, uniforms };
};

export default createFragmentShader;
