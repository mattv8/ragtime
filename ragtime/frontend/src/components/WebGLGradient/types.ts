export interface FragmentShaderUniforms {
  [key: string]: any;
}

export interface ShaderOptions {
  blurAmount?: number;
  blurQuality?: number;
  blurExponentRange?: [number, number];
}

export interface ShaderResult {
  shader: string;
  uniforms: FragmentShaderUniforms;
}

export type CreateFragmentShader = (options: ShaderOptions) => ShaderResult;
