export type OrtTensor = {
  dims: number[];
  data: Float32Array;
};

export type OrtInferenceSession = {
  inputNames: string[];
  outputNames: string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>>;
};

export type OrtApi = {
  InferenceSession: {
    create(path: string): Promise<OrtInferenceSession>;
  };
  Tensor: new (
    type: string,
    data: Float32Array,
    dims: number[],
  ) => unknown;
  env: {
    wasm: {
      numThreads: number;
    };
  };
};

declare global {
  interface Window {
    ort?: OrtApi;
  }
}

export function getOrt(): OrtApi | null {
  return window.ort ?? null;
}

export function configureOrt(): OrtApi | null {
  const ort = getOrt();
  if (ort) {
    ort.env.wasm.numThreads = 1;
  }
  return ort;
}
