/**
 * Creates a deferred promise with externalized resolve/reject functions.
 * Useful for testing async flows and completing promises from test code.
 */
export function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}
