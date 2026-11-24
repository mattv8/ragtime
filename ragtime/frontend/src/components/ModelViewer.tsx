import { ArtifactRequestError, type ModelArtifact } from '@/utils/modelArtifact';
import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { Maximize2, Minimize2, RotateCcw, Scan, X } from 'lucide-react';
import { subscribeToThemeChanges } from '@/theme/themeSnapshot';
import type { Viewer } from 'online-3d-viewer';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';

export interface ModelViewerProps {
  artifact: ModelArtifact;
  onClose?: () => void;
  onError?: (error: unknown) => void;
  height?: number | string;
  showCloseButton?: boolean;
  onExpand?: () => void;
  allowFullscreen?: boolean;
  /** Image shown while the model loads (like a video poster); removed once it is ready or fails. */
  poster?: string;
}

export default function ModelViewer({
  artifact,
  onClose,
  onError,
  height,
  showCloseButton = true,
  onExpand,
  allowFullscreen = true,
  poster,
}: ModelViewerProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const errorCallback = useRef(onError);
  errorCallback.current = onError;
  const initialCamera = useRef<ReturnType<Viewer['GetCamera']>>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [fullscreen, setFullscreen] = useState(false);
  const [interactive, setInteractive] = useState(false);
  const [slow, setSlow] = useState(false);
  const [toolbarIndex, setToolbarIndex] = useState(onExpand || allowFullscreen ? 2 : 0);

  useEffect(() => {
    const canvasElement = canvasRef.current;
    const controller = new AbortController();
    let disposed = false;
    let unsubscribe = () => {};
    let resizeObserver: ResizeObserver | undefined;
    let removeNavigationListeners = () => {};
    let modelLoader: import('online-3d-viewer').ThreeModelLoader | null = null;
    const slowTimer = window.setTimeout(() => setSlow(true), 4000);
    const fail = (cause: unknown) => {
      if (disposed) return;
      setLoading(false);
      setError(cause instanceof Error ? cause.message : 'Unable to load model.');
      errorCallback.current?.(cause);
    };
    const load = async () => {
      try {
        const [ov, response] = await Promise.all([
          import('online-3d-viewer'),
          fetch(`${artifact.content_url}?inline=1`, {
            credentials: 'include',
            signal: controller.signal,
          }),
        ]);
        if (!response.ok) throw new ArtifactRequestError(response.status);
        const blob = await response.blob();
        if (disposed || !canvasRef.current) return;
        const element = canvasRef.current;
        const canvas = document.createElement('canvas');
        element.appendChild(canvas);
        const viewer = new ov.Viewer();
        const listeners: Array<
          [
            string,
            EventListenerOrEventListenerObject,
            boolean | AddEventListenerOptions | undefined,
          ]
        > = [];
        const originalAdd = document.addEventListener;
        document.addEventListener = function (
          type: string,
          listener: EventListenerOrEventListenerObject,
          options?: boolean | AddEventListenerOptions,
        ) {
          if (listener && ['mousemove', 'mouseup', 'mouseleave'].includes(type)) {
            listeners.push([type, listener, options]);
          }
          originalAdd.call(document, type, listener, options);
        };
        removeNavigationListeners = () => {
          for (const [type, listener, options] of listeners) {
            document.removeEventListener(type, listener, options);
          }
        };
        viewerRef.current = viewer;
        try {
          viewer.Init(canvas);
        } finally {
          document.addEventListener = originalAdd;
        }
        if (typeof ResizeObserver !== 'undefined') {
          resizeObserver = new ResizeObserver(() => {
            if (!disposed) viewer.Resize(element.clientWidth, element.clientHeight);
          });
          resizeObserver.observe(element);
        }
        viewer.Resize(element.clientWidth, element.clientHeight);
        const applyTheme = () => {
          const element = canvasRef.current;
          if (!element) return;
          const color = getComputedStyle(element).backgroundColor;
          const components = color.match(/\d+/g)?.map(Number) ?? [255, 255, 255];
          viewer.SetBackgroundColor(
            new ov.RGBAColor(components[0], components[1], components[2], 255),
          );
          const edgeShade = components[0] + components[1] + components[2] < 384 ? 200 : 45;
          viewer.SetEdgeSettings(
            new ov.EdgeSettings(true, new ov.RGBColor(edgeShade, edgeShade, edgeShade), 25),
          );
          viewer.Render();
        };
        applyTheme();
        unsubscribe = subscribeToThemeChanges(applyTheme);
        const loader = new ov.ThreeModelLoader();
        modelLoader = loader;
        loader.LoadModel(
          ov.InputFilesFromFileObjects([new File([blob], `${artifact.id}.${artifact.format}`)]),
          new ov.ImportSettings(),
          {
            onLoadStart: () => {},
            onFileListProgress: () => {},
            onFileLoadProgress: () => {},
            onImportStart: () => {},
            onVisualizationStart: () => {},
            onTextureLoaded: () => {
              if (!disposed) viewer.Render();
            },
            onLoadError: () => fail(new Error('Unable to parse model.')),
            onModelFinished: (_result: unknown, object: unknown) => {
              if (disposed) {
                ov.DisposeThreeObjects(object);
                loader.Destroy();
                return;
              }
              viewer.SetMainObject(object);
              const sphere = viewer.GetBoundingSphere(() => true);
              viewer.AdjustClippingPlanesToSphere(sphere);
              viewer.SetUpVector(ov.Direction.Y, false);
              viewer.FitSphereToWindow(sphere, false);
              initialCamera.current = viewer.GetCamera().Clone();
              setLoading(false);
            },
          },
        );
      } catch (cause) {
        if (!disposed) fail(cause);
      }
    };
    void load();
    return () => {
      disposed = true;
      window.clearTimeout(slowTimer);
      controller.abort();
      unsubscribe();
      resizeObserver?.disconnect();
      removeNavigationListeners();
      modelLoader?.Destroy();
      const viewer = viewerRef.current;
      if (viewer?.renderer) {
        viewer.renderer.forceContextLoss?.();
        viewer.Destroy();
      }
      canvasElement?.replaceChildren();
      viewerRef.current = null;
    };
  }, [artifact.content_url, artifact.id, artifact.format]);

  useEffect(() => {
    const update = () => setFullscreen(document.fullscreenElement === containerRef.current);
    const escape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape' && !document.fullscreenElement) setFullscreen(false);
    };
    document.addEventListener('fullscreenchange', update);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('fullscreenchange', update);
      document.removeEventListener('keydown', escape);
    };
  }, []);

  const fit = () => {
    const viewer = viewerRef.current;
    if (viewer)
      viewer.FitSphereToWindow(
        viewer.GetBoundingSphere(() => true),
        false,
      );
  };
  const reset = () => {
    if (initialCamera.current) viewerRef.current?.SetCamera(initialCamera.current.Clone());
  };
  const handleKey = (event: KeyboardEvent<HTMLDivElement>) => {
    const navigation = viewerRef.current?.navigation;
    if (!navigation) return;
    if (event.key === 'ArrowLeft') navigation.Orbit(-5, 0);
    else if (event.key === 'ArrowRight') navigation.Orbit(5, 0);
    else if (event.key === 'ArrowUp') navigation.Orbit(0, -5);
    else if (event.key === 'ArrowDown') navigation.Orbit(0, 5);
    else if (event.key === '+' || event.key === '=') navigation.Zoom(0.1);
    else if (event.key === '-') navigation.Zoom(-0.1);
    else if (event.key === '0') {
      reset();
      event.preventDefault();
      return;
    } else if (event.key.toLowerCase() === 'f') {
      fit();
      event.preventDefault();
      return;
    } else return;
    navigation.Update();
    event.preventDefault();
  };
  const toggleFullscreen = async () => {
    try {
      if (document.fullscreenElement === containerRef.current) await document.exitFullscreen();
      else if (containerRef.current?.requestFullscreen)
        await containerRef.current.requestFullscreen();
      else setFullscreen((value) => !value);
    } catch {
      setFullscreen((value) => !value);
    }
  };
  const handleToolbarKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const buttons = Array.from(
      event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)'),
    );
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? buttons.length - 1
          : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next]?.focus();
    event.preventDefault();
  };

  return (
    <div
      id={`model-artifact-viewer-${artifact.id}`}
      data-model-artifact-viewer
      className={`model-artifact-viewer${fullscreen ? ' model-artifact-viewer-fullscreen' : ''}`}
      ref={containerRef}
      style={height === undefined ? undefined : { height }}
    >
      <div
        className={`model-artifact-viewer-canvas${interactive ? ' is-interactive' : ''}`}
        ref={canvasRef}
        role="group"
        tabIndex={0}
        aria-label={`Interactive 3D model of ${artifact.file_name}. Drag to rotate, scroll to zoom. Arrow keys rotate, plus and minus zoom, 0 resets.`}
        onKeyDown={handleKey}
        onClick={() => {
          setInteractive(true);
          canvasRef.current?.focus();
        }}
        onFocus={() => setInteractive(true)}
        onBlur={() => setInteractive(false)}
      />
      {loading && !error && poster && (
        <img className="model-artifact-viewer-poster" src={poster} alt="" aria-hidden="true" />
      )}
      {loading && (
        <div className="model-artifact-status" role="status">
          <MiniLoadingSpinner ariaHidden />
          {artifact.format.toLowerCase() === 'step'
            ? 'Loading and parsing STEP…'
            : 'Loading 3D view…'}
          {(slow || artifact.format.toLowerCase() === 'step') &&
            ' This can take a while for big assemblies.'}
        </div>
      )}
      {error && (
        <div role="alert" className="model-artifact-error">
          {error}
        </div>
      )}
      <div
        className="model-artifact-toolbar"
        role="toolbar"
        aria-label="3D view controls"
        data-model-artifact-toolbar
        onKeyDown={handleToolbarKey}
      >
        <button
          type="button"
          className="model-artifact-tool-btn"
          data-action="reset-view"
          aria-label="Reset view"
          disabled={loading || Boolean(error)}
          tabIndex={toolbarIndex === 0 && !loading && !error ? 0 : -1}
          onFocus={() => setToolbarIndex(0)}
          onClick={reset}
        >
          <RotateCcw size={16} />
        </button>
        <button
          type="button"
          className="model-artifact-tool-btn"
          data-action="fit"
          aria-label="Fit to model"
          disabled={loading || Boolean(error)}
          tabIndex={toolbarIndex === 1 && !loading && !error ? 0 : -1}
          onFocus={() => setToolbarIndex(1)}
          onClick={fit}
        >
          <Scan size={16} />
        </button>
        {(onExpand || allowFullscreen) && (
          <button
            type="button"
            className="model-artifact-tool-btn"
            data-action={onExpand ? 'expand' : 'fullscreen'}
            aria-label={onExpand ? 'Open large view' : 'Fullscreen'}
            aria-pressed={onExpand ? undefined : fullscreen}
            tabIndex={toolbarIndex === 2 ? 0 : -1}
            onFocus={() => setToolbarIndex(2)}
            onClick={() => (onExpand ? onExpand() : void toggleFullscreen())}
          >
            {fullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        )}
        {showCloseButton && onClose && (
          <button
            type="button"
            className="model-artifact-tool-btn"
            data-action="close-3d"
            aria-label="Close 3D view"
            tabIndex={toolbarIndex === 3 ? 0 : -1}
            onFocus={() => setToolbarIndex(3)}
            onClick={onClose}
          >
            <X size={16} />
          </button>
        )}
      </div>
    </div>
  );
}
