import { useState, useEffect, useRef, useCallback, useMemo, memo, isValidElement, type ReactNode, type CSSProperties } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { diffLines } from 'diff';
import { Copy, Check, Pencil, Slash, Trash2, Maximize2, Minimize2, X, AlertCircle, RefreshCw, Play, FileText, Bug, ChevronDown, ChevronRight, ChevronLeft, Bot, MessageSquare, MessageSquarePlus, BrainCircuit, Clock, Diff, Wrench, Database, Search, Terminal, BarChart3, Globe, Code, FolderSearch, Image as ImageIcon, Link } from 'lucide-react';
import { api } from '@/api';
import type { Conversation, ChatMessage, ChatTask, User, ContentPart, ConversationMember, UserSpaceAvailableTool, ProviderPromptDebugRecord, MessageEvent, ProviderModelState, WorkspaceChatStateResponse, LlmProviderWire, UserSpaceFile, UserSpaceSnapshotFileDiff, ConversationBranchPointInfo } from '@/types';
import { FileAttachment, attachmentsToContentParts, formatAttachmentSize, resizeAttachmentImageDataUrl, type AttachmentFile } from './FileAttachment';
import { ModelSelector } from './ModelSelector';
import { ResizeHandle } from './ResizeHandle';
import { calculateConversationContextUsage, parseStoredModelIdentifier } from '@/utils/contextUsage';
import {
  formatChatTimestamp,
  getCookieValue,
  setSessionCookieValue,
  clampNumber,
  isInterruptDismissed,
  dismissInterruptAlert,
  clearInterruptDismiss,
  resolveInterruptDismissTransition,
} from '@/utils';
import type { InterruptChatStateSnapshot } from '@/utils/cookies';
import { ContextUsagePie } from './shared/ContextUsagePie';
import { FileDiffOverlay } from './shared/FileDiffOverlay';
import { MemberManagementButton } from './shared/MemberManagementButton';
import { MemberManagementModal } from './shared/MemberManagementModal';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';
import { ToolSelectorDropdown, type ToolGroupInfo } from './shared/ToolSelectorDropdown';
import { UserSpaceFileDiffView, formatDiffStatus } from './shared/UserSpaceFileDiffView';
import { useAvailableModels } from '@/contexts/AvailableModelsContext';

interface CodeBlockProps {
  inline?: boolean;
  className?: string;
  children?: ReactNode | ReactNode[];
}

// Renders markdown code blocks with inset styling and copy support
function CodeBlock({ className, children }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const codeText = useMemo(() => {
    if (!children) return '';
    const raw = Array.isArray(children) ? children.join('') : String(children);
    return raw.replace(/\n$/, '');
  }, [children]);

  const language = useMemo(() => {
    if (!className) return 'text';
    return className.replace('language-', '') || 'text';
  }, [className]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(codeText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code block:', err);
    }
  }, [codeText]);

  return (
    <div className="markdown-codeblock">
      <div className="markdown-codeblock-header">
        <span className="markdown-codeblock-lang">{language}</span>
        <button
          type="button"
          className={`markdown-codeblock-copy ${copied ? 'is-copied' : ''}`}
          onClick={handleCopy}
          aria-label={copied ? 'Code copied' : 'Copy code'}
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre className="markdown-codeblock-pre">
        <code className={className}>{codeText}</code>
      </pre>
    </div>
  );
}

const PreBlock = ({ children, ...rest }: React.HTMLAttributes<HTMLPreElement> & { children?: ReactNode }) => {
  if (isValidElement<{ className?: string; children?: ReactNode }>(children) && children.type === 'code') {
    return <CodeBlock {...children.props} />;
  }
  return <pre {...rest}>{children}</pre>;
};

const markdownComponents: Components = {
  pre: PreBlock,
};

// Memoized markdown component to prevent re-parsing on every render
// Only re-renders when content actually changes
const MemoizedMarkdown = memo(function MemoizedMarkdown({ content }: { content: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{content}</ReactMarkdown>;
});

// Tool call info for display during streaming
interface ActiveToolCall {
  tool: string;
  input?: Record<string, unknown>;
  output?: string;
  presentation?: {
    kind?: string;
    rerun_kind?: string;
  };
  connection?: {
    tool_config_id: string;
    tool_config_name?: string;
    tool_type?: string;
    connection_mode?: string;
  };
  status: 'running' | 'complete';
  generating_lines?: number;
}

// Local render event to keep streaming items in arrival order
type StreamingRenderEvent =
  | { type: 'content'; content: string }
  | { type: 'tool'; toolCall: ActiveToolCall }
  | { type: 'reasoning'; content: string; durationSeconds?: number };

// Parse table metadata from SQL tool output
// Format: <!--TABLEDATA:{"columns":[...],"rows":[...]}-->
interface TableData {
  columns: string[];
  rows: (string | number | null)[][];
}

function parseTableMetadata(output: string): { tableData: TableData | null; displayText: string } {
  // Match from <!--TABLEDATA: to --> but use greedy .* since JSON may contain }
  // The --> acts as a reliable terminator
  const startMarker = '<!--TABLEDATA:';
  const endMarker = '-->';
  const startIdx = output.indexOf(startMarker);
  if (startIdx === -1) {
    return { tableData: null, displayText: output };
  }

  const jsonStart = startIdx + startMarker.length;
  const endIdx = output.indexOf(endMarker, jsonStart);
  if (endIdx === -1) {
    return { tableData: null, displayText: output };
  }

  const jsonStr = output.substring(jsonStart, endIdx);

  try {
    const tableData = JSON.parse(jsonStr) as TableData;
    // Remove the metadata from display text
    const fullMarker = output.substring(startIdx, endIdx + endMarker.length);
    const displayText = output.replace(fullMarker, '').replace(/^\n/, '');
    return { tableData, displayText };
  } catch {
    return { tableData: null, displayText: output };
  }
}

// Parse chart data from create_chart tool output
interface ChartConfig {
  type: 'bar' | 'line' | 'pie' | 'doughnut' | 'scatter' | 'radar' | 'polarArea';
  data: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      backgroundColor?: string | string[];
      borderColor?: string;
      borderWidth?: number;
    }[];
  };
  options?: Record<string, unknown>;
}

// Global URL regex for efficient linkification
const URL_PATTERN = /(https?:\/\/[^\s<]+[^<.,:;"')\]\s])/g;

const KNOWN_PROVIDER_KEYS = new Set(['openai', 'anthropic', 'ollama', 'github_copilot', 'github_models']);

function normalizeProviderAlias(provider?: string | null): string {
  const value = (provider || '').trim().toLowerCase();
  return value === 'github_models' ? 'github_copilot' : value;
}

function providersEquivalent(selected?: string | null, actual?: string | null): boolean {
  const selectedNorm = normalizeProviderAlias(selected);
  const actualNorm = normalizeProviderAlias(actual);
  if (!selectedNorm || !actualNorm) {
    return false;
  }
  return selectedNorm === actualNorm;
}

function inferProviderFromModelId(modelId: string): string | null {
  const raw = (modelId || '').trim();
  const slashIndex = raw.indexOf('/');
  if (slashIndex <= 0) {
    return null;
  }
  const maybeProvider = normalizeProviderAlias(raw.slice(0, slashIndex));
  return KNOWN_PROVIDER_KEYS.has(maybeProvider) ? maybeProvider : null;
}

function toProviderScopedModelKey(provider: string | null | undefined, modelId: string): string {
  const normalizedProvider = normalizeProviderAlias(provider);
  return normalizedProvider ? `${normalizedProvider}::${modelId}` : modelId;
}

function conversationUpdatedAtMs(conversation: Conversation | null | undefined): number {
  const parsed = Date.parse(conversation?.updated_at || '');
  return Number.isFinite(parsed) ? parsed : 0;
}

function mergeConversationFromWorkspaceSnapshot(
  current: Conversation,
  incoming: Conversation,
): Conversation {
  const incomingUpdatedAt = conversationUpdatedAtMs(incoming);
  const currentUpdatedAt = conversationUpdatedAtMs(current);
  const incomingIsNewer = incomingUpdatedAt > currentUpdatedAt;
  const shouldUseIncomingMessages = incomingIsNewer
    || incoming.messages.length > current.messages.length
    || (incoming.messages.length === current.messages.length && incomingUpdatedAt >= currentUpdatedAt);

  return {
    ...current,
    ...incoming,
    model: incomingIsNewer ? (incoming.model || current.model) : (current.model || incoming.model),
    // Preserve optimistic local messages when the workspace snapshot has the same timestamp.
    messages: shouldUseIncomingMessages
      ? incoming.messages
      : current.messages,
  };
}

type BranchLookup = {
  branch_point_index: number;
  parent_branch_id?: string | null;
};

function findLineageBranchIdForPoint(
  branchesById: ReadonlyMap<string, BranchLookup>,
  activeBranchId: string | null | undefined,
  branchPointIndex: number,
): string | null {
  if (!activeBranchId) return null;

  const visited = new Set<string>();
  let currentBranchId: string | null = activeBranchId;
  while (currentBranchId && !visited.has(currentBranchId)) {
    visited.add(currentBranchId);
    const branch = branchesById.get(currentBranchId);
    if (!branch) return null;
    if (branch.branch_point_index === branchPointIndex) {
      return currentBranchId;
    }
    currentBranchId = branch.parent_branch_id ?? null;
  }

  return null;
}

function findProviderState(
  providerStates: ProviderModelState[] | undefined,
  provider: string | null,
): ProviderModelState | null {
  const target = normalizeProviderAlias(provider);
  if (!target || !providerStates?.length) {
    return null;
  }
  return (
    providerStates.find((state) => providersEquivalent(state.provider, target)) || null
  );
}

// Helper component to parse URLs and render them as clickable links
const LinkifiedText = memo(function LinkifiedText({ text }: { text: string }) {
  if (typeof text !== 'string') return <>{text}</>;

  const parts = text.split(URL_PATTERN);
  if (parts.length === 1) {
    return <>{text}</>;
  }

  return (
    <>
      {parts.map((part, i) => {
        if (part.match(URL_PATTERN)) {
          return (
            <a
              key={i}
              href={part}
              target="_blank"
              rel="noopener noreferrer"
              className="chat-link"
              onClick={(e) => e.stopPropagation()}
            >
              {part}
            </a>
          );
        }
        return part;
      })}
    </>
  );
});

interface ChartData {
  __chart__: true;
  config: ChartConfig;
  description?: string;
  data_connection?: {
    source_tool?: string;
    source_tool_config_id?: string;
    source_tool_type?: string;
    source_input?: Record<string, unknown>;
    refresh_interval_seconds?: number;
  };
}

// Parse datatable data from create_datatable tool output
interface DataTableConfig {
  columns: { title: string }[];
  data: unknown[][];
  pageLength?: number;
  searching?: boolean;
  ordering?: boolean;
  paging?: boolean;
  info?: boolean;
  [key: string]: unknown;  // Allow additional DataTables options
}

interface DataTableData {
  __datatable__: true;
  title: string;
  config: DataTableConfig;
  description?: string;
  data_connection?: {
    source_tool?: string;
    source_tool_config_id?: string;
    source_tool_type?: string;
    source_input?: Record<string, unknown>;
    refresh_interval_seconds?: number;
  };
}

interface StoredChatLayout {
  showSidebar: boolean;
  sidebarWidth: number;
  inputAreaHeight: number;
  isInputAreaCollapsed: boolean;
  isMessagesCollapsed: boolean;
}

const CHAT_LAYOUT_COOKIE_PREFIX = 'chat_layout_';

function getChatLayoutCookieName(userId: string): string {
  return `${CHAT_LAYOUT_COOKIE_PREFIX}${encodeURIComponent(userId)}`;
}

function readStoredChatLayout(cookieName: string): StoredChatLayout | null {
  const raw = getCookieValue(cookieName);
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as Partial<StoredChatLayout>;
    return {
      showSidebar: parsed.showSidebar ?? true,
      sidebarWidth: typeof parsed.sidebarWidth === 'number' ? parsed.sidebarWidth : 280,
      inputAreaHeight: typeof parsed.inputAreaHeight === 'number' ? parsed.inputAreaHeight : 96,
      isInputAreaCollapsed: Boolean(parsed.isInputAreaCollapsed),
      isMessagesCollapsed: Boolean(parsed.isMessagesCollapsed),
    };
  } catch {
    return null;
  }
}

function parseChartData(output: string): ChartData | null {
  try {
    const parsed = JSON.parse(output);
    if (parsed && parsed.__chart__ === true && parsed.config) {
      return parsed as ChartData;
    }
  } catch {
    // Not JSON or not chart data
  }
  return null;
}

function parseDataTableData(output: string): DataTableData | null {
  try {
    const parsed = JSON.parse(output);
    if (parsed && parsed.__datatable__ === true && parsed.config) {
      return parsed as DataTableData;
    }
  } catch (e) {
    // Failed to parse JSON as datatable data
  }
  return null;
}

// Component to render a simple data table (for SQL results with table metadata)
const DataTable = memo(function DataTable({ data }: { data: TableData }) {
  return (
    <div className="tool-result-table-wrapper">
      <table className="tool-result-table">
        <thead>
          <tr>
            {data.columns.map((col, i) => (
              <th key={i}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, rowIdx) => (
            <tr key={rowIdx}>
              {row.map((cell, cellIdx) => (
                <td key={cellIdx}>
                  {cell === null ? (
                    <span className="null-value">NULL</span>
                  ) : typeof cell === 'string' ? (
                    <LinkifiedText text={cell} />
                  ) : (
                    String(cell)
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="tool-result-row-count">({data.rows.length} row{data.rows.length !== 1 ? 's' : ''})</div>
    </div>
  );
});

// Component to render a Chart.js chart
const ChartDisplay = memo(function ChartDisplay({ chartData }: { chartData: ChartData }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<unknown>(null);
  const [resizeKey, setResizeKey] = useState(0);

  // Get theme colors from CSS variables
  const getThemeColors = useCallback(() => {
    const root = document.documentElement;
    const styles = getComputedStyle(root);
    return {
      textColor: styles.getPropertyValue('--color-text-secondary').trim() || '#9ca3af',
      textMuted: styles.getPropertyValue('--color-text-muted').trim() || '#6b7280',
      gridColor: styles.getPropertyValue('--color-border').trim() || '#374151',
    };
  }, []);

  const createChart = useCallback(() => {
    try {
      // @ts-expect-error Chart.js is loaded from CDN
      const Chart = window.Chart;
      if (!Chart) {
        console.error('Chart.js not loaded');
        return;
      }

      if (canvasRef.current && containerRef.current) {
        // Destroy previous chart if exists
        if (chartInstanceRef.current) {
          (chartInstanceRef.current as { destroy: () => void }).destroy();
        }

        const colors = getThemeColors();
        const dpr = window.devicePixelRatio || 1;

        // Calculate display size
        const containerWidth = containerRef.current.clientWidth - 32;
        const displayHeight = Math.min(350, containerWidth * 0.6);

        // Set canvas backing store size (scaled for DPI)
        canvasRef.current.width = containerWidth * dpr;
        canvasRef.current.height = displayHeight * dpr;

        // Set CSS display size
        canvasRef.current.style.width = containerWidth + 'px';
        canvasRef.current.style.height = displayHeight + 'px';

        // Theme-aware options for axes and legends
        const themeOptions = {
          color: colors.textColor,
          plugins: {
            legend: {
              labels: {
                color: colors.textColor,
              },
            },
            title: {
              color: colors.textColor,
            },
          },
          scales: {
            x: {
              ticks: { color: colors.textColor },
              grid: { color: colors.gridColor },
            },
            y: {
              ticks: { color: colors.textColor },
              grid: { color: colors.gridColor },
            },
          },
        };

        // Deep merge theme options with user config
        const config = {
          ...chartData.config,
          options: {
            responsive: false,
            maintainAspectRatio: false,
            devicePixelRatio: dpr,
            animation: { duration: 400 },
            ...themeOptions,
            ...chartData.config.options,
            plugins: {
              ...Object(themeOptions.plugins),
              ...Object(chartData.config.options?.plugins),
            },
            scales: {
              ...Object(themeOptions.scales),
              ...Object(chartData.config.options?.scales),
            },
          },
        };

        chartInstanceRef.current = new Chart(canvasRef.current, config);
      }
    } catch (err) {
      console.error('Failed to create chart:', err);
    }
  }, [chartData, getThemeColors]);

  useEffect(() => {
    // Small delay to ensure container is sized
    const timeoutId = setTimeout(createChart, 50);

    return () => {
      clearTimeout(timeoutId);
      if (chartInstanceRef.current) {
        (chartInstanceRef.current as { destroy: () => void }).destroy();
      }
    };
  }, [chartData, resizeKey, createChart]);

  const handleResize = () => {
    setResizeKey(k => k + 1);
  };

  return (
    <div className="chart-container" ref={containerRef}>
      <button
        className="chart-resize-btn"
        onClick={handleResize}
        title="Resize chart"
      >
        <Maximize2 size={14} />
      </button>
      <canvas ref={canvasRef}></canvas>
      {chartData.description && (
        <p className="chart-description">
          <LinkifiedText text={chartData.description} />
        </p>
      )}
    </div>
  );
});

// Component to render an interactive DataTable using DataTables.js
const DataTableDisplay = memo(function DataTableDisplay({ tableData }: { tableData: DataTableData }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const tableInstanceRef = useRef<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Small delay to ensure DOM is ready
    const timeoutId = setTimeout(() => {
      try {
        // @ts-expect-error jQuery is loaded from CDN
        const $ = window.jQuery;
        if (!$) {
          setError('jQuery not loaded');
          return;
        }
        if (!$.fn.DataTable) {
          setError('DataTables.js not loaded');
          return;
        }

        if (containerRef.current) {
          // Find or create the table element
          let tableEl = containerRef.current.querySelector('table');
          if (!tableEl) {
            tableEl = document.createElement('table');
            tableEl.className = 'display';
            tableEl.style.width = '100%';
            containerRef.current.querySelector('.datatable-table-wrapper')?.appendChild(tableEl);
          }

          // Destroy previous instance if exists
          if (tableInstanceRef.current) {
            try {
              (tableInstanceRef.current as { destroy: () => void }).destroy();
            } catch {
              // Ignore destroy errors
            }
          }

          // Prepare columns with linkification support
          const existingColumns = tableData.config.columns || [];
          const preparedColumns = existingColumns.map((col: any) => {
            const existingRender = col.render;
            return {
              ...col,
              render: (data: any, type: string, row: any, meta: any) => {
                let val = data;
                // Call existing renderer if it exists
                if (typeof existingRender === 'function') {
                  val = existingRender(data, type, row, meta);
                } else if (typeof existingRender === 'string' && (window as any).$.fn.dataTable.render[existingRender]) {
                  // Handle built-in renderers if they are passed as strings (rare but possible)
                  val = (window as any).$.fn.dataTable.render[existingRender]()(data, type, row, meta);
                }

                if (type === 'display' && typeof val === 'string' && val.match(URL_PATTERN)) {
                  // Replace URLs with clickable <a> tags
                  return val.replace(URL_PATTERN, '<a href="$1" target="_blank" rel="noopener noreferrer" class="chat-link">$1</a>');
                }
                return val;
              }
            };
          });

          // Initialize DataTable
          tableInstanceRef.current = $(tableEl).DataTable({
            ...tableData.config,
            columns: preparedColumns,
            destroy: true, // Allow re-initialization
            columnDefs: [
              ...(Array.isArray((tableData.config as any).columnDefs) ? (tableData.config as any).columnDefs : [])
            ]
          });
        }
      } catch (err) {
        console.error('Failed to create datatable:', err);
        setError(String(err));
      }
    }, 100);

    return () => {
      clearTimeout(timeoutId);
      if (tableInstanceRef.current) {
        try {
          (tableInstanceRef.current as { destroy: () => void }).destroy();
        } catch {
          // Ignore destroy errors
        }
      }
    };
  }, [tableData]);

  if (error) {
    // Fallback to simple table if DataTables fails
    return (
      <div className="datatable-container">
        {tableData.title && <h4 className="datatable-title">{tableData.title}</h4>}
        <div className="tool-result-table-wrapper">
          <table className="tool-result-table">
            <thead>
              <tr>
                {tableData.config.columns.map((col, i) => (
                  <th key={i}>{typeof col === 'string' ? col : col.title}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableData.config.data.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  {(row as unknown[]).map((cell, cellIdx) => (
                    <td key={cellIdx}>
                      {cell === null ? (
                        'NULL'
                      ) : typeof cell === 'string' ? (
                        <LinkifiedText text={cell} />
                      ) : (
                        String(cell)
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <div className="datatable-container" ref={containerRef}>
      {tableData.title && <h4 className="datatable-title">{tableData.title}</h4>}
      <div className="datatable-table-wrapper">
        <table className="display" style={{ width: '100%' }}></table>
      </div>
      {tableData.description && (
        <p className="datatable-description">{tableData.description}</p>
      )}
    </div>
  );
});

interface UserSpaceWriteToolPayload {
  status?: string;
  path?: string;
  message?: string;
  error?: string;
  action_required?: string;
  persisted?: boolean;
  rejected?: boolean;
  write_signature?: string;
  contract_violations?: string[];
  warnings?: string[];
  file?: UserSpaceFile;
}

interface ParsedUserSpaceWriteResult {
  toolName: 'upsert_userspace_file' | 'patch_userspace_file';
  status: string;
  path: string;
  message: string;
  error: string | null;
  actionRequired: string | null;
  writeSignature: string | null;
  persisted: boolean;
  rejected: boolean;
  noChanges: boolean;
  file: UserSpaceFile | null;
  details: string[];
}

const USERSPACE_WRITE_TOOL_NAMES = new Set(['upsert_userspace_file', 'patch_userspace_file']);
const USERSPACE_WRITE_DIFF_CACHE_MAX_ENTRIES = 100;
const userspaceWriteDiffCache = new Map<string, UserSpaceSnapshotFileDiff>();

function trimUserspaceWriteDiffCache(): void {
  while (userspaceWriteDiffCache.size > USERSPACE_WRITE_DIFF_CACHE_MAX_ENTRIES) {
    const oldestKey = userspaceWriteDiffCache.keys().next().value;
    if (oldestKey === undefined) {
      break;
    }
    userspaceWriteDiffCache.delete(oldestKey);
  }
}

function parseUserspaceWriteToolResult(toolName: string, output?: string | null): ParsedUserSpaceWriteResult | null {
  if (!output || !USERSPACE_WRITE_TOOL_NAMES.has(toolName)) {
    return null;
  }

  try {
    const parsed = JSON.parse(output) as UserSpaceWriteToolPayload;
    if (!parsed || typeof parsed !== 'object') {
      return null;
    }

    const status = typeof parsed.status === 'string' ? parsed.status : '';
    const file = parsed.file && typeof parsed.file === 'object' ? parsed.file : null;
    const path = typeof parsed.path === 'string' && parsed.path.trim()
      ? parsed.path.trim()
      : typeof file?.path === 'string' && file.path.trim()
        ? file.path.trim()
        : '';
    if (!path) {
      return null;
    }

    return {
      toolName: toolName as 'upsert_userspace_file' | 'patch_userspace_file',
      status,
      path,
      message: typeof parsed.message === 'string' ? parsed.message.trim() : '',
      error: typeof parsed.error === 'string' && parsed.error.trim() ? parsed.error.trim() : null,
      actionRequired: typeof parsed.action_required === 'string' && parsed.action_required.trim()
        ? parsed.action_required.trim()
        : null,
      writeSignature: typeof parsed.write_signature === 'string' && parsed.write_signature.trim()
        ? parsed.write_signature.trim()
        : null,
      persisted: parsed.persisted === true || status.startsWith('persisted'),
      rejected: parsed.rejected === true || status.includes('rejected'),
      noChanges: status === 'no_changes',
      file,
      details: [
        ...(Array.isArray(parsed.contract_violations)
          ? parsed.contract_violations.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
          : []),
        ...(Array.isArray(parsed.warnings)
          ? parsed.warnings.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
          : []),
      ],
    };
  } catch {
    // JSON.parse failed — output is likely truncated (> 2000 chars).
    // Extract key fields via regex so the write summary + diff still render.
    return parseUserspaceWriteToolResultFromTruncated(toolName, output);
  }
}

/** Regex fallback for truncated (invalid JSON) tool output. */
function parseUserspaceWriteToolResultFromTruncated(
  toolName: string,
  output: string,
): ParsedUserSpaceWriteResult | null {
  const str = (pattern: RegExp): string => {
    const m = output.match(pattern);
    return m ? m[1] : '';
  };
  const bool = (pattern: RegExp): boolean => pattern.test(output);

  const path = str(/"path"\s*:\s*"([^"]+)"/);
  if (!path) return null;

  const status = str(/"status"\s*:\s*"([^"]+)"/);
  return {
    toolName: toolName as 'upsert_userspace_file' | 'patch_userspace_file',
    status,
    path,
    message: str(/"message"\s*:\s*"([^"]*)"/) || '',
    error: null,
    actionRequired: null,
    writeSignature: str(/"write_signature"\s*:\s*"([^"]+)"/) || null,
    persisted: bool(/"persisted"\s*:\s*true/) || status.startsWith('persisted'),
    rejected: bool(/"rejected"\s*:\s*true/) || status.includes('rejected'),
    noChanges: status === 'no_changes',
    file: null,
    details: [],
  };
}

function formatUserspaceWriteSummary(result: ParsedUserSpaceWriteResult): string {
  const lines: string[] = [result.message || `${result.toolName} completed for ${result.path}.`];
  if (result.error) {
    lines.push(`Error: ${result.error}`);
  }
  if (result.actionRequired) {
    lines.push(`Action required: ${result.actionRequired}`);
  }
  if (result.details.length > 0) {
    lines.push('', ...result.details.map((detail) => `- ${detail}`));
  }
  return lines.filter(Boolean).join('\n');
}

function buildUserspaceToolDiffCacheKey(snapshotId: string, result: ParsedUserSpaceWriteResult): string {
  return `${snapshotId}:${result.path}:${result.writeSignature || result.file?.updated_at || result.status}`;
}

function calculateUserspaceDiffLineCounts(before: string, after: string): { additions: number; deletions: number } {
  const changes = diffLines(before, after);
  let additions = 0;
  let deletions = 0;

  for (const change of changes) {
    const raw = change.value;
    const lines = raw.endsWith('\n') ? raw.slice(0, -1).split('\n') : raw.split('\n');
    if (change.added) {
      additions += lines.length;
    } else if (change.removed) {
      deletions += lines.length;
    }
  }

  return { additions, deletions };
}

function mergeUserspaceWriteDiff(
  baselineDiff: UserSpaceSnapshotFileDiff,
  result: ParsedUserSpaceWriteResult,
): UserSpaceSnapshotFileDiff {
  const nextAfterContent = result.file?.content ?? baselineDiff.after_content;
  const nextAfterPath = result.file?.path ?? baselineDiff.after_path ?? result.path;
  const { additions, deletions } = calculateUserspaceDiffLineCounts(baselineDiff.before_content, nextAfterContent);

  return {
    ...baselineDiff,
    path: result.path,
    after_path: nextAfterPath,
    after_content: nextAfterContent,
    additions,
    deletions,
    is_deleted_in_current: false,
    is_untracked_in_current: baselineDiff.status === 'A' || baselineDiff.is_untracked_in_current,
  };
}

// Component to display a tool call with collapsible details
// Memoized to prevent re-renders when tool call data hasn't changed
interface ToolCallDisplayProps {
  toolCall: ActiveToolCall;
  defaultExpanded?: boolean;
  conversationId?: string;
  workspaceId?: string;
  siblingEvents?: Array<{ type: string; tool?: string; output?: string }>;
  onRetrySuccess?: (newOutput: string) => void;
  onOpenWorkspaceFile?: (path: string) => void;
}

interface ScreenshotToolOutput {
  preview_image_url?: string;
  render?: {
    width?: number;
    height?: number;
    effective_wait_after_load_ms?: number;
  };
  screenshot?: {
    preview_image_url?: string;
    render?: {
      width?: number;
      height?: number;
      effective_wait_after_load_ms?: number;
    };
  };
}

interface ParsedTerminalOutput {
  status: string;
  command?: string;
  cwd: string;
  exit_code: number;
  stdout: string;
  stderr: string;
  error?: string;
  timed_out?: boolean;
  truncated?: boolean;
}

const TERMINAL_TOOL_NAMES = new Set(['run_terminal_command']);
const TERMINAL_TOOL_CONNECTION_TYPES = new Set(['ssh_shell']);
const TERMINAL_PRESENTATION_KIND = 'terminal';
const USERSPACE_EXEC_RERUN_KIND = 'userspace_exec';
const CONVERSATION_TOOL_RERUN_KIND = 'conversation_tool';

function normalizedPresentationValue(value?: string | null): string {
  return (value || '').trim().toLowerCase();
}

function isTerminalToolCall(toolCall: ActiveToolCall): boolean {
  if (normalizedPresentationValue(toolCall.presentation?.kind) === TERMINAL_PRESENTATION_KIND) {
    return true;
  }

  if (TERMINAL_TOOL_NAMES.has(toolCall.tool)) return true;

  const toolType = toolCall.connection?.tool_type?.trim().toLowerCase();
  const connectionMode = normalizedPresentationValue(toolCall.connection?.connection_mode);
  return Boolean(
    (toolType && TERMINAL_TOOL_CONNECTION_TYPES.has(toolType))
    || (toolType === 'odoo_shell' && connectionMode === 'ssh')
  );
}

function canRerunToolCall(toolCall: ActiveToolCall): boolean {
  if (normalizedPresentationValue(toolCall.presentation?.rerun_kind) === USERSPACE_EXEC_RERUN_KIND) {
    return true;
  }
  if (normalizedPresentationValue(toolCall.presentation?.rerun_kind) === CONVERSATION_TOOL_RERUN_KIND) {
    return true;
  }
  const toolType = toolCall.connection?.tool_type?.trim().toLowerCase();
  const connectionMode = normalizedPresentationValue(toolCall.connection?.connection_mode);
  if (
    ((toolType && TERMINAL_TOOL_CONNECTION_TYPES.has(toolType))
      || (toolType === 'odoo_shell' && connectionMode === 'ssh'))
    && toolCall.connection?.tool_config_id
  ) {
    return true;
  }
  return toolCall.tool === 'run_terminal_command';
}

function getTerminalRerunKind(toolCall: ActiveToolCall): string | null {
  const presentationKind = normalizedPresentationValue(toolCall.presentation?.rerun_kind);
  if (presentationKind === USERSPACE_EXEC_RERUN_KIND || presentationKind === CONVERSATION_TOOL_RERUN_KIND) {
    return presentationKind;
  }
  const toolType = toolCall.connection?.tool_type?.trim().toLowerCase();
  const connectionMode = normalizedPresentationValue(toolCall.connection?.connection_mode);
  if (
    ((toolType && TERMINAL_TOOL_CONNECTION_TYPES.has(toolType))
      || (toolType === 'odoo_shell' && connectionMode === 'ssh'))
    && toolCall.connection?.tool_config_id
  ) {
    return CONVERSATION_TOOL_RERUN_KIND;
  }
  return toolCall.tool === 'run_terminal_command' ? USERSPACE_EXEC_RERUN_KIND : null;
}

function parseTerminalOutput(output: string | undefined | null): ParsedTerminalOutput | null {
  if (!output) return null;
  try {
    const parsed = JSON.parse(output) as Record<string, unknown>;
    const exitCodeRaw = parsed.exit_code;
    const hasTerminalText =
      typeof parsed.stdout === 'string'
      || typeof parsed.stderr === 'string'
      || typeof parsed.error === 'string';
    const hasValidExitCode =
      typeof exitCodeRaw === 'number'
      || (typeof exitCodeRaw === 'string' && /^-?\d+$/.test(exitCodeRaw));

    if (!hasTerminalText || !hasValidExitCode) return null;

    return {
      status: String(parsed.status ?? 'unknown'),
      command: typeof parsed.command === 'string' ? parsed.command : undefined,
      cwd: String(parsed.cwd ?? '.'),
      exit_code: Number(parsed.exit_code ?? 0),
      stdout: String(parsed.stdout ?? ''),
      stderr: String(parsed.stderr ?? ''),
      error: typeof parsed.error === 'string' ? parsed.error : undefined,
      timed_out: Boolean(parsed.timed_out),
      truncated: Boolean(parsed.truncated),
    };
  } catch {
    return null;
  }
}

const ToolCallDisplay = memo(function ToolCallDisplay({
  toolCall,
  defaultExpanded = false,
  conversationId,
  workspaceId,
  siblingEvents,
  onRetrySuccess,
  onOpenWorkspaceFile,
}: ToolCallDisplayProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [copiedQuery, setCopiedQuery] = useState(false);
  const [copiedResult, setCopiedResult] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryOutput, setRetryOutput] = useState<string | null>(null);
  const [retryError, setRetryError] = useState<string | null>(null);
  const [isRerunning, setIsRerunning] = useState(false);
  const [zoomedImage, setZoomedImage] = useState<string | null>(null);
  const [userspaceFileDiff, setUserspaceFileDiff] = useState<UserSpaceSnapshotFileDiff | null>(null);
  const [userspaceFileDiffError, setUserspaceFileDiffError] = useState<string | null>(null);
  const [userspaceFileDiffLoading, setUserspaceFileDiffLoading] = useState(false);
  const [userspaceFileDiffKey, setUserspaceFileDiffKey] = useState<string | null>(null);
  const [showUserspaceDiffOverlay, setShowUserspaceDiffOverlay] = useState(false);
  const latestOutput = retryOutput || toolCall.output;
  const parsedTerminalOutput = useMemo(() => parseTerminalOutput(latestOutput), [latestOutput]);

  // Check if this is a visualization tool that can be retried
  const isVisualizationTool = toolCall.tool === 'create_chart' || toolCall.tool === 'create_datatable';
  const isTerminalCommand = isTerminalToolCall(toolCall) || Boolean(parsedTerminalOutput);
  const rerunKind = getTerminalRerunKind(toolCall);
  const canRerun = canRerunToolCall(toolCall);
  const hasRerunContext = rerunKind === USERSPACE_EXEC_RERUN_KIND
    ? Boolean(workspaceId)
    : rerunKind === CONVERSATION_TOOL_RERUN_KIND
      ? Boolean(conversationId && toolCall.connection?.tool_config_id)
      : false;
  const activeTerminalOutput = isTerminalCommand && isRerunning ? retryOutput : latestOutput;

  // Parse terminal output for terminal-style rendering
  const terminalOutput = useMemo(() => {
    if (!isTerminalCommand) return null;
    return parseTerminalOutput(activeTerminalOutput);
  }, [activeTerminalOutput, isTerminalCommand]);

  // Check if this tool call failed based on output content
  const hasErrorInOutput = useMemo(() => {
    const output = isTerminalCommand && isRerunning ? retryOutput : latestOutput;
    if (!output) return false;

    // Prefer structured JSON status checks (avoids false positives on keys like "error_count": 0)
    try {
      const parsed = JSON.parse(output) as Record<string, unknown>;
      if (parsed && typeof parsed === 'object') {
        if (parsed.rejected === true || parsed.success === false) return true;
        if (parsed.ok === false) return true;
        if (typeof parsed.status === 'string') {
          const status = parsed.status.toLowerCase();
          if (status.includes('failed') || status.includes('error') || status.includes('rejected')) {
            return true;
          }
        }
        if (typeof parsed.error === 'string' && parsed.error.trim().length > 0) return true;
        return false;
      }
    } catch {
      // Non-JSON output: fall through to conservative text checks
    }

    const outputLower = output.toLowerCase();
    return (
      /(^|\n)\s*error\s*[:\-]/i.test(output) ||
      outputLower.includes('validation error') ||
      outputLower.includes('exception') ||
      outputLower.includes('traceback') ||
      outputLower.includes('tool error') ||
      outputLower.includes('failed')
    );
  }, [isTerminalCommand, isRerunning, latestOutput, retryOutput]);

  // Effective output (use retry output if available)
  const effectiveOutput = isTerminalCommand && isRerunning ? (retryOutput || '') : latestOutput;

  const userspaceWriteResult = useMemo(() => {
    if (hasErrorInOutput) {
      return parseUserspaceWriteToolResult(toolCall.tool, toolCall.output);
    }
    return parseUserspaceWriteToolResult(toolCall.tool, effectiveOutput);
  }, [effectiveOutput, hasErrorInOutput, toolCall.output, toolCall.tool]);

  const userspaceDiffReady = Boolean(
    userspaceWriteResult
    && userspaceWriteResult.persisted
    && !userspaceWriteResult.rejected
    && !userspaceWriteResult.noChanges
    && workspaceId,
  );

  useEffect(() => {
    if (!expanded || !userspaceDiffReady || !workspaceId || !userspaceWriteResult) {
      return;
    }

    let cancelled = false;

    const loadDiff = async () => {
      setUserspaceFileDiffError(null);
      setUserspaceFileDiffLoading(true);

      try {
        const timeline = await api.getUserSpaceSnapshotTimeline(workspaceId);
        const snapshotId = timeline.current_snapshot_id ?? null;
        if (!snapshotId) {
          throw new Error('No snapshot baseline is available for this workspace yet.');
        }

        const cacheKey = buildUserspaceToolDiffCacheKey(snapshotId, userspaceWriteResult);
        if (!cancelled) {
          setUserspaceFileDiffKey(cacheKey);
        }

        const cached = userspaceWriteDiffCache.get(cacheKey);
        if (cached) {
          if (!cancelled) {
            setUserspaceFileDiff(cached);
            setUserspaceFileDiffLoading(false);
          }
          return;
        }

        let finalDiff: UserSpaceSnapshotFileDiff;
        try {
          const baselineDiff = await api.getUserSpaceSnapshotFileDiff(workspaceId, snapshotId, userspaceWriteResult.path);
          // When the tool payload includes file.content, merge it as the
          // stable "after" side so historical diffs stay accurate even if
          // the file was edited after this tool call. When file.content is
          // missing (stripped/truncated events), fall back to the snapshot
          // API diff which shows snapshot → current workspace state.
          finalDiff = userspaceWriteResult.file?.content != null
            ? mergeUserspaceWriteDiff(baselineDiff, userspaceWriteResult)
            : baselineDiff;
        } catch {
          // Snapshot diff returned 404 — the file has no changes relative
          // to the current snapshot (typically because a snapshot was
          // created after this tool write captured the change). Fall back
          // to reading the current file content and showing it as an
          // "added" diff so the card isn't blank.
          const currentFile = await api.getUserSpaceFile(workspaceId, userspaceWriteResult.path);
          const afterContent = userspaceWriteResult.file?.content ?? currentFile.content ?? '';
          const lines = afterContent.split('\n').length;
          finalDiff = {
            workspace_id: workspaceId,
            snapshot_id: snapshotId,
            path: userspaceWriteResult.path,
            status: 'A',
            before_content: '',
            after_content: afterContent,
            additions: lines,
            deletions: 0,
            is_binary: false,
            is_deleted_in_current: false,
            is_untracked_in_current: false,
          };
        }
        userspaceWriteDiffCache.set(cacheKey, finalDiff);
        trimUserspaceWriteDiffCache();

        if (!cancelled) {
          setUserspaceFileDiff(finalDiff);
        }
      } catch (err) {
        if (!cancelled) {
          setUserspaceFileDiff(null);
          setUserspaceFileDiffError(err instanceof Error ? err.message : 'Failed to load file diff');
        }
      } finally {
        if (!cancelled) {
          setUserspaceFileDiffLoading(false);
        }
      }
    };

    void loadDiff();

    return () => {
      cancelled = true;
    };
  }, [expanded, userspaceDiffReady, workspaceId, userspaceWriteResult]);

  const screenshotPreview = useMemo(() => {
    if (toolCall.tool !== 'capture_userspace_screenshot' || !effectiveOutput) {
      return null;
    }
    try {
      const parsed = JSON.parse(effectiveOutput) as ScreenshotToolOutput;
      const nested = parsed.screenshot && typeof parsed.screenshot === 'object'
        ? parsed.screenshot
        : null;
      const url = typeof parsed.preview_image_url === 'string'
        ? parsed.preview_image_url.trim()
        : typeof nested?.preview_image_url === 'string'
          ? nested.preview_image_url.trim()
          : '';
      if (!url) return null;
      const render = parsed.render ?? nested?.render;
      const width = Number(render?.width || 0);
      const height = Number(render?.height || 0);
      const effectiveWait = Number(render?.effective_wait_after_load_ms || 0);
      return {
        imageUrl: url,
        width: Number.isFinite(width) && width > 0 ? width : null,
        height: Number.isFinite(height) && height > 0 ? height : null,
        effectiveWait: Number.isFinite(effectiveWait) && effectiveWait > 0 ? effectiveWait : null,
      };
    } catch {
      return null;
    }
  }, [effectiveOutput, toolCall.tool]);

  // Check if this is a chart tool
  const chartData = useMemo(() => {
    if (toolCall.tool === 'create_chart' && effectiveOutput && !hasErrorInOutput) {
      return parseChartData(effectiveOutput);
    }
    return null;
  }, [toolCall.tool, effectiveOutput, hasErrorInOutput]);

  // Check if this is a datatable tool
  const datatableData = useMemo(() => {
    if (toolCall.tool === 'create_datatable' && effectiveOutput && !hasErrorInOutput) {
      const result = parseDataTableData(effectiveOutput);
      return result;
    }
    return null;
  }, [toolCall.tool, effectiveOutput, hasErrorInOutput]);

  // Determine if this visualization tool call actually failed
  // (either error in output OR parsing failed for visualization tools)
  const isFailed = useMemo(() => {
    if (hasErrorInOutput) return true;
    // If it's a visualization tool with output but parsing failed, that's also a failure
    if (isVisualizationTool && toolCall.output && toolCall.status === 'complete') {
      if (toolCall.tool === 'create_chart' && !chartData) return true;
      if (toolCall.tool === 'create_datatable' && !datatableData) return true;
    }
    return false;
  }, [hasErrorInOutput, isVisualizationTool, toolCall.output, toolCall.status, toolCall.tool, chartData, datatableData]);

  // Parse table metadata from output if present (for SQL results)
  const { tableData, displayText } = useMemo(() => {
    if (!toolCall.output) return { tableData: null, displayText: '' };
    return parseTableMetadata(toolCall.output);
  }, [toolCall.output]);

  // Memoize formatted input to avoid recalculating on every render
  const inputDisplay = useMemo(() => {
    if (!toolCall.input) return '';
    // Try to find the query or code in common field names
    const queryFields = ['query', 'sql', 'code', 'command', 'python_code'];
    for (const field of queryFields) {
      if (toolCall.input[field] && typeof toolCall.input[field] === 'string') {
        return toolCall.input[field] as string;
      }
    }
    // Fall back to JSON
    return JSON.stringify(toolCall.input, null, 2);
  }, [toolCall.input]);

  const copyToClipboard = useCallback(async (text: string, type: 'query' | 'result') => {
    try {
      await navigator.clipboard.writeText(text);
      if (type === 'query') {
        setCopiedQuery(true);
        setTimeout(() => setCopiedQuery(false), 2000);
      } else {
        setCopiedResult(true);
        setTimeout(() => setCopiedResult(false), 2000);
      }
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, []);

  // Handle retry for visualization tools
  const handleRetry = useCallback(async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (!conversationId) {
      setRetryError('Cannot retry: missing conversation context');
      return;
    }

    // Find source data from sibling events (previous tool calls with TABLEDATA)
    let sourceData: { columns: string[]; rows: unknown[][] } | null = null;

    if (siblingEvents) {
      for (const event of siblingEvents) {
        if (event.type === 'tool' && event.output) {
          // Check for TABLEDATA metadata in the output
          const metadata = parseTableMetadata(event.output);
          if (metadata.tableData) {
            sourceData = {
              columns: metadata.tableData.columns,
              rows: metadata.tableData.rows
            };
            break;
          }
        }
      }
    }

    if (!sourceData) {
      setRetryError('Cannot retry: no table data found from previous queries');
      setExpanded(true);
      return;
    }
    setIsRetrying(true);
    setRetryError(null);

    try {
      const toolType = toolCall.tool === 'create_chart' ? 'chart' : 'datatable';
      const response = await fetch(`/indexes/conversations/${conversationId}/retry-visualization`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          tool_type: toolType,
          source_data: sourceData,
          title: 'Data'  // Default title
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success && result.output) {
        setRetryOutput(result.output);
        onRetrySuccess?.(result.output);
      } else {
        setRetryError(result.error || 'Unknown error');
        setExpanded(true);
      }
    } catch (err) {
      setRetryError(err instanceof Error ? err.message : 'Request failed');
      setExpanded(true);
    } finally {
      setIsRetrying(false);
    }
  }, [conversationId, siblingEvents, toolCall.tool, onRetrySuccess]);

  // Handle re-run for terminal commands
  const handleRerunCommand = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!canRerun || !rerunKind || isRerunning || !inputDisplay) return;
    setIsRerunning(true);
    setRetryOutput(null);
    setRetryError(null);
    try {
      if (rerunKind === USERSPACE_EXEC_RERUN_KIND) {
        if (!workspaceId) return;
        const result = await api.execWorkspaceCommand(
          workspaceId,
          inputDisplay,
          toolCall.input?.timeout_seconds as number || 30,
          (toolCall.input?.cwd as string) || undefined,
        );
        const payload = {
          tool: 'run_terminal_command',
          status: result.timed_out ? 'command_timed_out' : (result.exit_code !== 0 ? 'command_failed' : 'completed'),
          cwd: result.cwd ?? (toolCall.input?.cwd as string) ?? '.',
          exit_code: result.exit_code ?? 0,
          ...(result.stdout ? { stdout: result.stdout } : {}),
          ...(result.stderr ? { stderr: result.stderr } : {}),
          ...(result.timed_out ? { timed_out: true } : {}),
          ...(result.truncated ? { truncated: true } : {}),
        };
        setRetryOutput(JSON.stringify(payload, null, 2));
        return;
      }

      if (rerunKind === CONVERSATION_TOOL_RERUN_KIND) {
        if (!conversationId || !toolCall.connection?.tool_config_id) return;
        const result = await api.retryTerminalToolCall(
          conversationId,
          {
            tool_config_id: toolCall.connection.tool_config_id,
            input: toolCall.input || {},
          },
          workspaceId,
        );
        if (!result.success || !result.output) {
          throw new Error(result.error || 'Re-run failed');
        }
        setRetryOutput(result.output);
      }
    } catch (err) {
      setRetryError(err instanceof Error ? err.message : 'Re-run failed');
    } finally {
      setIsRerunning(false);
    }
  }, [canRerun, rerunKind, isRerunning, inputDisplay, workspaceId, conversationId, toolCall.connection, toolCall.input]);

  // Special rendering for chart tool - show chart inline without collapsible
  if (chartData) {
    return (
      <div className="tool-call tool-call-chart tool-call-complete">
        <ChartDisplay chartData={chartData} />
      </div>
    );
  }

  // Special rendering for datatable tool - show table inline without collapsible
  if (datatableData) {
    return (
      <div className="tool-call tool-call-datatable tool-call-complete">
        <DataTableDisplay tableData={datatableData} />
      </div>
    );
  }

  // Determine the tool-type icon (always visible)
  const getToolIcon = () => {
    const name = toolCall.tool.toLowerCase();
    if (name.includes('sql') || name.includes('database') || name.includes('db')) return <Database size={14} />;
    if (name.includes('search') || name.includes('retrieval') || name.includes('index')) return <Search size={14} />;
    if (name.includes('chart') || name.includes('datatable') || name.includes('visuali')) return <BarChart3 size={14} />;
    if (name.includes('shell') || name.includes('command') || name.includes('ssh') || name.includes('terminal') || name.includes('exec')) return <Terminal size={14} />;
    if (name.includes('screenshot') || name.includes('image') || name.includes('preview')) return <ImageIcon size={14} />;
    if (name.includes('file') || name.includes('read') || name.includes('write') || name.includes('userspace')) return <FileText size={14} />;
    if (name.includes('web') || name.includes('http') || name.includes('url') || name.includes('fetch') || name.includes('browse')) return <Globe size={14} />;
    if (name.includes('code') || name.includes('odoo') || name.includes('python')) return <Code size={14} />;
    if (name.includes('schema') || name.includes('pdm') || name.includes('metadata')) return <FolderSearch size={14} />;
    return <Wrench size={14} />;
  };

  // Determine the status icon (overrides tool icon when active)
  const getStatusIcon = () => {
    if (toolCall.status === 'running' || isRerunning) {
      return <MiniLoadingSpinner variant="icon" size={14} />;
    }
    if (isFailed && isVisualizationTool) {
      return <AlertCircle size={14} className="tool-call-error-icon" />;
    }
    if (isFailed && isTerminalCommand) {
      return <AlertCircle size={14} className="tool-call-error-icon" />;
    }
    return null;
  };

  const statusIcon = getStatusIcon();
  const toolIcon = getToolIcon();
  const userspaceWriteSummary = userspaceWriteResult ? formatUserspaceWriteSummary(userspaceWriteResult) : '';
  const userspaceDiffStatus = userspaceFileDiff ? formatDiffStatus(userspaceFileDiff.status) : null;

  return (
  <>
    <div className={`tool-call tool-call-${toolCall.status}${isFailed ? ' tool-call-failed' : ''}`}>
      <div className="tool-call-header-row">
        <button
          className="tool-call-header"
          onClick={() => setExpanded(!expanded)}
        >
          <span className="tool-call-icon">{statusIcon || toolIcon}</span>
          {isTerminalCommand ? (
            <span className="tool-call-name tool-call-name-terminal">
              {inputDisplay || toolCall.tool}
            </span>
          ) : (
            <span className="tool-call-name">{toolCall.tool}</span>
          )}
          {toolCall.status === 'running' && toolCall.generating_lines ? (
            <span className="tool-call-progress">{toolCall.generating_lines} lines</span>
          ) : null}
          <span className="tool-call-toggle" aria-hidden="true">
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </span>
        </button>
        {isFailed && isVisualizationTool && !isRetrying && (
          <button
            type="button"
            className="tool-call-retry-btn"
            onClick={handleRetry}
            title="Retry creating visualization from source data"
          >
            <RefreshCw size={12} />
            <span>Retry</span>
          </button>
        )}
        {isRetrying && (
          <span className="tool-call-retrying">
            <MiniLoadingSpinner variant="icon" size={12} />
            <span>Retrying...</span>
          </span>
        )}
      </div>
      {expanded && (
        <div className="tool-call-details">
          {retryError && (
            <div className="tool-call-section tool-call-error">
              <div className="tool-call-section-header">
                <span className="tool-call-section-label">Retry Error:</span>
              </div>
              <pre className="tool-call-code tool-call-error-text">{retryError}</pre>
            </div>
          )}
          {inputDisplay && !userspaceWriteResult && !isTerminalCommand && (
            <div className="tool-call-section">
              <div className="tool-call-section-header">
                <span className="tool-call-section-label">Query:</span>
                <button
                  className="tool-call-copy-btn"
                  onClick={() => copyToClipboard(inputDisplay, 'query')}
                  title="Copy query"
                >
                  {copiedQuery ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
              <pre className="tool-call-code">{inputDisplay}</pre>
            </div>
          )}
          {isTerminalCommand && terminalOutput ? (
            <div className="tool-call-section tool-call-terminal-section">
              <div className="tool-call-terminal-block">
                <div className="tool-call-terminal-header-bar">
                  <span className="tool-call-terminal-cwd">{terminalOutput.cwd === '.' ? '~' : `~/${terminalOutput.cwd}`}</span>
                  <div className="tool-call-terminal-header-actions">
                    <button
                      className="tool-call-copy-btn"
                      onClick={() => copyToClipboard(inputDisplay, 'query')}
                      title="Copy command"
                    >
                      {copiedQuery ? <Check size={12} /> : <Terminal size={12} />}
                    </button>
                    <button
                      className="tool-call-copy-btn"
                      onClick={() => copyToClipboard(
                        [terminalOutput.stdout, terminalOutput.stderr].filter(Boolean).join('\n') || inputDisplay,
                        'result'
                      )}
                      title="Copy output"
                    >
                      {copiedResult ? <Check size={12} /> : <Copy size={12} />}
                    </button>
                    {canRerun && hasRerunContext && (
                      <button
                        className="tool-call-copy-btn tool-call-terminal-rerun-btn"
                        onClick={handleRerunCommand}
                        title="Re-run command"
                        disabled={isRerunning}
                      >
                        {isRerunning ? <MiniLoadingSpinner variant="icon" size={12} /> : <Play size={12} />}
                      </button>
                    )}
                  </div>
                </div>
                <pre className="tool-call-terminal-output">
                  <span className="tool-call-terminal-prompt-line">$ {inputDisplay}</span>
                  {'\n'}
                  {terminalOutput.stdout}
                  {terminalOutput.stderr ? (
                    <span className="tool-call-terminal-stderr">{terminalOutput.stderr}</span>
                  ) : null}
                </pre>
                {terminalOutput.truncated && (
                  <div className="tool-call-terminal-notice">Output truncated</div>
                )}
                {terminalOutput.error && terminalOutput.status !== 'completed' && (
                  <div className="tool-call-terminal-error-banner">
                    <AlertCircle size={12} />
                    <span>{terminalOutput.error}</span>
                  </div>
                )}
              </div>
            </div>
          ) : isTerminalCommand && (toolCall.status === 'running' || isRerunning) ? (
            <div className="tool-call-section tool-call-terminal-section">
              <div className="tool-call-terminal-block">
                <pre className="tool-call-terminal-output">
                  <span className="tool-call-terminal-prompt-line">$ {inputDisplay}</span>
                  {'\n'}
                  <MiniLoadingSpinner variant="icon" size={12} />
                </pre>
              </div>
            </div>
          ) : isTerminalCommand ? (
            <div className="tool-call-section tool-call-terminal-section">
              <div className="tool-call-terminal-block">
                <div className="tool-call-terminal-header-bar">
                  <span className="tool-call-terminal-cwd">~</span>
                  <div className="tool-call-terminal-header-actions">
                    <button
                      className="tool-call-copy-btn"
                      onClick={() => copyToClipboard(inputDisplay, 'query')}
                      title="Copy command"
                    >
                      {copiedQuery ? <Check size={12} /> : <Terminal size={12} />}
                    </button>
                    {canRerun && hasRerunContext && (
                      <button
                        className="tool-call-copy-btn tool-call-terminal-rerun-btn"
                        onClick={handleRerunCommand}
                        title="Re-run command"
                        disabled={isRerunning}
                      >
                        {isRerunning ? <MiniLoadingSpinner variant="icon" size={12} /> : <Play size={12} />}
                      </button>
                    )}
                  </div>
                </div>
                <pre className="tool-call-terminal-output">
                  <span className="tool-call-terminal-prompt-line">$ {inputDisplay}</span>
                  {'\n'}
                  <span className="tool-call-terminal-empty">No output</span>
                </pre>
              </div>
            </div>
          ) : null}
          {toolCall.output && !isTerminalCommand && (
            screenshotPreview && !hasErrorInOutput ? (
              <div className="tool-call-section">
                <div className="tool-call-screenshot-meta">
                  {screenshotPreview.width && screenshotPreview.height
                    ? `${screenshotPreview.width}\u00d7${screenshotPreview.height}`
                    : 'Screenshot'}
                  {screenshotPreview.effectiveWait
                    ? ` | settled ${screenshotPreview.effectiveWait}ms`
                    : ''}
                </div>
                <img
                  src={screenshotPreview.imageUrl}
                  alt="Captured User Space screenshot"
                  className="tool-call-screenshot-image"
                  loading="lazy"
                  onClick={() => setZoomedImage(screenshotPreview.imageUrl)}
                  style={{ cursor: 'zoom-in' }}
                />
              </div>
            ) : userspaceWriteResult ? (
              <>
                <div className="tool-call-section">
                  <div className="tool-call-section-header">
                    <span className="tool-call-section-label">Result:</span>
                    {userspaceFileDiff ? (
                      <button
                        type="button"
                        className="tool-call-retry-btn"
                        onClick={() => setShowUserspaceDiffOverlay(true)}
                        title="Open full diff"
                      >
                        <Diff size={12} />
                        <span>Full Diff</span>
                      </button>
                    ) : (
                      <button
                        className="tool-call-copy-btn"
                        onClick={() => copyToClipboard(userspaceWriteSummary || displayText || toolCall.output!, 'result')}
                        title="Copy result"
                      >
                        {copiedResult ? <Check size={12} /> : <Copy size={12} />}
                      </button>
                    )}
                  </div>
                  <div className="tool-call-userspace-write-summary">
                    <div className="tool-call-userspace-write-summary-row">
                      {onOpenWorkspaceFile ? (
                        <button
                          className="tool-call-userspace-write-path tool-call-userspace-write-path-link"
                          title={userspaceWriteResult.path}
                          onClick={() => onOpenWorkspaceFile(userspaceWriteResult.path)}
                        >
                          {userspaceWriteResult.path}
                        </button>
                      ) : (
                        <span className="tool-call-userspace-write-path" title={userspaceWriteResult.path}>{userspaceWriteResult.path}</span>
                      )}
                      <span className={`userspace-snapshot-diff-status userspace-snapshot-diff-status-${(userspaceFileDiff?.status ?? 'm').toLowerCase()}`}>
                        {userspaceFileDiff?.status ?? 'M'}
                      </span>
                    </div>
                    <div className="tool-call-userspace-write-meta">
                      {userspaceDiffStatus || userspaceWriteResult.message || 'Updated'}
                      {userspaceFileDiff ? ` | +${userspaceFileDiff.additions} -${userspaceFileDiff.deletions}` : ''}
                    </div>
                  </div>
                  {userspaceFileDiffLoading && (
                    <div className="tool-call-userspace-diff-loading">
                      <MiniLoadingSpinner variant="icon" size={14} />
                      <span>Loading diff from current snapshot...</span>
                    </div>
                  )}
                  {!userspaceFileDiffLoading && !userspaceFileDiff && (
                    <pre className="tool-call-code">{userspaceWriteSummary || displayText || toolCall.output}</pre>
                  )}
                  {userspaceFileDiffError && (
                    <div className="tool-call-userspace-diff-error">{userspaceFileDiffError}</div>
                  )}
                </div>
                {!userspaceFileDiffLoading && userspaceFileDiff && (
                  <div className="tool-call-userspace-diff-card">
                    <UserSpaceFileDiffView
                      diff={userspaceFileDiff}
                      beforeLabel="Last Snapshot"
                      afterLabel="Tool Result"
                      compact
                    />
                  </div>
                )}
              </>
            ) : (
              <div className="tool-call-section">
                <div className="tool-call-section-header">
                  <span className="tool-call-section-label">Result:</span>
                  <button
                    className="tool-call-copy-btn"
                    onClick={() => copyToClipboard(displayText || toolCall.output!, 'result')}
                    title="Copy result"
                  >
                    {copiedResult ? <Check size={12} /> : <Copy size={12} />}
                  </button>
                </div>
                {tableData ? (
                  <DataTable data={tableData} />
                ) : (
                  <pre className="tool-call-code">{displayText}</pre>
                )}
              </div>
            )
          )}
        </div>
      )}
    </div>
    {zoomedImage && (
      <div
        className="image-modal-overlay"
        onClick={() => setZoomedImage(null)}
        onKeyDown={(e) => e.key === 'Escape' && setZoomedImage(null)}
        role="dialog"
        aria-modal="true"
        tabIndex={-1}
      >
        <div className="image-modal-content" onClick={(e) => e.stopPropagation()}>
          <button
            className="image-modal-close"
            onClick={() => setZoomedImage(null)}
            title="Close"
          >
            <X size={24} />
          </button>
          <img src={zoomedImage!} alt="Screenshot full view" />
        </div>
      </div>
    )}
    {showUserspaceDiffOverlay && (
      <FileDiffOverlay
        key={userspaceFileDiffKey ?? 'userspace-file-diff-overlay'}
        diff={userspaceFileDiff}
        loading={userspaceFileDiffLoading}
        error={userspaceFileDiffError}
        title="Userspace File Diff"
        beforeLabel="Last Snapshot"
        afterLabel="Tool Result"
        onDismiss={() => setShowUserspaceDiffOverlay(false)}
        onOverlayClick={() => {}}
        onOverlayMouseEnter={() => {}}
        onOverlayMouseLeave={() => {}}
      />
    )}
  </>
  );
})

// Consolidated streaming content - groups content events between tool calls
// This avoids re-rendering markdown for every token and dramatically improves performance
interface StreamingSegment {
  type: 'content' | 'tool' | 'reasoning';
  content?: string;  // For content/reasoning segments - consolidated text
  toolCall?: ActiveToolCall;  // For tool segments
  isComplete?: boolean;  // For reasoning segments - whether thinking has finished
  durationSeconds?: number;  // For reasoning segments - persisted final elapsed time
  embeddedToolCalls?: ActiveToolCall[];  // For reasoning segments - nested tool executions
  reasoningParts?: ReasoningPart[];  // For reasoning segments - ordered text/tool parts
}

// Memoized component for rendering streaming segments efficiently
// Collapsible reasoning/thinking display

interface ReasoningPart {
  type: 'text' | 'tool';
  text?: string;
  toolCall?: ActiveToolCall;
}

const ReasoningDisplay = memo(function ReasoningDisplay({
  content,
  isComplete,
  parts,
  toolCalls,
  durationSeconds,
  visibility = 'compact',
  workspaceId,
  conversationId,
  onOpenWorkspaceFile,
}: {
  content: string;
  isComplete: boolean;
  parts?: ReasoningPart[];
  toolCalls?: ActiveToolCall[];
  durationSeconds?: number;
  visibility?: 'compact' | 'expanded' | 'hidden';
  workspaceId?: string;
  conversationId?: string;
  onOpenWorkspaceFile?: (path: string) => void;
}) {
  const [isExpanded, setIsExpanded] = useState(() => {
    if (visibility === 'expanded') return true;
    if (visibility === 'hidden') return false;
    return !isComplete; // compact: open while streaming, closed after
  });
  // Track whether the user has manually toggled, so we don't override their choice while streaming
  const userToggledRef = useRef(false);
  const [copied, setCopied] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const startTimeRef = useRef<number>(Date.now());
  const [elapsed, setElapsed] = useState(0);

  const handleToggle = useCallback(() => {
    userToggledRef.current = true;
    setIsExpanded(prev => !prev);
  }, []);

  // Strip <tool_call> XML blocks from reasoning text — these are the model's
  // planned tool calls that get promoted to real tool events. We show the real
  // executed tool cards (from parts/embeddedToolCalls) instead of the raw XML.
  const cleanToolCallXml = useCallback((text: string) => {
    // Strip complete <tool_call>...</tool_call> blocks
    let cleaned = text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '');
    // During streaming, truncate at any unclosed <tool_call> tag
    if (!isComplete) {
      const openIdx = cleaned.lastIndexOf('<tool_call>');
      if (openIdx !== -1 && cleaned.indexOf('</tool_call>', openIdx) === -1) {
        cleaned = cleaned.substring(0, openIdx);
      }
      // Also catch partially-typed opening tags (e.g. "<tool_c")
      const partial = cleaned.match(/<tool_c[^>]*$/);
      if (partial && partial.index !== undefined) {
        cleaned = cleaned.substring(0, partial.index);
      }
    }
    return cleaned.replace(/\n{3,}/g, '\n\n').trimEnd();
  }, [isComplete]);

  // Prefer structured reasoning parts from streaming events (ordered text + tools).
  // Clean tool-call XML from each text part.
  const renderedParts = useMemo(() => {
    const raw = (parts && parts.length > 0) ? parts : [{ type: 'text' as const, text: content }];
    return raw.map(part => {
      if (part.type !== 'text' || !part.text) return part;
      const cleaned = cleanToolCallXml(part.text);
      if (!cleaned) return null;
      return { ...part, text: cleaned };
    }).filter(Boolean) as ReasoningPart[];
  }, [parts, content, cleanToolCallXml]);

  const cleanedContent = useMemo(() => cleanToolCallXml(content), [content, cleanToolCallXml]);
  const hasToolCalls = renderedParts.some((part) => part.type === 'tool' && !!part.toolCall);

  // Elapsed timer while streaming
  useEffect(() => {
    if (isComplete) return;
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [isComplete]);

  // Auto-scroll reasoning content while streaming
  useEffect(() => {
    if (!isComplete && isExpanded && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [content, isComplete, isExpanded]);

  // Auto-collapse when thinking completes (compact mode) — but never override a manual toggle
  useEffect(() => {
    if (isComplete && visibility === 'compact' && !userToggledRef.current) {
      const timer = setTimeout(() => setIsExpanded(false), 600);
      return () => clearTimeout(timer);
    }
    // Reset manual toggle flag when completion state changes (new stream)
    if (!isComplete) {
      userToggledRef.current = false;
    }
  }, [isComplete, visibility]);

  // Respect visibility changes
  useEffect(() => {
    if (visibility === 'expanded') setIsExpanded(true);
    else if (visibility === 'hidden') setIsExpanded(false);
  }, [visibility]);

  // Metadata for header
  const charCount = cleanedContent.length;
  const toolCount = renderedParts.filter((part) => part.type === 'tool' && !!part.toolCall).length || (toolCalls?.length ?? 0);

  // Summary: first meaningful line for compact header
  const summaryLine = useMemo(() => {
    const lines = cleanedContent.split('\n').map(l => l.trim()).filter(l => l.length > 10);
    const first = (lines[0] || '').replace(/^\*\*(.+)\*\*$/, '$1');
    return first.length > 120 ? first.slice(0, 117) + '...' : first;
  }, [cleanedContent]);

  const formatElapsed = (s: number) => {
    if (s < 60) return `${s}s`;
    return `${Math.floor(s / 60)}m ${s % 60}s`;
  };
  const resolvedElapsed = isComplete ? (durationSeconds ?? elapsed) : elapsed;

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy reasoning:', err);
    }
  }, [content]);

  // Convert standalone **title** lines in reasoning text to ### markdown headers
  // so the model's section headings are visually distinct rather than showing raw asterisks.
  const formatReasoningText = useCallback((text: string) => {
    let result = text.replace(/\r\n?/g, '\n');
    // Some streamed reasoning chunks place the next **Heading** immediately after
    // the previous sentence with no newline at all. Split those into their own block first.
    result = result.replace(/([^\n])(\*\*([A-Z][^*\n]{2,})\*\*)/g, '$1\n\n$2');
    result = result.replace(/(\*\*([A-Z][^*\n]{2,})\*\*)([^\n])/g, '$1\n\n$3');
    result = result.replace(/^\*\*([^*\n]+)\*\*\s*$/gm, (_, title) => `### ${title}`);
    // Ensure a blank line before and after headings so markdown always treats them as blocks.
    result = result.replace(/([^\n])\n(#{1,6} )/g, '$1\n\n$2');
    result = result.replace(/(#{1,6} [^\n]+)\n([^\n#])/g, '$1\n\n$2');
    result = result.replace(/\n{3,}/g, '\n\n');
    return result;
  }, []);

  // Keep this after all hooks so hook order remains stable across renders.
  if (visibility === 'hidden' && isComplete) return null;

  return (
    <div className={`reasoning-block ${isComplete ? 'reasoning-complete' : 'reasoning-active'}`}>
      <button
        className="reasoning-header"
        onClick={handleToggle}
        aria-expanded={isExpanded}
      >
        <BrainCircuit size={14} className="reasoning-icon" />
        <span className="reasoning-label">
          {isComplete ? 'Thought process' : 'Thinking...'}
        </span>
        <span className="reasoning-meta">
          {resolvedElapsed > 0 && (
            <span className="reasoning-meta-item" title={isComplete ? 'Reasoning time' : 'Elapsed time'}>
              <Clock size={10} /> {formatElapsed(resolvedElapsed)}
            </span>
          )}
          {toolCount > 0 && (
            <span className="reasoning-meta-item" title={`${toolCount} tool call${toolCount !== 1 ? 's' : ''}`}>
              {toolCount} tool{toolCount !== 1 ? 's' : ''}
            </span>
          )}
          {isComplete && charCount > 0 && (
            <span className="reasoning-meta-item" title={`${charCount.toLocaleString()} characters`}>
              {charCount > 1000 ? `${(charCount / 1000).toFixed(1)}k` : charCount} chars
            </span>
          )}
        </span>
        <span className={`reasoning-chevron ${isExpanded ? 'expanded' : ''}`}>
          <ChevronDown size={14} />
        </span>
      </button>
      {/* Compact summary when collapsed and complete */}
      {!isExpanded && isComplete && summaryLine && (
        <div className="reasoning-summary">{summaryLine}</div>
      )}
      <div className={`reasoning-content ${isExpanded ? 'reasoning-content-expanded' : 'reasoning-content-collapsed'}`}>
        <div className="reasoning-content-inner" ref={contentRef}>
          {hasToolCalls ? renderedParts.map((part, i) => {
            if (part.type === 'text') {
              return (
                <div key={i} className="markdown-content">
                  <MemoizedMarkdown content={formatReasoningText(part.text ?? '')} />
                </div>
              );
            }
            if (!part.toolCall) return null;
            return (
              <div key={i} className="chat-tool-calls reasoning-embedded-tool">
                <ToolCallDisplay
                  toolCall={part.toolCall}
                  defaultExpanded={false}
                  workspaceId={workspaceId}
                  conversationId={conversationId}
                  onOpenWorkspaceFile={onOpenWorkspaceFile}
                />
              </div>
            );
          }) : (
            <div className="markdown-content">
              <MemoizedMarkdown content={formatReasoningText(cleanedContent)} />
            </div>
          )}
        </div>
        {isExpanded && isComplete && (
          <div className="reasoning-actions">
            <button className="reasoning-copy-btn" onClick={handleCopy} title="Copy reasoning">
              {copied ? <Check size={12} /> : <Copy size={12} />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
});

const StreamingSegmentDisplay = memo(function StreamingSegmentDisplay({
  segment,
  showToolCalls,
  workspaceId,
  conversationId,
  onOpenWorkspaceFile,
}: {
  segment: StreamingSegment;
  showToolCalls: boolean;
  workspaceId?: string;
  conversationId?: string;
  onOpenWorkspaceFile?: (path: string) => void;
}) {
  if (segment.type === 'reasoning' && segment.content) {
    return (
      <ReasoningDisplay
        content={segment.content}
        isComplete={!!segment.isComplete}
        durationSeconds={segment.durationSeconds}
        parts={segment.reasoningParts}
        toolCalls={segment.embeddedToolCalls}
        workspaceId={workspaceId}
        conversationId={conversationId}
        onOpenWorkspaceFile={onOpenWorkspaceFile}
      />
    );
  } else if (segment.type === 'tool' && segment.toolCall && showToolCalls) {
    return (
      <div className="chat-tool-calls">
        <ToolCallDisplay
          toolCall={segment.toolCall}
          defaultExpanded={false}
          workspaceId={workspaceId}
          conversationId={conversationId}
          onOpenWorkspaceFile={onOpenWorkspaceFile}
        />
      </div>
    );
  } else if (segment.type === 'content' && segment.content) {
    // For streaming content, use plain text to avoid markdown re-parsing on every token
    // The final saved message will be properly rendered with markdown
    return (
      <div className="chat-message-text markdown-content">
        <MemoizedMarkdown content={segment.content} />
      </div>
    );
  }
  return null;
});

// Default context limit fallback when model not found in API response
const DEFAULT_CONTEXT_LIMIT = 8192;

// Helper to extract text and attachments from message content
function parseMessageContent(content: string | ContentPart[]): { text: string; attachments: ContentPart[] } {
  if (typeof content === 'string') {
    // Try to parse as JSON content parts (for backward compatibility)
    try {
      const parsed = JSON.parse(content);
      if (Array.isArray(parsed)) {
        const text = parsed
          .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
          .map(p => p.text)
          .join('\n');
        const attachments = parsed.filter((p): p is ContentPart => p.type !== 'text');
        return { text, attachments };
      }
    } catch {
      // Not JSON, treat as plain text
    }
    return { text: content, attachments: [] };
  }

  // Already parsed array
  const text = content
    .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
    .map(p => p.text)
    .join('\n');
  const attachments = content.filter((p): p is ContentPart => p.type !== 'text');
  return { text, attachments };
}

function getGroupToolIds(tools: UserSpaceAvailableTool[], groupId: string): string[] {
  return tools
    .filter((tool) => tool.group_id === groupId)
    .map((tool) => tool.id);
}

// Component to display message attachments
interface MessageAttachmentsProps {
  attachments: ContentPart[];
  onImageClick?: (url: string) => void;
}

const MessageAttachments = memo(function MessageAttachments({ attachments, onImageClick }: MessageAttachmentsProps) {
  if (attachments.length === 0) return null;

  return (
    <div className="message-attachments">
      {attachments.map((attachment, idx) => {
        if (attachment.type === 'image_url') {
          return (
            <div key={idx} className="message-attachment">
              <img
                src={attachment.image_url.url}
                alt="Attached image"
                className="message-attachment-image"
                onClick={() => onImageClick?.(attachment.image_url.url)}
              />
            </div>
          );
        } else if (attachment.type === 'file') {
          return (
            <div key={idx} className="message-attachment message-attachment-file">
              <FileText className="message-attachment-file-icon" size={16} />
              <span className="message-attachment-file-name" title={attachment.file_path}>
                {attachment.filename}
              </span>
            </div>
          );
        }
        return null;
      })}
    </div>
  );
});

const ChatTitle = memo(({ title }: { title: string }) => {
  const [displayTitle, setDisplayTitle] = useState(title);
  const previousTitleRef = useRef(title);

  useEffect(() => {
    if (previousTitleRef.current === title) return;

    if (previousTitleRef.current === 'Untitled Chat' && title !== 'Untitled Chat') {
      let i = 0;
      setDisplayTitle('');
      const interval = setInterval(() => {
        i++;
        if (i <= title.length) {
          setDisplayTitle(title.substring(0, i));
        } else {
          clearInterval(interval);
        }
      }, 30);
      previousTitleRef.current = title;
      return () => clearInterval(interval);
    } else {
      setDisplayTitle(title);
      previousTitleRef.current = title;
    }
  }, [title]);

  return <>{displayTitle}</>;
});

interface ChatPanelProps {
  currentUser: User;
  debugMode?: boolean;
  workspaceId?: string;
  workspaceChatState?: WorkspaceChatStateResponse | null;
  workspaceAvailableTools?: UserSpaceAvailableTool[];
  workspaceSelectedToolIds?: string[];
  workspaceSelectedToolGroupIds?: string[];
  onToggleWorkspaceTool?: (toolId: string) => void | Promise<void>;
  onToggleWorkspaceToolGroup?: (groupId: string) => void | Promise<void>;
  workspaceToolGroups?: ToolGroupInfo[];
  workspaceSavingTools?: boolean;
  conversationShareableUserIds?: string[];
  onUserMessageSubmitted?: (message: string) => void | Promise<void>;
  onTaskComplete?: () => void;
  onConversationStateChange?: (hasLive: boolean, hasInterrupted: boolean) => void;
  onActiveConversationChange?: (conversationId: string | null) => void;
  onBranchSwitch?: (branchId: string, associatedSnapshotId: string | null) => void | Promise<void>;
  onFullscreenChange?: (fullscreen: boolean) => void;
  onOpenWorkspaceFile?: (path: string) => void;
  onMessageSnapshotRestored?: (details?: {
    rolledBackSnapshot?: boolean;
    requiresRuntimeRestart?: boolean;
  }) => void;
  /**
   * Fired by the chat panel whenever something happened that may have
   * created, mutated, or moved the workspace snapshot cursor:
   *   - chat branch creation (auto-snapshot)
   *   - per-message snapshot rollback (delete-with-restore)
   *   - live agent `create_userspace_snapshot` tool calls observed during
   *     a streaming assistant turn
   * The parent decides whether to refresh the snapshots panel.
   */
  onSnapshotsMaybeChanged?: () => void;
  inputBanner?: ReactNode;
  embedded?: boolean;
  readOnly?: boolean;
  readOnlyMessage?: string;
  allowAdminReadOnlyBypass?: boolean;
}

export function ChatPanel({
  currentUser,
  debugMode = false,
  workspaceId,
  workspaceChatState,
  workspaceAvailableTools,
  workspaceSelectedToolIds,
  workspaceSelectedToolGroupIds,
  onToggleWorkspaceTool,
  onToggleWorkspaceToolGroup,
  workspaceToolGroups,
  workspaceSavingTools = false,
  conversationShareableUserIds,
  onUserMessageSubmitted,
  onTaskComplete,
  onConversationStateChange,
  onActiveConversationChange,
  onBranchSwitch,
  onFullscreenChange,
  onOpenWorkspaceFile,
  onMessageSnapshotRestored,
  onSnapshotsMaybeChanged,
  inputBanner,
  embedded = false,
  readOnly = false,
  readOnlyMessage,
  allowAdminReadOnlyBypass = false,
}: ChatPanelProps) {
  const MIN_INPUT_AREA_HEIGHT = 96;
  const INPUT_AREA_COLLAPSE_THRESHOLD = 80;
  const chatLayoutCookieName = getChatLayoutCookieName(currentUser.id);

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [isConversationSwitchLoading, setIsConversationSwitchLoading] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [attachments, setAttachments] = useState<AttachmentFile[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [streamingEvents, setStreamingEvents] = useState<StreamingRenderEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(!embedded);
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [inputAreaHeight, setInputAreaHeight] = useState(MIN_INPUT_AREA_HEIGHT);
  const [isInputAreaCollapsed, setIsInputAreaCollapsed] = useState(false);
  const [isMessagesCollapsed, setIsMessagesCollapsed] = useState(false);
  const [isManualResize, setIsManualResize] = useState(false);
  const [autoResizeState, setAutoResizeState] = useState<'growing' | 'shrinking' | null>(null);
  const [editingTitle, setEditingTitle] = useState<string | null>(null);
  const [titleInput, setTitleInput] = useState('');
  const [editingMessageIdx, setEditingMessageIdx] = useState<number | null>(null);
  const [editMessageContent, setEditMessageContent] = useState('');
  const [editMessageAttachments, setEditMessageAttachments] = useState<AttachmentFile[]>([]);
  const [hitMaxIterations, setHitMaxIterations] = useState(false);

  // Chat branching state
  const [branchPoints, setBranchPoints] = useState<ConversationBranchPointInfo[]>([]);
  const [branchSwitching, setBranchSwitching] = useState(false);
  const [branchSelections, setBranchSelections] = useState<Record<number, string>>({});
  const [copiedMessageIdx, setCopiedMessageIdx] = useState<number | null>(null);
  const [pendingDeleteIdx, setPendingDeleteIdx] = useState<number | null>(null);
  const activeConversationId = activeConversation?.id ?? null;
  const branchPointsByIndex = useMemo(
    () => new Map(branchPoints.map((point) => [point.branch_point_index, point] as const)),
    [branchPoints],
  );
  const branchesById = useMemo(
    () => new Map(branchPoints.flatMap((point) => point.branches.map((branch) => [branch.id, branch] as const))),
    [branchPoints],
  );
  const [showToolCalls, setShowToolCalls] = useState(() => {
    const saved = localStorage.getItem('chat-show-tool-calls');
    return saved !== null ? saved === 'true' : true;
  });
  const [lastSentMessage, setLastSentMessage] = useState<string>('');
  const [isConnectionError, setIsConnectionError] = useState(false);
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});
  const isAdmin = currentUser.role === 'admin';
  const isReadOnly = readOnly && !(allowAdminReadOnlyBypass && isAdmin);
  const effectiveReadOnlyMessage = readOnlyMessage || 'Workspace is read-only. Viewers can review messages but cannot send prompts.';

  // Inline confirmation for delete (conversation ID waiting for confirmation)
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  useEffect(() => {
    setPendingDeleteIdx(null);
  }, [activeConversationId]);

  // Background task state
  const [activeTask, setActiveTask] = useState<ChatTask | null>(null);
  const [interruptedTask, setInterruptedTask] = useState<ChatTask | null>(null);  // Last interrupted task for continue
  const [interruptedConversationIds, setInterruptedConversationIds] = useState<Set<string>>(new Set());
  const [interruptDismissed, setInterruptDismissed] = useState(false);
  const prevChatStateRef = useRef<InterruptChatStateSnapshot | undefined>(undefined);
  const [_isPollingTask, setIsPollingTask] = useState(false);

  const syncConversationActiveTaskId = useCallback((conversationId: string, taskId: string | null) => {
    setConversations((prev) => prev.map((conversation) => {
      if (conversation.id !== conversationId) return conversation;
      if ((conversation.active_task_id ?? null) === taskId) return conversation;
      return { ...conversation, active_task_id: taskId };
    }));
    setActiveConversation((prev) => {
      if (!prev || prev.id !== conversationId) return prev;
      if ((prev.active_task_id ?? null) === taskId) return prev;
      return { ...prev, active_task_id: taskId };
    });
  }, []);

  // Read/reset dismiss cookie when workspaceId changes.
  // Seed prevHasInterruptedRef from the cookie: if the user already dismissed an
  // interrupted alert, treat prior state as "interrupted" so the first poll that
  // sees interrupted=true isn't mistaken for a fresh false -> true transition.
  useEffect(() => {
    if (workspaceId) {
      const dismissed = isInterruptDismissed(currentUser.id, workspaceId);
      setInterruptDismissed(dismissed);
      prevChatStateRef.current = undefined;
    } else {
      setInterruptDismissed(false);
      prevChatStateRef.current = undefined;
    }
  }, [workspaceId, currentUser.id]);

  // Un-dismiss automatically when a fresh interruption fires (false -> true transition).
  // On the very first render prevHasInterruptedRef is null (unset), so we skip the
  // transition check to avoid clearing a valid dismiss cookie on page refresh.
  useEffect(() => {
    const rawInterrupted = Boolean(interruptedTask) || interruptedConversationIds.size > 0;
    const hasLive = Boolean(activeTask)
      || conversations.some(c => Boolean(c.active_task_id));
    if (!workspaceId) {
      prevChatStateRef.current = undefined;
      return;
    }

    const resolved = resolveInterruptDismissTransition(
      prevChatStateRef.current,
      rawInterrupted,
      hasLive,
      interruptDismissed,
    );

    if (resolved.shouldClearDismiss) {
      clearInterruptDismiss(currentUser.id, workspaceId);
      setInterruptDismissed(false);
    }

    prevChatStateRef.current = resolved.nextState;
  }, [
    interruptedTask,
    interruptedConversationIds,
    activeTask,
    conversations,
    workspaceId,
    currentUser.id,
    interruptDismissed,
  ]);

  // Notify parent of live/interrupted conversation state for workspace picker indicators
  // Reports effective interrupted state (false when dismissed) so workspace picker reflects dismissals
  useEffect(() => {
    if (!onConversationStateChange) return;
    const rawInterrupted = Boolean(interruptedTask) || interruptedConversationIds.size > 0;
    const hasInterrupted = rawInterrupted && !interruptDismissed;
    // Don't count conversations with stale active_task_ids as live
    const hasLiveTask = Boolean(activeTask)
      || conversations.some(c => Boolean(c.active_task_id));
    // Report live and interrupted independently so the workspace picker can
    // show the spinner even when another conversation is interrupted.
    onConversationStateChange(hasLiveTask, hasInterrupted);
  }, [activeTask, conversations, interruptedTask, interruptedConversationIds, interruptDismissed, onConversationStateChange]);

  const lastSeenVersionRef = useRef<number>(0);  // Track last seen version for delta polling
  // Available models from shared context
  const {
    models: availableModels,
    loading: modelsLoading,
    error: modelsError,
    readiness: modelsReadiness,
    refresh: refreshModels,
  } = useAvailableModels();
  const [isWorkspaceConversationMenuOpen, setIsWorkspaceConversationMenuOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [conversationMembers, setConversationMembers] = useState<ConversationMember[]>([]);
  const [conversationToolIds, setConversationToolIds] = useState<string[]>([]);
  const [conversationToolGroupIds, setConversationToolGroupIds] = useState<string[]>([]);
  const [availableTools, setAvailableTools] = useState<UserSpaceAvailableTool[]>([]);
  const [toolGroups, setToolGroups] = useState<ToolGroupInfo[]>([]);
  const [showMembersModal, setShowMembersModal] = useState(false);
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [savingMembers, setSavingMembers] = useState(false);
  const [savingTools, setSavingTools] = useState(false);
  const [isConversationListLoading, setIsConversationListLoading] = useState(true);
  const [showPromptDebugModal, setShowPromptDebugModal] = useState(false);
  const [promptDebugRecords, setPromptDebugRecords] = useState<ProviderPromptDebugRecord[]>([]);
  const [promptDebugLoading, setPromptDebugLoading] = useState(false);
  const [promptDebugError, setPromptDebugError] = useState<string | null>(null);
  const [copiedPromptMessageKey, setCopiedPromptMessageKey] = useState<string | null>(null);
  const [promptDebugMessageIndex, setPromptDebugMessageIndex] = useState<number | null>(null);

  const useWorkspaceToolSource = Boolean(
    embedded
    && workspaceId
    && workspaceAvailableTools
    && workspaceSelectedToolIds
    && onToggleWorkspaceTool,
  );

  const effectiveAvailableTools = useWorkspaceToolSource
    ? (workspaceAvailableTools ?? [])
    : availableTools;
  const effectiveToolIds = useWorkspaceToolSource
    ? (workspaceSelectedToolIds ?? [])
    : conversationToolIds;
  const resolvedConversationToolIds = useMemo(
    () => conversationToolIds,
    [conversationToolIds]
  );
  const resolvedEffectiveToolIds = useMemo(
    () => effectiveToolIds,
    [effectiveToolIds]
  );
  const resolvedConversationToolIdSet = useMemo(
    () => new Set(resolvedConversationToolIds),
    [resolvedConversationToolIds]
  );
  const resolvedEffectiveToolIdSet = useMemo(
    () => new Set(resolvedEffectiveToolIds),
    [resolvedEffectiveToolIds]
  );
  const effectiveSavingTools = useWorkspaceToolSource ? workspaceSavingTools : savingTools;

  const effectiveToolGroupIds = useWorkspaceToolSource
    ? (workspaceSelectedToolGroupIds ?? [])
    : conversationToolGroupIds;
  const effectiveToolGroups = useWorkspaceToolSource
    ? (workspaceToolGroups ?? [])
    : toolGroups;
  const conversationToolGroupIdSet = useMemo(
    () => new Set(conversationToolGroupIds),
    [conversationToolGroupIds]
  );
  const effectiveToolGroupIdSet = useMemo(
    () => new Set(effectiveToolGroupIds),
    [effectiveToolGroupIds]
  );

  // Computed conversation ownership and permissions
  const conversationOwnerId = useMemo(() => {
    if (!activeConversation) return null;
    const ownerMember = conversationMembers.find(m => m.role === 'owner');
    return ownerMember?.user_id || activeConversation.user_id || null;
  }, [activeConversation, conversationMembers]);

  const myConversationRole = useMemo(() => {
    if (!activeConversation || !currentUser) return null;
    const myMember = conversationMembers.find(m => m.user_id === currentUser.id);
    return myMember?.role || null;
  }, [activeConversation, currentUser, conversationMembers]);

  const isConversationOwner = myConversationRole === 'owner' || activeConversation?.user_id === currentUser?.id;
  const isConversationViewer = myConversationRole === 'viewer';
  const hasWorkspaceChatCollaboration = Boolean(workspaceId);
  const canManageConversationMembers = Boolean(activeConversation) && (hasWorkspaceChatCollaboration || isConversationOwner);
  const canUseConversationTools = Boolean(activeConversation) && !isReadOnly && (hasWorkspaceChatCollaboration || !isConversationViewer);
  const showPromptDebugButton = Boolean(debugMode && isAdmin && activeConversation);

  const toggleFullscreen = useCallback(() => {
    const next = !isFullscreen;
    setIsFullscreen(next);
    onFullscreenChange?.(next);
  }, [isFullscreen, onFullscreenChange]);

  const loadPromptDebugRecords = useCallback(async (messageIndex: number | null) => {
    if (!activeConversation || !showPromptDebugButton) return;
    if (messageIndex === null) {
      setPromptDebugRecords([]);
      setPromptDebugError('No assistant message was selected for prompt debug.');
      return;
    }
    setPromptDebugLoading(true);
    setPromptDebugError(null);
    try {
      const records = await api.getConversationProviderDebugPrompts(
        activeConversation.id,
        messageIndex,
        workspaceId,
        200,
      );
      setPromptDebugRecords(records);
    } catch (err) {
      setPromptDebugError(
        err instanceof Error ? err.message : 'Failed to load prompt debug records.'
      );
    } finally {
      setPromptDebugLoading(false);
    }
  }, [activeConversation, showPromptDebugButton, workspaceId]);

  const closePromptDebugModal = useCallback(() => {
    setShowPromptDebugModal(false);
    setPromptDebugMessageIndex(null);
  }, []);

  const openPromptDebugForAssistantMessage = useCallback((messageIndex: number) => {
    if (!activeConversation) return;
    const msg = activeConversation.messages[messageIndex];
    if (!msg || msg.role !== 'assistant') return;
    setPromptDebugMessageIndex(messageIndex);
    setShowPromptDebugModal(true);
  }, [activeConversation]);

  const copyPromptText = useCallback(async (messageKey: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedPromptMessageKey(messageKey);
      window.setTimeout(() => {
        setCopiedPromptMessageKey((current) => (current === messageKey ? null : current));
      }, 1500);
    } catch {
      setPromptDebugError('Failed to copy prompt text to clipboard.');
    }
  }, []);

  const formatPromptMessageContent = useCallback((content: unknown): string => {
    if (typeof content === 'string') return content;

    if (Array.isArray(content)) {
      return content.map((part) => {
        if (!part || typeof part !== 'object') return String(part ?? '');
        const typedPart = part as Record<string, unknown>;
        const partType = typedPart.type;
        if (partType === 'text') return String(typedPart.text ?? '');
        if (partType === 'image_url') return '[image]';
        if (partType === 'file') return `[file: ${String(typedPart.filename ?? typedPart.file_path ?? 'attachment')}]`;
        return JSON.stringify(typedPart, null, 2);
      }).join('\n');
    }

    if (content && typeof content === 'object') {
      return JSON.stringify(content, null, 2);
    }
    return String(content ?? '');
  }, []);

  const chronologicalPromptDebugRecords = useMemo(() => {
    return [...promptDebugRecords].sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at));
  }, [promptDebugRecords]);

  useEffect(() => {
    if (!showPromptDebugModal || promptDebugMessageIndex === null) return;
    void loadPromptDebugRecords(promptDebugMessageIndex);
  }, [showPromptDebugModal, promptDebugMessageIndex, loadPromptDebugRecords]);

  // Image modal state
  const [modalImageUrl, setModalImageUrl] = useState<string | null>(null);

  // Keep latest conversation available to long-lived async callbacks.
  const activeConversationRef = useRef<Conversation | null>(null);
  const streamingEventsRef = useRef<StreamingRenderEvent[]>([]);
  const streamingContentRef = useRef('');

  useEffect(() => {
    return () => {
      onFullscreenChange?.(false);
    };
  }, [onFullscreenChange]);

  useEffect(() => {
    activeConversationRef.current = activeConversation;
  }, [activeConversation]);

  useEffect(() => {
    onActiveConversationChange?.(activeConversation?.id ?? null);
  }, [activeConversation?.id, onActiveConversationChange]);

  useEffect(() => {
    streamingEventsRef.current = streamingEvents;
  }, [streamingEvents]);

  useEffect(() => {
    streamingContentRef.current = streamingContent;
  }, [streamingContent]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatMessagesRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const processingTaskRef = useRef<string | null>(null);
  // Tracks which streaming `create_userspace_snapshot` tool_end events have
  // already triggered an `onSnapshotsMaybeChanged` notification, keyed by
  // a stable string per (taskId, eventIndex). Cleared when the task ends.
  const notifiedSnapshotEventKeysRef = useRef<Set<string>>(new Set());
  const titleSourceRef = useRef<Map<string, EventSource>>(new Map());
  const workspaceConversationDropdownRef = useRef<HTMLDivElement>(null);
  const chatMainRef = useRef<HTMLDivElement>(null);
  const selectConversationRequestIdRef = useRef(0);
  const prevSidebarWidth = useRef(280);
  const prevInputAreaHeight = useRef(MIN_INPUT_AREA_HEIGHT);
  const prevInputLengthRef = useRef(0);
  const skipNextLayoutPersistRef = useRef(true);

  useEffect(() => {
    skipNextLayoutPersistRef.current = true;
    const stored = readStoredChatLayout(chatLayoutCookieName);

    if (!stored) {
      setShowSidebar(!embedded);
      setSidebarWidth(280);
      setInputAreaHeight(MIN_INPUT_AREA_HEIGHT);
      setIsInputAreaCollapsed(false);
      setIsMessagesCollapsed(false);
      prevSidebarWidth.current = 280;
      prevInputAreaHeight.current = MIN_INPUT_AREA_HEIGHT;
      return;
    }

    const nextSidebarWidth = clampNumber(stored.sidebarWidth, 0, 480);
    const nextInputAreaHeight = Math.max(MIN_INPUT_AREA_HEIGHT, stored.inputAreaHeight);
    const nextIsInputAreaCollapsed = stored.isInputAreaCollapsed;
    const nextIsMessagesCollapsed = nextIsInputAreaCollapsed ? false : stored.isMessagesCollapsed;

    setShowSidebar(embedded ? false : stored.showSidebar);
    setSidebarWidth(nextSidebarWidth);
    setInputAreaHeight(nextInputAreaHeight);
    setIsInputAreaCollapsed(nextIsInputAreaCollapsed);
    setIsMessagesCollapsed(nextIsMessagesCollapsed);
    if (nextInputAreaHeight > MIN_INPUT_AREA_HEIGHT) {
      setIsManualResize(true);
    }

    if (nextSidebarWidth >= 120) {
      prevSidebarWidth.current = nextSidebarWidth;
    }
    prevInputAreaHeight.current = nextInputAreaHeight;
  }, [MIN_INPUT_AREA_HEIGHT, chatLayoutCookieName, embedded]);

  useEffect(() => {
    if (embedded || typeof window === 'undefined') {
      return;
    }

    const mediaQuery = window.matchMedia('(max-width: 768px)');

    const collapseOnMobile = (matches: boolean) => {
      if (matches) {
        setShowSidebar(false);
      }
    };

    collapseOnMobile(mediaQuery.matches);

    const listener = (event: MediaQueryListEvent) => {
      collapseOnMobile(event.matches);
    };

    if (typeof mediaQuery.addEventListener === 'function') {
      mediaQuery.addEventListener('change', listener);
      return () => mediaQuery.removeEventListener('change', listener);
    }

    mediaQuery.addListener(listener);
    return () => mediaQuery.removeListener(listener);
  }, [embedded]);

  useEffect(() => {
    if (skipNextLayoutPersistRef.current) {
      skipNextLayoutPersistRef.current = false;
      return;
    }

    const persisted: StoredChatLayout = {
      showSidebar: embedded ? false : showSidebar,
      sidebarWidth: clampNumber(sidebarWidth, 0, 480),
      inputAreaHeight: Math.max(MIN_INPUT_AREA_HEIGHT, inputAreaHeight),
      isInputAreaCollapsed,
      isMessagesCollapsed: isInputAreaCollapsed ? false : isMessagesCollapsed,
    };

    setSessionCookieValue(chatLayoutCookieName, JSON.stringify(persisted));
  }, [
    MIN_INPUT_AREA_HEIGHT,
    chatLayoutCookieName,
    embedded,
    inputAreaHeight,
    isInputAreaCollapsed,
    isMessagesCollapsed,
    showSidebar,
    sidebarWidth,
  ]);

  const handleResizeSidebar = useCallback((delta: number) => {
    if (embedded) return;
    setSidebarWidth((prev) => {
      const next = Math.min(480, Math.max(0, prev + delta));
      if (next < 120) {
        if (prev >= 120) prevSidebarWidth.current = prev;
        setShowSidebar(false);
        return prevSidebarWidth.current || prev || 280;
      }
      prevSidebarWidth.current = next;
      return next;
    });
  }, [embedded]);

  const expandSidebar = useCallback(() => {
    setShowSidebar(true);
    const restored = prevSidebarWidth.current || 280;
    setSidebarWidth(Math.min(480, Math.max(180, restored)));
  }, []);

  const getMaxInputAreaHeight = useCallback(() => {
    const chatMain = chatMainRef.current;
    if (!chatMain) return 600;

    const containerHeight = chatMain.clientHeight;
    let occupiedHeight = 0;

    for (const child of Array.from(chatMain.children)) {
      const el = child as HTMLElement;
      if (el.classList.contains('chat-input-area')) continue;
      if (el.classList.contains('chat-messages')) continue;
      occupiedHeight += el.getBoundingClientRect().height;
    }

    return Math.max(MIN_INPUT_AREA_HEIGHT, containerHeight - occupiedHeight);
  }, [MIN_INPUT_AREA_HEIGHT]);

  const handleResizeInputArea = useCallback((delta: number) => {
    // Switch to manual-resize mode — textarea fills via CSS, container height
    // is drag-controlled.
    setIsManualResize(true);
    const inputArea = chatMainRef.current?.querySelector('.chat-input-area') as HTMLElement | null;
    if (inputArea) {
      const ta = inputArea.querySelector('.chat-input') as HTMLElement | null;
      if (ta) {
        ta.style.height = '';
        ta.style.maxHeight = '';
      }
    }

    setInputAreaHeight((prev) => {
      const proposed = prev - delta;
      const draggingDown = delta > 0;
      const draggingUp = delta < 0;
      const atMinHeight = prev <= MIN_INPUT_AREA_HEIGHT;
      const crossedCollapseThreshold = proposed < INPUT_AREA_COLLAPSE_THRESHOLD;

      // --- Collapse input area (dragging down) ---
      if (draggingDown && (atMinHeight || crossedCollapseThreshold)) {
        if (!isInputAreaCollapsed && prev > MIN_INPUT_AREA_HEIGHT) {
          prevInputAreaHeight.current = prev;
        }
        setIsInputAreaCollapsed(true);
        if (isMessagesCollapsed) setIsMessagesCollapsed(false);
        setIsManualResize(false);
        return prev;
      }

      // --- Compute max input height from container ---
      const maxInputHeight = getMaxInputAreaHeight();

      // --- Collapse messages area (dragging up past max) ---
      if (draggingUp && proposed >= maxInputHeight) {
        if (!isMessagesCollapsed) {
          prevInputAreaHeight.current = prev;
          setIsMessagesCollapsed(true);
        }
        setIsManualResize(false);
        return maxInputHeight;
      }

      const next = Math.min(maxInputHeight, Math.max(MIN_INPUT_AREA_HEIGHT, proposed));
      prevInputAreaHeight.current = next;
      if (isInputAreaCollapsed) {
        setIsInputAreaCollapsed(false);
      }
      if (isMessagesCollapsed) {
        setIsMessagesCollapsed(false);
      }
      return next;
    });
  }, [INPUT_AREA_COLLAPSE_THRESHOLD, MIN_INPUT_AREA_HEIGHT, getMaxInputAreaHeight, isInputAreaCollapsed, isMessagesCollapsed]);

  const expandInputArea = useCallback(() => {
    setIsInputAreaCollapsed(false);
    const nextHeight = Math.max(MIN_INPUT_AREA_HEIGHT, prevInputAreaHeight.current || MIN_INPUT_AREA_HEIGHT);
    setInputAreaHeight(nextHeight);
    if (nextHeight > MIN_INPUT_AREA_HEIGHT) {
      setIsManualResize(true);
    } else {
      setIsManualResize(false);
    }
    requestAnimationFrame(() => inputRef.current?.focus());
  }, [MIN_INPUT_AREA_HEIGHT]);

  const expandMessages = useCallback(() => {
    setIsMessagesCollapsed(false);
    const nextHeight = Math.max(MIN_INPUT_AREA_HEIGHT, prevInputAreaHeight.current || MIN_INPUT_AREA_HEIGHT);
    setInputAreaHeight(nextHeight);
    if (nextHeight > MIN_INPUT_AREA_HEIGHT) {
      setIsManualResize(true);
    } else {
      setIsManualResize(false);
    }
  }, [MIN_INPUT_AREA_HEIGHT]);

  // Adjust the input area container height to fit the textarea content.
  // The textarea itself always fills its container via CSS (height: 100%);
  // this function only manages the container height.
  const autoResizeInput = useCallback((el?: HTMLTextAreaElement | null) => {
    const textarea = el ?? inputRef.current;
    if (!textarea) return;

    if (isManualResize) {
      if (textarea.value === '' && prevInputLengthRef.current > 0) {
        setIsManualResize(false);
        setInputAreaHeight(MIN_INPUT_AREA_HEIGHT);
      }
      prevInputLengthRef.current = textarea.value.length;
      return;
    }

    prevInputLengthRef.current = textarea.value.length;

    const wrapper = textarea.closest('.chat-input-area') as HTMLElement | null;

    let wrapperOverhead = 0;
    if (wrapper) {
      const wrapperStyle = getComputedStyle(wrapper);
      const verticalPadding = parseFloat(wrapperStyle.paddingTop) + parseFloat(wrapperStyle.paddingBottom);
      const borderWidth = parseFloat(wrapperStyle.borderTopWidth) + parseFloat(wrapperStyle.borderBottomWidth);
      wrapperOverhead = verticalPadding + borderWidth;
    }

    // Measure intrinsic content height by collapsing textarea to 0.
    // The inline style overrides the CSS height:100% during measurement.
    textarea.style.height = '0px';
    const contentHeight = textarea.scrollHeight;
    // Clear inline height — let CSS height:100% fill the container.
    textarea.style.height = '';

    // Content-driven auto-resize: grow or shrink container to fit.
    const rawMax = getMaxInputAreaHeight();
    const maxInputHeight = embedded
      ? rawMax
      : Math.min(rawMax, Math.floor(window.innerHeight * 0.5));

    const needed = contentHeight + wrapperOverhead;
    const target = Math.min(maxInputHeight, Math.max(MIN_INPUT_AREA_HEIGHT, Math.ceil(needed)));

    setInputAreaHeight((prev) => {
      if (prev === target) return prev;
      // Enable transition classes for content-driven resize only
      const shrinking = target < prev;
      setAutoResizeState(shrinking ? 'shrinking' : 'growing');

      const cleanup = () => {
        setAutoResizeState(null);
        wrapper?.removeEventListener('transitionend', cleanup);
      };
      wrapper?.addEventListener('transitionend', cleanup, { once: true });
      // Fallback removal in case transitionend doesn't fire
      setTimeout(cleanup, shrinking ? 80 : 180);

      return target;
    });
  }, [MIN_INPUT_AREA_HEIGHT, embedded, getMaxInputAreaHeight, isManualResize]);

  // Cover programmatic value changes: clearing after send, loading a conversation, etc.
  useEffect(() => {
    autoResizeInput();
  }, [autoResizeInput, inputValue]);

  // onChange handler: resize synchronously — e.target.scrollHeight is already
  // correct for the new value (including pasted text) by the time onChange fires.
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
    autoResizeInput(e.target);
  }, [autoResizeInput]);

  useEffect(() => {
    if (!isWorkspaceConversationMenuOpen) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (
        workspaceConversationDropdownRef.current &&
        !workspaceConversationDropdownRef.current.contains(event.target as Node)
      ) {
        setIsWorkspaceConversationMenuOpen(false);
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsWorkspaceConversationMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isWorkspaceConversationMenuOpen]);

  useEffect(() => {
    setIsWorkspaceConversationMenuOpen(false);
  }, [activeConversation?.id, workspaceId, embedded]);



  const getOwnerKey = useCallback((conv: Conversation) => conv.username || conv.user_id || 'unknown', []);

  const getOwnerLabel = useCallback(
    (conv: Conversation) => conv.display_name || conv.username || 'Unknown user',
    [],
  );

  const defaultContextLimit = useMemo(() => {
    const maxAvailableLimit = availableModels.reduce((max, model) => {
      const limit = Number(model.context_limit || 0);
      return Number.isFinite(limit) ? Math.max(max, limit) : max;
    }, 0);
    return Math.max(DEFAULT_CONTEXT_LIMIT, maxAvailableLimit);
  }, [availableModels]);

  const groupedConversations = useMemo(() => {
    if (!isAdmin) return [] as Array<{ key: string; label: string; conversations: Conversation[]; isCurrentUserGroup: boolean }>;

    const groups = conversations.reduce<Record<string, { label: string; conversations: Conversation[]; isCurrentUserGroup: boolean }>>((acc, conv) => {
      const key = getOwnerKey(conv);
      const label = getOwnerLabel(conv);
      const existing = acc[key];
      acc[key] = existing
        ? {
            label: existing.label || label,
            conversations: [...existing.conversations, conv],
            isCurrentUserGroup: existing.isCurrentUserGroup || conv.user_id === currentUser.id,
          }
        : {
            label,
            conversations: [conv],
            isCurrentUserGroup: conv.user_id === currentUser.id,
          };
      return acc;
    }, {});

    return Object.entries(groups)
      .map(([key, value]) => ({
        key,
        label: value.label,
        conversations: value.conversations,
        isCurrentUserGroup: value.isCurrentUserGroup,
      }))
      .sort((a, b) => {
        if (a.isCurrentUserGroup !== b.isCurrentUserGroup) {
          return a.isCurrentUserGroup ? -1 : 1;
        }
        return a.label.localeCompare(b.label);
      });
  }, [conversations, currentUser.id, getOwnerKey, getOwnerLabel, isAdmin]);

  useEffect(() => {
    if (!isAdmin) {
      if (Object.keys(collapsedGroups).length > 0) {
        setCollapsedGroups({});
      }
      return;
    }

    setCollapsedGroups(prev => {
      const next = { ...prev };
      let changed = false;

      conversations.forEach(conv => {
        const key = getOwnerKey(conv);
        if (!(key in next)) {
          next[key] = true; // collapsed by default
          changed = true;
        }
      });

      return changed ? next : prev;
    });
  }, [collapsedGroups, conversations, getOwnerKey, isAdmin]);

  const toggleGroup = useCallback((key: string) => {
    setCollapsedGroups(prev => ({ ...prev, [key]: !prev[key] }));
  }, []);

  // Preserve partial assistant output locally when server state lags behind stream completion/cancel.
  const buildFallbackAssistantFromStreaming = useCallback((): ChatMessage | null => {
    const sourceEvents = streamingEventsRef.current;
    const sourceContent = streamingContentRef.current;

    if (sourceEvents.length === 0 && !sourceContent.trim()) return null;

    const events: MessageEvent[] = [];
    let textContent = '';

    for (const ev of sourceEvents) {
      if (ev.type === 'content') {
        textContent += ev.content;
        events.push({ type: 'content', content: ev.content });
      } else if (ev.type === 'reasoning') {
        events.push({ type: 'reasoning', content: ev.content });
      } else if (ev.type === 'tool') {
        events.push({
          type: 'tool',
          tool: ev.toolCall.tool,
          input: ev.toolCall.input,
          output: ev.toolCall.output,
          presentation: ev.toolCall.presentation,
          connection: ev.toolCall.connection,
        });
      }
    }

    const content = textContent || sourceContent;
    if (!content.trim() && events.length === 0) return null;

    return {
      role: 'assistant',
      content,
      timestamp: new Date().toISOString(),
      events: events.length > 0 ? events : undefined,
    };
  }, []);

  const applyFallbackAssistantIfNeeded = useCallback((conversation: Conversation): Conversation => {
    const fallback = buildFallbackAssistantFromStreaming();
    if (!fallback) return conversation;

    const lastMessage = conversation.messages[conversation.messages.length - 1];
    if (lastMessage?.role === 'assistant') return conversation;

    return {
      ...conversation,
      messages: [...conversation.messages, fallback],
    };
  }, [buildFallbackAssistantFromStreaming]);

  // Live notification: as the agent stream lands a completed
  // `create_userspace_snapshot` tool call, tell the parent so the snapshots
  // panel can refresh in real time. Dedup per (taskId, eventIndex) so we
  // only fire once per snapshot, even across re-renders.
  useEffect(() => {
    if (!onSnapshotsMaybeChanged) return;
    if (!streamingEvents.length) return;
    const taskId = processingTaskRef.current ?? activeTask?.id ?? 'idle';
    let fired = false;
    streamingEvents.forEach((ev, idx) => {
      if (ev.type !== 'tool') return;
      if (ev.toolCall.tool !== 'create_userspace_snapshot') return;
      if (ev.toolCall.status !== 'complete') return;
      const key = `${taskId}:${idx}`;
      if (notifiedSnapshotEventKeysRef.current.has(key)) return;
      notifiedSnapshotEventKeysRef.current.add(key);
      fired = true;
    });
    if (fired) {
      try {
        onSnapshotsMaybeChanged();
      } catch (err) {
        console.warn('onSnapshotsMaybeChanged threw:', err);
      }
    }
  }, [streamingEvents, onSnapshotsMaybeChanged, activeTask?.id]);

  // Reset the dedup set whenever the active task changes (or clears) so a
  // new turn starts fresh.
  useEffect(() => {
    notifiedSnapshotEventKeysRef.current.clear();
  }, [activeTask?.id]);

  // Memoized consolidated segments for streaming - groups adjacent content events
  // Merges ALL reasoning events into a single segment per turn for unified display
  const consolidatedSegments = useMemo((): StreamingSegment[] => {
    if (!streamingEvents.length) return [];

    const segments: StreamingSegment[] = [];
    let currentContent = '';
    let currentReasoning = '';
    let currentReasoningParts: ReasoningPart[] = [];
    let reasoningToolCalls: ActiveToolCall[] = [];
    let currentReasoningDurationSeconds: number | undefined;

    // Flush accumulated reasoning into a NEW reasoning segment (adjacent reasoning merges, non-adjacent stays separate)
    const flushReasoning = (isComplete: boolean) => {
      if (!currentReasoning) return;
      const seg: StreamingSegment = {
        type: 'reasoning',
        content: currentReasoning,
        isComplete,
        durationSeconds: currentReasoningDurationSeconds,
        embeddedToolCalls: reasoningToolCalls.length > 0 ? [...reasoningToolCalls] : [],
        reasoningParts: currentReasoningParts.length > 0 ? [...currentReasoningParts] : [{ type: 'text', text: currentReasoning }],
      };
      segments.push(seg);
      currentReasoning = '';
      currentReasoningParts = [];
      reasoningToolCalls = [];
      currentReasoningDurationSeconds = undefined;
    };

    const flushContent = () => {
      if (!currentContent) return;
      segments.push({ type: 'content', content: currentContent });
      currentContent = '';
    };

    for (const ev of streamingEvents) {
      if (ev.type === 'reasoning') {
        // Flush any pending content first — content breaks reasoning adjacency
        flushContent();
        // Accumulate reasoning (adjacent reasoning events merge)
        currentReasoning += ev.content;
        if (typeof ev.durationSeconds === 'number') {
          currentReasoningDurationSeconds = ev.durationSeconds;
        }
        const lastPart = currentReasoningParts[currentReasoningParts.length - 1];
        if (lastPart && lastPart.type === 'text') {
          lastPart.text = (lastPart.text || '') + ev.content;
        } else {
          currentReasoningParts.push({ type: 'text', text: ev.content });
        }
      } else if (ev.type === 'content') {
        // Flush any pending reasoning — it's now complete since content follows
        flushReasoning(true);
        // Accumulate content
        currentContent += ev.content;
      } else if (ev.type === 'tool') {
        if (currentReasoning) {
          // Tool arrived while reasoning is actively pending — embed in reasoning
          reasoningToolCalls.push(ev.toolCall);
          currentReasoningParts.push({ type: 'tool', toolCall: ev.toolCall });
        } else {
          // Reasoning is not pending — render as standalone tool
          flushContent();
          segments.push({ type: 'tool', toolCall: ev.toolCall });
        }
      }
    }

    // Flush remaining reasoning (still streaming, not yet complete)
    flushReasoning(false);
    // Flush remaining content
    flushContent();

    return segments;
  }, [streamingEvents]);

  // Save showToolCalls preference to localStorage
  useEffect(() => {
    localStorage.setItem('chat-show-tool-calls', showToolCalls.toString());
  }, [showToolCalls]);

  // Refresh available models once on mount (stable trigger, independent of model-fetch identity)
  useEffect(() => {
    refreshModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reset state and load conversations when workspace changes (or on initial mount)
  useEffect(() => {
    setConversations([]);
    setActiveConversation(null);
    setIsConversationSwitchLoading(false);
    if (!workspaceId) {
      loadConversations();
    } else {
      // Workspace mode: set loading until applyWorkspaceChatState clears it
      setIsConversationListLoading(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceId]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (!shouldAutoScrollRef.current || !chatMessagesRef.current) return;

    chatMessagesRef.current.scrollTo({
      top: chatMessagesRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [activeConversation?.messages, streamingContent, consolidatedSegments]);

  const handleScroll = useCallback(() => {
    if (!chatMessagesRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = chatMessagesRef.current;
    // Use a small threshold to account for fractional pixels
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    shouldAutoScrollRef.current = isAtBottom;
  }, []);

  // Focus input when conversation changes
  useEffect(() => {
    inputRef.current?.focus();
  }, [activeConversation?.id]);

  const applyWorkspaceChatState = useCallback((nextWorkspaceState: WorkspaceChatStateResponse) => {
    const visibleConversations = nextWorkspaceState.conversations;

    setConversations((prev) => {
      let changed = prev.length !== visibleConversations.length;

      const prevById = new Map(prev.map((conversation) => [conversation.id, conversation]));
      const next = visibleConversations.map((conversation) => {
        const existing = prevById.get(conversation.id);
        if (!existing) {
          changed = true;
          return conversation;
        }

        if (
          existing.active_task_id !== conversation.active_task_id
          || existing.title !== conversation.title
          || existing.model !== conversation.model
        ) {
          changed = true;
          return { ...existing, ...conversation };
        }

        return existing;
      });

      return changed ? next : prev;
    });

    setActiveConversation((current) => {
      const targetConversationId = nextWorkspaceState.selected_conversation_id ?? current?.id ?? null;
      if (!targetConversationId) {
        return visibleConversations[0] ?? null;
      }
      const matchingConversation = visibleConversations.find((conversation) => conversation.id === targetConversationId);
      if (current && matchingConversation && current.id === matchingConversation.id) {
        return mergeConversationFromWorkspaceSnapshot(current, matchingConversation);
      }
      return matchingConversation ?? visibleConversations[0] ?? null;
    });

    const nextInterruptedIds = new Set<string>(nextWorkspaceState.interrupted_conversation_ids);
    setInterruptedConversationIds((prev) => {
      if (prev.size !== nextInterruptedIds.size) return nextInterruptedIds;
      for (const id of nextInterruptedIds) {
        if (!prev.has(id)) return nextInterruptedIds;
      }
      return prev;
    });

    const selectedConversationId = nextWorkspaceState.selected_conversation_id;
    if (!selectedConversationId) {
      setActiveTask(null);
      setInterruptedTask(null);
      setIsConversationListLoading(false);
      return;
    }

    const activeT = nextWorkspaceState.active_task;
    const interruptedT = nextWorkspaceState.interrupted_task;
    if (activeT && (activeT.status === 'pending' || activeT.status === 'running')) {
      setActiveTask(activeT);
      setInterruptedTask(null);
      syncConversationActiveTaskId(selectedConversationId, activeT.id);
      return;
    }

    setActiveTask(null);
    setInterruptedTask(interruptedT ?? null);
    syncConversationActiveTaskId(selectedConversationId, null);
    setIsConversationListLoading(false);
  }, [syncConversationActiveTaskId]);

  const loadConversations = async () => {
    setIsConversationListLoading(true);
    try {
      const workspaceState = workspaceId
        ? (workspaceChatState ?? await api.getWorkspaceChatState(workspaceId, activeConversationRef.current?.id ?? null))
        : null;
      const [data, workspacePage] = await Promise.all([
        workspaceState?.conversations
          ? Promise.resolve(workspaceState.conversations)
          : api.listConversations(workspaceId),
        !workspaceId
          ? api.listUserSpaceWorkspaces(0, 200).catch((workspaceErr) => {
              console.warn('Failed to load userspace workspaces for conversation filtering:', workspaceErr);
              return null;
            })
          : Promise.resolve(null),
      ]);
      let userspaceConversationIds = new Set<string>();

      const getLinkedWorkspaceId = (conversation: Conversation): string | null => {
        const camelWorkspaceId = (conversation as Conversation & { workspaceId?: string | null }).workspaceId;
        return conversation.workspace_id ?? camelWorkspaceId ?? null;
      };

      if (workspacePage) {
        userspaceConversationIds = new Set(
          workspacePage.items.flatMap((workspace) => workspace.conversation_ids || [])
        );
      }

      const visibleConversations = data.filter((conversation) => {
        const linkedWorkspaceId = getLinkedWorkspaceId(conversation);
        if (workspaceId) {
          return linkedWorkspaceId === workspaceId;
        }
        return !linkedWorkspaceId && !userspaceConversationIds.has(conversation.id);
      });

      setConversations(visibleConversations);
      setActiveConversation((current) => {
        if (!current) {
          return visibleConversations[0] ?? null;
        }
        const matchingConversation = visibleConversations.find((conversation) => conversation.id === current.id);
        return matchingConversation ?? visibleConversations[0] ?? null;
      });

      if (workspaceState) {
        applyWorkspaceChatState(workspaceState);
      }
    } catch (err) {
      console.error('Failed to load conversations:', err);
    } finally {
      setIsConversationListLoading(false);
    }
  };

  useEffect(() => {
    if (!workspaceId || !workspaceChatState) return;
    applyWorkspaceChatState(workspaceChatState);
  }, [applyWorkspaceChatState, workspaceChatState, workspaceId]);

  // Poll workspace conversation summaries so live/attention indicators update without refresh
  useEffect(() => {
    if (!workspaceId || workspaceChatState) return;

    let cancelled = false;
    let pollInProgress = false;

    const pollWorkspaceConversationStates = async () => {
      if (pollInProgress) return;
      pollInProgress = true;
      try {
        const workspaceState = await api.getWorkspaceChatState(
          workspaceId,
          activeConversationRef.current?.id ?? null,
        );
        if (cancelled) return;
        applyWorkspaceChatState(workspaceState);
      } catch (err) {
        console.error('Failed to poll workspace conversation states:', err);
      } finally {
        pollInProgress = false;
      }
    };

    void pollWorkspaceConversationStates();
    const interval = setInterval(() => {
      void pollWorkspaceConversationStates();
    }, 3000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [applyWorkspaceChatState, workspaceChatState, workspaceId, activeConversation?.id]);

  const fetchConversationMembers = useCallback(async (conversationId: string) => {
    try {
      const members = await api.getConversationMembers(conversationId);
      setConversationMembers(members);
    } catch (err) {
      console.error('Failed to fetch conversation members:', err);
      setConversationMembers([]);
    }
  }, []);

  const fetchConversationTools = useCallback(async (conversationId: string) => {
    try {
      const data = await api.getConversationTools(conversationId);
      setConversationToolIds(data.tool_config_ids);
      setConversationToolGroupIds(data.tool_group_ids);
    } catch (err) {
      console.error('Failed to fetch conversation tools:', err);
      setConversationToolIds([]);
      setConversationToolGroupIds([]);
    }
  }, []);

  const refreshBranchPoints = useCallback(async (conversationId: string): Promise<ConversationBranchPointInfo[]> => {
    try {
      const points = await api.getConversationBranchPoints(conversationId, workspaceId);
      setBranchPoints(points);
      return points;
    } catch {
      setBranchPoints([]);
      return [];
    }
  }, [workspaceId]);

  useEffect(() => {
    setBranchPoints([]);
    setBranchSelections({});
    setCopiedMessageIdx(null);
  }, [activeConversationId]);

  const fetchAvailableTools = useCallback(async () => {
    try {
      const [tools, groups] = await Promise.all([
        api.listUserSpaceAvailableTools(),
        api.listUserSpaceToolGroups(),
      ]);
      setAvailableTools(tools);
      setToolGroups(groups.map((g) => ({ id: g.id, name: g.name })));
    } catch (err) {
      console.error('Failed to fetch available tools:', err);
      setAvailableTools([]);
      setToolGroups([]);
    }
  }, []);

  // Load conversation members, tools, and branches when conversation changes
  useEffect(() => {
    if (activeConversationId) {
      if (!useWorkspaceToolSource) {
        void fetchConversationTools(activeConversationId);
      }
      void fetchConversationMembers(activeConversationId);
      void refreshBranchPoints(activeConversationId);
    } else {
      setConversationMembers([]);
      setBranchPoints([]);
    }
  }, [activeConversationId, fetchConversationMembers, fetchConversationTools, useWorkspaceToolSource, refreshBranchPoints]);

  // Load available tools on mount
  useEffect(() => {
    if ((!embedded || Boolean(workspaceId)) && !useWorkspaceToolSource) {
      fetchAvailableTools();
    }
  }, [embedded, fetchAvailableTools, useWorkspaceToolSource, workspaceId]);

  const handleOpenMembersModal = useCallback(async () => {
    if (!activeConversation || !canManageConversationMembers) return;
    try {
      const users = await api.listUsers();
      setAllUsers(users);
    } catch {
      setAllUsers([]);
    }
    setShowMembersModal(true);
  }, [activeConversation, canManageConversationMembers]);

  const handleSaveMembers = useCallback(async (members: ConversationMember[]) => {
    if (!activeConversation) return;
    setSavingMembers(true);
    try {
      await api.updateConversationMembers(activeConversation.id, { members });
      await fetchConversationMembers(activeConversation.id);
      setShowMembersModal(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update members');
      throw err;
    } finally {
      setSavingMembers(false);
    }
  }, [activeConversation, fetchConversationMembers]);

  const persistConversationToolSelection = useCallback(async (
    nextToolIds: string[],
    nextGroupIds: string[],
    fallbackMessage: string,
  ) => {
    if (!activeConversation) return;

    setSavingTools(true);
    try {
      await api.updateConversationTools(activeConversation.id, {
        tool_config_ids: nextToolIds,
        tool_group_ids: nextGroupIds,
      });
      setConversationToolIds(nextToolIds);
      setConversationToolGroupIds(nextGroupIds);
    } catch (err) {
      setError(err instanceof Error ? err.message : fallbackMessage);
    } finally {
      setSavingTools(false);
    }
  }, [activeConversation]);

  const handleToggleConversationTool = useCallback(async (toolId: string) => {
    if (!activeConversation || isConversationViewer) return;
    const targetTool = availableTools.find((tool) => tool.id === toolId);
    const currentGroupIds = new Set(conversationToolGroupIds);

    if (targetTool?.group_id && currentGroupIds.has(targetTool.group_id)) {
      const nextGroupIds = new Set(currentGroupIds);
      nextGroupIds.delete(targetTool.group_id);
      const nextSelected = new Set(
        getGroupToolIds(availableTools, targetTool.group_id).filter((id) => id !== toolId)
      );

      await persistConversationToolSelection(
        Array.from(nextSelected),
        Array.from(nextGroupIds),
        'Failed to update tool selection',
      );
      return;
    }

    const nextSelected = new Set(resolvedConversationToolIds);
    if (nextSelected.has(toolId)) {
      nextSelected.delete(toolId);
    } else {
      nextSelected.add(toolId);
    }

    await persistConversationToolSelection(
      Array.from(nextSelected),
      Array.from(currentGroupIds),
      'Failed to update tool selection',
    );
  }, [activeConversation, availableTools, conversationToolGroupIds, isConversationViewer, persistConversationToolSelection, resolvedConversationToolIds]);

  const handleToggleConversationToolGroup = useCallback(async (groupId: string) => {
    if (!activeConversation || isConversationViewer) return;
    const groupToolIds = getGroupToolIds(availableTools, groupId);
    const nextGroupIds = new Set(conversationToolGroupIds);
    if (nextGroupIds.has(groupId)) {
      nextGroupIds.delete(groupId);
    } else {
      nextGroupIds.add(groupId);
    }
    const nextToolIds = new Set(resolvedConversationToolIds);

    if (!nextGroupIds.has(groupId)) {
      for (const toolId of groupToolIds) {
        nextToolIds.delete(toolId);
      }
    }

    await persistConversationToolSelection(
      Array.from(nextToolIds),
      Array.from(nextGroupIds),
      'Failed to update tool group selection',
    );
  }, [activeConversation, availableTools, conversationToolGroupIds, isConversationViewer, persistConversationToolSelection, resolvedConversationToolIds]);

  const handleToggleInlineTool = useCallback(async (toolId: string) => {
    if (useWorkspaceToolSource && onToggleWorkspaceTool) {
      await onToggleWorkspaceTool(toolId);
      return;
    }
    await handleToggleConversationTool(toolId);
  }, [handleToggleConversationTool, onToggleWorkspaceTool, useWorkspaceToolSource]);

  const handleToggleInlineToolGroup = useCallback(async (groupId: string) => {
    if (useWorkspaceToolSource && onToggleWorkspaceToolGroup) {
      await onToggleWorkspaceToolGroup(groupId);
      return;
    }
    await handleToggleConversationToolGroup(groupId);
  }, [handleToggleConversationToolGroup, onToggleWorkspaceToolGroup, useWorkspaceToolSource]);

  const formatUserLabel = useCallback((user?: Pick<User, 'username' | 'display_name'> | null, fallbackId?: string) => {
    const username = user?.username?.trim() || fallbackId?.trim() || 'unknown';
    const displayName = user?.display_name?.trim();
    if (displayName && displayName !== username) {
      return `${displayName} (@${username})`;
    }
    return `@${username}`;
  }, []);

  // Resolve context limit from stored conversation model value.
  // Handles provider-scoped and legacy model id formats to avoid 8k fallback mismatches.
  const getContextLimit = useCallback((storedModel: string): number => {
    const parsed = parseStoredModelIdentifier(storedModel);
    const modelId = parsed.modelId.trim();
    const provider = parsed.provider?.trim().toLowerCase();

    if (!modelId) {
      return defaultContextLimit;
    }

    const exactProviderMatch = provider
      ? availableModels.find((model) => model.provider.toLowerCase() === provider && model.id === modelId)
      : undefined;
    if (exactProviderMatch) {
      return exactProviderMatch.context_limit;
    }

    const exactModelMatch = availableModels.find((model) => model.id === modelId);
    if (exactModelMatch) {
      return exactModelMatch.context_limit;
    }

    const slashIndex = modelId.indexOf('/');
    if (slashIndex > 0) {
      const inferredProvider = modelId.slice(0, slashIndex).toLowerCase();
      const providerModelId = modelId.slice(slashIndex + 1);

      const providerScopedMatch = availableModels.find((model) =>
        model.id === providerModelId && (provider ? model.provider.toLowerCase() === provider : model.provider.toLowerCase() === inferredProvider)
      );
      if (providerScopedMatch) {
        return providerScopedMatch.context_limit;
      }

      const unscopedMatch = availableModels.find((model) => model.id === providerModelId);
      if (unscopedMatch) {
        return unscopedMatch.context_limit;
      }
    }

    return defaultContextLimit;
  }, [availableModels, defaultContextLimit]);

  const sendReadinessBlockReason = useMemo(() => {
    if (!activeConversation) {
      return null;
    }

    const parsed = parseStoredModelIdentifier(activeConversation.model || '');
    const modelId = parsed.modelId.trim();
    const explicitProvider = normalizeProviderAlias(parsed.provider);

    let matchedModel = undefined;
    if (modelId) {
      if (explicitProvider) {
        matchedModel = availableModels.find(
          (model) => providersEquivalent(model.provider, explicitProvider) && model.id === modelId,
        );
      }
      if (!matchedModel) {
        matchedModel = availableModels.find((model) => model.id === modelId);
      }
      if (!matchedModel && modelId.includes('/')) {
        const slashIndex = modelId.indexOf('/');
        const inferredProvider = normalizeProviderAlias(modelId.slice(0, slashIndex));
        const providerModelId = modelId.slice(slashIndex + 1);
        matchedModel = availableModels.find((model) =>
          model.id === providerModelId
          && providersEquivalent(model.provider, explicitProvider || inferredProvider),
        );
      }
    }

    const inferredProvider = inferProviderFromModelId(modelId);
    const resolvedProvider = normalizeProviderAlias(
      matchedModel?.provider || explicitProvider || inferredProvider,
    ) || null;
    const providerState = findProviderState(modelsReadiness?.provider_states, resolvedProvider);

    const hasKnownActiveModel = Boolean(matchedModel);
    const hasProviderFailure = Boolean(
      providerState
      && providerState.configured
      && !providerState.loading
      && (!providerState.connected || !providerState.available),
    );
    if (hasProviderFailure) {
      return providerState?.error || 'Selected model provider is disconnected.';
    }

    if (providerState && !providerState.configured && resolvedProvider) {
      return 'Selected model provider is not configured.';
    }

    if (modelsLoading && !hasKnownActiveModel) {
      return 'Loading available models...';
    }

    if (
      resolvedProvider === 'github_copilot'
      && modelsReadiness?.copilot_refresh_in_progress
      && !hasKnownActiveModel
    ) {
      return 'Refreshing GitHub Copilot credentials...';
    }

    if (!hasKnownActiveModel && modelsError) {
      return 'Failed to load available models. Please refresh and try again.';
    }

    if (!hasKnownActiveModel && !modelsLoading && resolvedProvider) {
      return 'Selected model is not available from the configured provider.';
    }

    return null;
  }, [activeConversation, availableModels, modelsError, modelsLoading, modelsReadiness]);

  const applyCreatedConversation = useCallback((conversation: Conversation) => {
    setConversations(prev => [conversation, ...prev]);
    setActiveConversation(conversation);
    setConversationToolIds([]);
    setConversationToolGroupIds([]);
  }, []);

  const createNewConversation = async () => {
    if (isReadOnly) return;
    try {
      shouldAutoScrollRef.current = true;
      const conversation = await api.createConversation(undefined, workspaceId);
      applyCreatedConversation(conversation);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create conversation');
    }
  };


  // Listen for auto-generated titles per conversation using SSE.
  // IMPORTANT: We only open ONE SSE connection for the active conversation to
  // avoid exhausting the browser's HTTP/1.1 connection limit (6 per origin).
  // Opening SSE streams for every untitled conversation would saturate the
  // connection pool and lock up the UI.
  const stopTitleStreamFor = useCallback((conversationId: string) => {
    const es = titleSourceRef.current.get(conversationId);
    if (es) {
      es.close();
    }
    titleSourceRef.current.delete(conversationId);
  }, []);

  const startTitleStreamFor = useCallback((conversationId: string, title: string) => {
    if (titleSourceRef.current.has(conversationId)) return;

    if (title !== 'Untitled Chat') return;

    try {
      const url = api.getConversationEventsUrl(conversationId, workspaceId);
      const es = new EventSource(url, { withCredentials: true });

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'title_update' && data.title) {
            // Update conversation title
            setConversations(prev => prev.map(c => {
               if (c.id === conversationId) {
                 return { ...c, title: data.title };
               }
               return c;
            }));

            // Also update active conversation if it matches
            setActiveConversation(prev => {
              if (prev && prev.id === conversationId) {
                return { ...prev, title: data.title };
              }
              return prev;
            });

            // Close stream after receiving title
            es.close();
            titleSourceRef.current.delete(conversationId);
          }
        } catch (e) {
          console.error("Failed to parse title update", e);
        }
      };

      es.onerror = () => {
        es.close();
        titleSourceRef.current.delete(conversationId);
      };

      titleSourceRef.current.set(conversationId, es);
    } catch (e) {
      console.error("Failed to start title stream", e);
    }
  }, [workspaceId]);

  // Only subscribe to title events for the ACTIVE conversation to avoid
  // saturating the browser's 6-connection-per-origin limit with idle SSE streams.
  useEffect(() => {
    const activeId = activeConversation?.id;

    // Close streams for any conversation that is NOT the active one
    titleSourceRef.current.forEach((_, id) => {
      if (id !== activeId) {
        stopTitleStreamFor(id);
      }
    });

    // Open a stream for the active conversation if it still needs a title
    if (activeId && activeConversation?.title) {
      startTitleStreamFor(activeId, activeConversation.title);
    }

    return () => {
      titleSourceRef.current.forEach(es => es.close());
      titleSourceRef.current.clear();
    };
  }, [
    activeConversation?.id,
    activeConversation?.title,
    startTitleStreamFor,
    stopTitleStreamFor,
  ]);

  // =========================================================================
  // Background Task Streaming (Resume & Reconnect)
  // =========================================================================

  const stopTaskStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    // Also clear processing ref so we can reconnect if needed
    processingTaskRef.current = null;
    setIsPollingTask(false);
    lastSeenVersionRef.current = 0;
  }, []);

  const clearActiveStreamingUi = useCallback(() => {
    stopTaskStreaming();
    setActiveTask(null);
    setIsStreaming(false);
    setStreamingContent('');
    setStreamingEvents([]);
  }, [stopTaskStreaming]);

  const connectTaskStream = useCallback(async (taskId: string) => {
    // Prevent duplicate connection for same task
    if (processingTaskRef.current === taskId) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    processingTaskRef.current = taskId;
    setIsPollingTask(true);
    setIsStreaming(true);

    // Create new abort controller for this stream
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    lastSeenVersionRef.current = 0;

    try {
      const stream = api.streamChatTask(taskId, 0, abortController.signal, workspaceId);

      for await (const data of stream) {
        // Handle explicit completion event
        if (data.type === 'completion' || data.completed) {
            const status = data.status || (data.completed ? 'completed' : 'unknown');
            if (status === 'failed' && data.error) {
                setError(data.error);
            }
            break; // Exit loop, cleanup below
        }

        // Handle streaming state update
        let state = data;
        if (data.type === 'state') state = data.state;

        // Validation: ensure it looks like a streaming state
        if (state && typeof state === 'object') {
           const { content, events, version, hit_max_iterations } = state;

           if (version !== undefined) lastSeenVersionRef.current = version;

           if (content !== undefined) {
             setStreamingContent(prev => prev === content ? prev : content);
           }

           if (events && Array.isArray(events)) {
             setStreamingEvents(prev => {
                // Skip update if events haven't changed (simple length + last event check)
                if (prev.length === events.length && prev.length > 0) {
                    const lastP = prev[prev.length - 1];
                    const lastN = events[events.length - 1];
                    // Simple check on last item to avoid unnecessary re-renders
                    if (JSON.stringify(lastP) === JSON.stringify(lastN)) return prev;
                }

                // Convert to StreamingRenderEvent format
                return events.map((ev: any) => {
                    if (ev.type === 'content') return { type: 'content' as const, content: ev.content || '' };
                    if (ev.type === 'reasoning') {
                      return {
                        type: 'reasoning' as const,
                        content: ev.content || '',
                        durationSeconds: typeof ev.duration_seconds === 'number' ? ev.duration_seconds : undefined,
                      };
                    }
                  const hasOutput = ev && typeof ev === 'object' && Object.prototype.hasOwnProperty.call(ev, 'output');
                    return {
                        type: 'tool' as const,
                        toolCall: {
                            tool: ev.tool || '',
                            input: ev.input,
                            output: ev.output,
                            presentation: ev.presentation,
                          connection: ev.connection,
                      status: hasOutput ? 'complete' : 'running',
                      generating_lines: ev.generating_lines,
                        }
                    };
                });
             });
           }

           if (hit_max_iterations) setHitMaxIterations(true);
        }
      }
    } catch (err: any) {
        if (err.name !== 'AbortError') {
            console.error('Task stream error:', err);
        }
    } finally {
        if (processingTaskRef.current === taskId) {
            processingTaskRef.current = null;
            setIsPollingTask(false);
            setActiveTask(null);

            // Refresh conversation on completion before clearing streaming state.
            // This avoids a UI gap where the in-progress output disappears before
            // the persisted assistant message is rendered.
            const currentConversation = activeConversationRef.current;
            if (currentConversation) {
              syncConversationActiveTaskId(currentConversation.id, null);
              try {
                const updated = await api.getConversation(currentConversation.id, workspaceId);
                const resolved = applyFallbackAssistantIfNeeded(updated);
                setActiveConversation(resolved);
                setConversations(prev => prev.map(c => c.id === resolved.id ? resolved : c));
              } catch (e) {
                console.error(e);
              }
            }

            setIsStreaming(false);
            setStreamingContent('');
            setStreamingEvents([]);

            // Notify parent that the task finished (e.g. refresh workspace preview)
            if (onTaskComplete) {
                try { onTaskComplete(); } catch (e) { console.error(e); }
            }
        }
    }
  }, [applyFallbackAssistantIfNeeded, onTaskComplete, syncConversationActiveTaskId, workspaceId]);

  const startTaskAndStream = useCallback(async (conversationId: string, message: string) => {
    const previousConversation = activeConversation?.id === conversationId
      ? activeConversation
      : null;

    // 1. Optimistic update (User message)
    let content: string | ContentPart[] = message;
    try {
      const parsed = JSON.parse(message);
      if (Array.isArray(parsed) && parsed.some(p => p.type)) {
        content = parsed;
      }
    } catch {
      // Not JSON
    }

    const optimisticMsg: ChatMessage = {
      role: 'user',
      content: content as any,
      timestamp: new Date().toISOString()
    };

    if (activeConversation) {
      const updatedWithUser = {
        ...activeConversation,
        messages: [...activeConversation.messages, optimisticMsg]
      };
      setActiveConversation(updatedWithUser);
      setConversations(prev => prev.map(c => c.id === conversationId ? updatedWithUser : c));
    }

    let startedTaskId: string | null = null;
    try {
        // 2. Start background task
        const task = await api.sendMessageBackground(conversationId, message, workspaceId);
        startedTaskId = task.id;
        setActiveTask(task);
        setInterruptedTask(null);
      syncConversationActiveTaskId(conversationId, task.id);

        // 3. Connect to stream
        await connectTaskStream(task.id);
    } catch (err: any) {
       console.error(err);
       if (!startedTaskId && previousConversation) {
         setActiveConversation(previousConversation);
         setConversations(prev => prev.map(c => c.id === conversationId ? previousConversation : c));
         syncConversationActiveTaskId(conversationId, previousConversation.active_task_id ?? null);
       }
       setError(err.message || 'Failed to start task');
       clearActiveStreamingUi();
    }
  }, [activeConversation, clearActiveStreamingUi, connectTaskStream, syncConversationActiveTaskId, workspaceId]);

  // Keep task streaming in sync when workspace aggregate state sets activeTask.
  useEffect(() => {
    if (!activeTask) return;
    if (activeTask.status !== 'pending' && activeTask.status !== 'running') return;
    void connectTaskStream(activeTask.id);
  }, [activeTask?.id, activeTask?.status, connectTaskStream]);

  // Check for active/interrupted background task when conversation changes
  useEffect(() => {
    if (workspaceChatState || workspaceId) {
      return;
    }

    let checkInProgress = false;
    const checkTasks = async () => {
      if (checkInProgress) return;
      checkInProgress = true;
      // If we are switching conversations, ensure we stop any previous stream
      if (!activeConversation) {
        stopTaskStreaming();
        setActiveTask(null);
        setInterruptedTask(null);
        checkInProgress = false;
        return;
      }

      // If we are already streaming a task for this conversation, don't interrupt it.
      // But how do we know if the running task belongs to THIS conversation?
      // processingTaskRef stores taskId. activeTask stores taskId.
      // We should check API to be sure.

      try {
        const taskState = await api.getConversationTaskState(activeConversation.id, workspaceId);
        const activeT = taskState.active_task;
        const interruptedT = taskState.interrupted_task;

        // Use functional state update to avoid dependency issues if needed, but here simple set is fine
        if (activeT && (activeT.status === 'pending' || activeT.status === 'running')) {
            setActiveTask(activeT);
            setInterruptedTask(null);
            syncConversationActiveTaskId(activeConversation.id, activeT.id);

            // Connect to stream if not already processing this task
            connectTaskStream(activeT.id);
        } else {
            // No active task for this conversation.
            // If we are streaming something that is NOT this task, we should stop?
            // "stopTaskStreaming" was called in cleanup of previous effect run (when ID changed).
            // So we are clean here usually.

            // If we were processing a task that just finished, connectTaskStream finally block clears it.
            if (!activeT) {
                 setActiveTask(null);
                setInterruptedTask(interruptedT ?? null);
                syncConversationActiveTaskId(activeConversation.id, null);
            }
        }
      } catch (err) {
         console.error('Failed to check tasks:', err);
      } finally {
        checkInProgress = false;
      }
    };

    void checkTasks();
    const interval = setInterval(() => {
      void checkTasks();
    }, 3000);

    return () => {
        clearInterval(interval);
        // Stop streaming when conversation ID changes (unmounting this effect instance)
        stopTaskStreaming();
    };
  }, [activeConversation?.id, connectTaskStream, stopTaskStreaming, syncConversationActiveTaskId, workspaceChatState, workspaceId]);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      stopTaskStreaming();
    };
  }, [stopTaskStreaming]);

  const selectConversation = async (conversation: Conversation) => {
    const isSwitchingConversation = activeConversation?.id !== conversation.id;
    const requestId = ++selectConversationRequestIdRef.current;
    try {
      if (!embedded && typeof window !== 'undefined' && window.matchMedia('(max-width: 768px)').matches) {
        setShowSidebar(false);
      }
      shouldAutoScrollRef.current = true;
      if (isSwitchingConversation) {
        setIsConversationSwitchLoading(true);
      }
      // Stop any current streaming when switching
      clearActiveStreamingUi();

      // Refresh the conversation to get latest messages
      const fresh = await api.getConversation(conversation.id, workspaceId);
      if (requestId !== selectConversationRequestIdRef.current) {
        return;
      }
      setActiveConversation(fresh);
      // Sync sidebar title in case it was updated while another conversation was active
      if (fresh.title !== conversation.title) {
        setConversations(prev => prev.map(c => c.id === fresh.id ? { ...c, title: fresh.title } : c));
      }
      setError(null);
    } catch (err) {
      if (requestId !== selectConversationRequestIdRef.current) {
        return;
      }
      setError(err instanceof Error ? err.message : 'Failed to load conversation');
    } finally {
      if (requestId === selectConversationRequestIdRef.current && isSwitchingConversation) {
        setIsConversationSwitchLoading(false);
      }
    }
  };

  const deleteConversation = async (conversationId: string, e: React.MouseEvent) => {
    e.stopPropagation();

    // If already showing confirmation for this conversation, cancel it
    if (deleteConfirmId === conversationId) {
      setDeleteConfirmId(null);
      return;
    }

    // Show inline confirmation
    setDeleteConfirmId(conversationId);
  };

  const confirmDeleteConversation = async (conversationId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteConfirmId(null);

    try {
      await api.deleteConversation(conversationId, workspaceId);
      setConversations(prev => prev.filter(c => c.id !== conversationId));
      if (activeConversation?.id === conversationId) {
        setActiveConversation(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete conversation');
    }
  };

  const startEditingTitle = (conversation: Conversation, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingTitle(conversation.id);
    setTitleInput(conversation.title);
  };

  const saveTitle = async (conversationId: string) => {
    if (!titleInput.trim()) {
      setEditingTitle(null);
      return;
    }

    try {
      const updated = await api.updateConversationTitle(conversationId, titleInput.trim(), workspaceId);
      setConversations(prev => prev.map(c => c.id === conversationId ? updated : c));
      if (activeConversation?.id === conversationId) {
        setActiveConversation(updated);
      }
      setEditingTitle(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update title');
    }
  };

  const changeModel = async (newModel: string) => {
    if (!activeConversation || isStreaming) return;

    try {
      const parsedSelection = parseStoredModelIdentifier(newModel);
      const requestedModelId = (parsedSelection.modelId || newModel).trim();
      const requestedProvider = normalizeProviderAlias(parsedSelection.provider);
      const requestedProviderForApi = KNOWN_PROVIDER_KEYS.has(requestedProvider)
        ? (requestedProvider as LlmProviderWire)
        : undefined;

      let selected = availableModels.find((model) => (
        model.id === requestedModelId
        && providersEquivalent(model.provider, requestedProvider)
      ));
      if (!selected) {
        selected = availableModels.find((model) => model.id === requestedModelId);
      }

      const updated = await api.updateConversationModel(
        activeConversation.id,
        selected?.id || requestedModelId,
        workspaceId,
        selected?.provider || requestedProviderForApi,
      );
      setActiveConversation(updated);
      setConversations(prev => prev.map(c => c.id === updated.id ? updated : c));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to change model');
    }
  };

  const startFreshConversation = async () => {
    if (isReadOnly) return;
    // Start a fresh conversation and detach from any currently streaming one.
    shouldAutoScrollRef.current = true;
    try {
      clearActiveStreamingUi();
      const conversation = await api.createConversation(undefined, workspaceId);
      applyCreatedConversation(conversation);
      setInterruptedTask(null);
      setHitMaxIterations(false);
      setIsConnectionError(false);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create conversation');
    }
  };

  const stopStreaming = async () => {
    // Cancel regular streaming
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Cancel background task if active
    if (activeTask && (activeTask.status === 'pending' || activeTask.status === 'running')) {
      try {
        await api.cancelChatTask(activeTask.id, workspaceId);
      } catch (err) {
        console.error('Failed to cancel task:', err);
      }
    }

    stopTaskStreaming();
    setActiveTask(null);
    setInterruptedTask(null);
    if (activeConversationRef.current) {
      syncConversationActiveTaskId(activeConversationRef.current.id, null);
    }
    // Don't modify isStreaming yet to avoid UI flash/loss of content during refresh

    // Refresh conversation to get current state (including partial messages)
    const currentConversation = activeConversationRef.current;
    if (currentConversation) {
      try {
        const updated = await api.getConversation(currentConversation.id, workspaceId);
        const resolved = applyFallbackAssistantIfNeeded(updated);
        setActiveConversation(resolved);
        setConversations(prev => prev.map(c => c.id === resolved.id ? resolved : c));
      } catch (err) {
        console.error('Failed to update conversation after stop:', err);

        // If refresh fails, still preserve what the user saw in the stream.
        const localConversation = activeConversationRef.current;
        if (localConversation) {
          const fallbackConversation = applyFallbackAssistantIfNeeded(localConversation);
          setActiveConversation(fallbackConversation);
          setConversations(prev => prev.map(c => c.id === fallbackConversation.id ? fallbackConversation : c));
        }
      }
    }

    // Always clear streaming state at the end
    setStreamingContent('');
    setStreamingEvents([]);
    setIsStreaming(false);
  };

  const resendMessage = () => {
    if (lastSentMessage && !isStreaming) {
      setError(null);
      setIsConnectionError(false);
      // Directly send the message without setting inputValue
      sendMessageDirect(lastSentMessage);
    }
  };

  const continueConversation = async () => {
    if (!activeConversation || isStreaming) return;
    setHitMaxIterations(false);
    setInterruptedTask(null);
    // Directly send the continuation message
    sendMessageDirect('continue');
  };

  // Direct message send - bypasses inputValue state for programmatic sending
  const sendMessageDirect = async (message: string) => {
    if (!message.trim() || !activeConversation || isStreaming || isReadOnly) return;
    if (sendReadinessBlockReason) {
      setError(sendReadinessBlockReason);
      return;
    }
    shouldAutoScrollRef.current = true;

    const userMessage = message.trim();
    setError(null);
    setHitMaxIterations(false);
    setIsConnectionError(false);
    setLastSentMessage(userMessage);

    const contextLimit = getContextLimit(activeConversation.model);
    const contextUsage = calculateConversationContextUsage({
      messages: activeConversation.messages,
      persistedConversationTokens: activeConversation.total_tokens,
      contextLimit,
      inputText: userMessage,
    });

    if (!contextUsage.hasHeadroom) {
      setError(`Context limit nearly reached (${contextUsage.projectedInputPercent}%). Consider starting a new conversation.`);
      return;
    }

    if (onUserMessageSubmitted) {
      try {
        await onUserMessageSubmitted(userMessage);
      } catch (callbackError) {
        console.warn('Failed to run message submit callback:', callbackError);
      }
    }

    setIsStreaming(true);
    setStreamingContent('');
    setStreamingEvents([]);
    setHitMaxIterations(false);

    try {
      // Use background task streaming
      await startTaskAndStream(activeConversation.id, userMessage);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message';

      // Check for connection errors
      const isConnError = errorMessage.toLowerCase().includes('502') ||
                          errorMessage.toLowerCase().includes('503') ||
                          errorMessage.toLowerCase().includes('connection') ||
                          errorMessage.toLowerCase().includes('network') ||
                          errorMessage.toLowerCase().includes('fetch');

      setIsConnectionError(isConnError);
      setError(errorMessage);

      setIsStreaming(false);
      setStreamingContent('');
      setStreamingEvents([]);
    }
  };

  const sendMessage = async () => {
    if ((!inputValue.trim() && attachments.length === 0) || !activeConversation || isStreaming || isReadOnly) return;
    if (sendReadinessBlockReason) {
      setError(sendReadinessBlockReason);
      return;
    }

    const userMessage = inputValue.trim();
    const messageAttachments = [...attachments];

    setInputValue('');
    setAttachments([]);

    // Auto-collapse sidebar when user starts chatting
    setShowSidebar(false);

    // Convert attachments to content parts if present
    if (messageAttachments.length > 0) {
      const contentParts = attachmentsToContentParts(userMessage, messageAttachments);
      sendMessageDirect(JSON.stringify(contentParts));
    } else {
      sendMessageDirect(userMessage);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (isReadOnly) return;
    if (e.key === 'Enter' && !e.shiftKey) {
      if (isStreaming) return; // Allow typing while a response streams

      e.preventDefault();
      sendMessage();
    }
  };

  const contentPartsToAttachments = useCallback(async (parts: ContentPart[]): Promise<AttachmentFile[]> => {
    // Extract mime type from data URL (e.g., "data:image/png;base64,..." -> "image/png")
    const extractMimeFromDataUrl = (url: string): string => {
      const match = url.match(/^data:([^;,]+)/);
      return match?.[1] || 'image/png';
    };

    const mapped = await Promise.all(parts
      .filter(p => p.type === 'image_url' || p.type === 'file')
      .map(async (p, i) => {
        if (p.type === 'image_url') {
          const url = p.image_url.url;
          const mimeType = extractMimeFromDataUrl(url);
          const resized = await resizeAttachmentImageDataUrl(url, mimeType);
          return {
            id: `edit-attachment-${i}-${Date.now()}`,
            type: 'image' as const,
            name: 'Attached Image',
            size: 0,
            mimeType,
            preview: resized,
            dataUrl: resized
          } as AttachmentFile;
        }
        // Handle file type
        return {
          id: `edit-attachment-${i}-${Date.now()}`,
          type: 'file' as const,
          name: (p as any).filename || 'file',
          size: 0,
          mimeType: (p as any).mime_type || 'application/octet-stream',
          filePath: (p as any).file_path
        } as AttachmentFile;
      }));

    return mapped;
  }, []);

  const startEditMessage = (idx: number, content: string, attachments: ContentPart[] = []) => {
    setEditingMessageIdx(idx);
    setEditMessageContent(content);

    (async () => {
      const files = await contentPartsToAttachments(attachments);
      setEditMessageAttachments(files);
    })();
  };

  const cancelEditMessage = () => {
    setEditingMessageIdx(null);
    setEditMessageContent('');
    setEditMessageAttachments([]);
  };

  const createBranchForMessageMutation = useCallback(async (
    conversationId: string,
    branchPointIndex: number,
    messageCount: number,
  ) => {
    if (branchPointIndex < 0 || branchPointIndex >= messageCount) {
      return;
    }

    const createdBranch = await api.createConversationBranch(
      conversationId,
      { from_message_index: branchPointIndex, auto_snapshot: Boolean(workspaceId) },
      workspaceId,
    );

    if (createdBranch.parent_branch_id) {
      setBranchSelections((prev) => ({ ...prev, [branchPointIndex]: createdBranch.parent_branch_id! }));
    }

    // Branch creation with auto_snapshot may have created a new userspace
    // snapshot. Notify parent so the snapshots panel can refresh.
    if (workspaceId) {
      try {
        onSnapshotsMaybeChanged?.();
      } catch (notifyErr) {
        console.warn('onSnapshotsMaybeChanged threw:', notifyErr);
      }
    }
  }, [workspaceId, onSnapshotsMaybeChanged]);

  // Walk back from messageIdx to the nearest user message. Branches are
  // always anchored at user messages so the branch nav appears on a single,
  // predictable row and replay/edit/delete cannot collide by anchoring at
  // assistant indices.
  const findUserMessageIndexAtOrBefore = useCallback((
    messages: ChatMessage[],
    messageIdx: number,
  ): number => {
    let i = Math.min(messageIdx, messages.length - 1);
    while (i >= 0 && messages[i]?.role !== 'user') {
      i--;
    }
    return i;
  }, []);

  const getDeleteBranchPointIndex = useCallback((
    messages: ChatMessage[],
    messageIdx: number,
  ): number => {
    const messageCount = messages.length;
    const maxBranchPointIndex = Math.max(messageCount - 1, 0);
    const userIdx = findUserMessageIndexAtOrBefore(messages, messageIdx);
    if (userIdx >= 0) {
      return clampNumber(userIdx, 0, maxBranchPointIndex);
    }
    return clampNumber(messageIdx, 0, maxBranchPointIndex);
  }, [findUserMessageIndexAtOrBefore]);

  const switchBranch = useCallback(async (branchId: string) => {
    if (!activeConversation || branchSwitching) return;
    if (branchId.startsWith('__current__:')) return;
    const conversationId = activeConversation.id;
    setBranchSwitching(true);
    try {
      // Remember the old active branch's position before switching
      const oldActiveBranchId = activeConversation.active_branch_id;
      if (oldActiveBranchId) {
        const oldBranch = branchesById.get(oldActiveBranchId);
        if (oldBranch) {
          setBranchSelections(prev => ({ ...prev, [oldBranch.branch_point_index]: oldActiveBranchId }));
        }
      }

      const updated = await api.switchConversationBranch(conversationId, branchId, workspaceId);
      setActiveConversation(updated);
      setConversations(prev => prev.map(c => c.id === updated.id ? updated : c));
      const refreshedPoints = await refreshBranchPoints(conversationId);

      // Notify parent (UserSpacePanel) about the branch switch with associated snapshot
      if (onBranchSwitch) {
        const allBranches = refreshedPoints.flatMap(bp => bp.branches);
        const targetBranch = allBranches.find(b => b.id === branchId);
        onBranchSwitch(branchId, targetBranch?.associated_snapshot_id ?? null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to switch branch');
    } finally {
      setBranchSwitching(false);
    }
  }, [activeConversation, branchSwitching, workspaceId, branchesById, refreshBranchPoints, onBranchSwitch]);

  const copyMessageText = useCallback(async (idx: number, content: string | ContentPart[]) => {
    const { text } = parseMessageContent(content);
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageIdx(idx);
      setTimeout(() => setCopiedMessageIdx(prev => prev === idx ? null : prev), 2000);
    } catch { /* ignore */ }
  }, []);

  const replayFromMessage = useCallback(async (messageIdx: number) => {
    if (!activeConversation || isStreaming || isReadOnly) return;
    const conversationId = activeConversation.id;
    const selectedMessage = activeConversation.messages[messageIdx];
    if (!selectedMessage) return;

    // Replay always anchors at the user message that drove the (possibly
    // assistant) selected row, so the branch nav surfaces on the user row
    // and a single replay creates exactly one new branch.
    const userIdx = findUserMessageIndexAtOrBefore(activeConversation.messages, messageIdx);
    if (userIdx < 0) return;

    const userMsg = activeConversation.messages[userIdx];
    const truncateAt = userIdx;
    try {
      await createBranchForMessageMutation(
        conversationId,
        truncateAt,
        activeConversation.messages.length,
      );

      // Re-send the original user message content verbatim
      const rawContent = userMsg.content;
      const messageToSend = typeof rawContent === 'string'
        ? rawContent
        : JSON.stringify(rawContent);

      // Optimistic update
      const messagesToKeep = activeConversation.messages.slice(0, truncateAt);
      const optimisticMsg: ChatMessage = {
        role: 'user',
        content: rawContent as any,
        timestamp: new Date().toISOString(),
      };
      const optimisticConv: Conversation = {
        ...activeConversation,
        messages: [...messagesToKeep, optimisticMsg],
      };
      setActiveConversation(optimisticConv);
      setConversations(prev => prev.map(c => c.id === optimisticConv.id ? optimisticConv : c));

      shouldAutoScrollRef.current = true;
      setIsStreaming(true);
      setStreamingContent('');
      setStreamingEvents([]);
      setHitMaxIterations(false);
      setIsConnectionError(false);

      const task = await api.sendMessageBackground(conversationId, messageToSend, workspaceId);
      setActiveTask(task);
      syncConversationActiveTaskId(conversationId, task.id);
      await connectTaskStream(task.id);

      // Refresh branch points after the branch was created
      void refreshBranchPoints(conversationId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retry message');
    }
  }, [activeConversation, isStreaming, isReadOnly, createBranchForMessageMutation, findUserMessageIndexAtOrBefore, workspaceId, refreshBranchPoints, connectTaskStream, syncConversationActiveTaskId]);

  const deleteFromMessage = useCallback(async (messageIdx: number) => {
    if (!activeConversation || isStreaming || isReadOnly) return;
    const conversationId = activeConversation.id;
    const selectedMessage = activeConversation.messages[messageIdx];
    if (!selectedMessage) return;
    const selectedMessageId = selectedMessage.message_id;

    // Always anchor the chat branch at the user message that drove this row.
    // This guarantees the branch nav (X/N) renders on a single user row and
    // each walkback creates exactly one branch.
    const branchPointIndex = getDeleteBranchPointIndex(
      activeConversation.messages,
      messageIdx,
    );
    const rolledBackSnapshot = Boolean(
      workspaceId && selectedMessageId && selectedMessage.snapshot_restore,
    );

    try {
      // 1. Preserve the original messages on a new branch (auto-snapshots
      //    the current workspace state when running inside a workspace).
      await createBranchForMessageMutation(
        conversationId,
        branchPointIndex,
        activeConversation.messages.length,
      );

      // 2. If this message has an explicit snapshot link, restore the
      //    workspace files to that snapshot. The endpoint also truncates
      //    the chat to the link's stored restore_message_count, but since
      //    branch creation already truncated to branchPointIndex, that
      //    second truncate is a no-op (truncate cannot grow the chat).
      let updatedConversation: Conversation | null = null;
      if (rolledBackSnapshot && selectedMessageId) {
        const restored = await api.restoreConversationMessageSnapshot(
          conversationId,
          selectedMessageId,
          workspaceId,
        );
        updatedConversation = restored.conversation;
      }

      // 3. Source-of-truth resync: pull the conversation server-side so the
      //    chat reflects the truncation done by branch creation (and any
      //    further mutations the restore endpoint applied). This is one
      //    extra GET but keeps the optimistic state honest in every code
      //    path (delete-with-snapshot, delete-without-snapshot, both roles).
      if (!updatedConversation) {
        updatedConversation = await api.getConversation(conversationId, workspaceId);
      }

      setActiveConversation(updatedConversation);
      setConversations(prev => prev.map(c => c.id === updatedConversation!.id ? updatedConversation! : c));

      if (rolledBackSnapshot) {
        onMessageSnapshotRestored?.({
          rolledBackSnapshot: true,
          requiresRuntimeRestart: true,
        });
        // Walkback moved the snapshot cursor; tell the snapshots panel.
        try {
          onSnapshotsMaybeChanged?.();
        } catch (notifyErr) {
          console.warn('onSnapshotsMaybeChanged threw:', notifyErr);
        }
      }
      void refreshBranchPoints(conversationId);
    } catch (err) {
      console.error('Failed to delete from message:', err);
      const message = err instanceof Error ? err.message : 'Failed to delete message';
      alert(message);
    }
  }, [activeConversation, isStreaming, isReadOnly, createBranchForMessageMutation, getDeleteBranchPointIndex, onMessageSnapshotRestored, onSnapshotsMaybeChanged, refreshBranchPoints, workspaceId]);

  const submitEditMessage = async () => {
    if (isReadOnly || !activeConversation || editingMessageIdx === null || (!editMessageContent.trim() && editMessageAttachments.length === 0)) return;
    shouldAutoScrollRef.current = true;
    const conversationId = activeConversation.id;

    let messageToSend: string = editMessageContent.trim();
    if (editMessageAttachments.length > 0) {
      const normalizedEditAttachments = await Promise.all(editMessageAttachments.map(async (attachment) => {
        if (attachment.type !== 'image' || !attachment.dataUrl) {
          return attachment;
        }
        const resized = await resizeAttachmentImageDataUrl(attachment.dataUrl, attachment.mimeType);
        return {
          ...attachment,
          dataUrl: resized,
          preview: resized,
        };
      }));
      const parts = attachmentsToContentParts(messageToSend, normalizedEditAttachments);
      messageToSend = JSON.stringify(parts);
    }

    const truncateAt = Math.max(0, editingMessageIdx);

    // Clear the edit state
    setEditingMessageIdx(null);
    setEditMessageContent('');
    setEditMessageAttachments([]);
    setError(null);

    try {
      // 1. Create a branch to preserve the original messages
      await createBranchForMessageMutation(
        conversationId,
        truncateAt,
        activeConversation.messages.length,
      );

      // 2. Local Optimistic Update
      const messagesToKeep = activeConversation.messages.slice(0, truncateAt);

      let content: string | ContentPart[] = messageToSend;
      try {
        const parsed = JSON.parse(messageToSend);
        if (Array.isArray(parsed) && parsed.some(p => p.type)) {
          content = parsed;
        }
      } catch {
        // Not JSON
      }

      const optimisticMsg: ChatMessage = {
        role: 'user',
        content: content as any,
        timestamp: new Date().toISOString()
      };

      const optimisticConv: Conversation = {
        ...activeConversation,
        messages: [...messagesToKeep, optimisticMsg]
      };

      setActiveConversation(optimisticConv);
      setConversations(prev => prev.map(c => c.id === optimisticConv.id ? optimisticConv : c));

      setIsStreaming(true);
      setStreamingContent('');
      setStreamingEvents([]);
      setHitMaxIterations(false);
      setIsConnectionError(false);

      // 3. Start background task
      const task = await api.sendMessageBackground(conversationId, messageToSend, workspaceId);
      setActiveTask(task);
      setInterruptedTask(null);
      syncConversationActiveTaskId(conversationId, task.id);

      // 4. Connect to stream
      await connectTaskStream(task.id);

      // 5. Refresh branch points for UI
      void refreshBranchPoints(conversationId);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resend message');
      setIsStreaming(false);
      setStreamingContent('');
      setStreamingEvents([]);

      // Restore authoritative state from server on error
      try {
        const refreshed = await api.getConversation(conversationId, workspaceId);
        setActiveConversation(refreshed);
        setConversations(prev => prev.map(c => c.id === refreshed.id ? refreshed : c));
      } catch (refreshErr) {
        console.error('Failed to refresh conversation after edit error:', refreshErr);
      }
    }
  };

  // Memoize context usage calculation to avoid recalculating on every render
  const contextUsage = useMemo(() => {
    if (!activeConversation) {
      return {
        currentTokens: 0,
        totalTokens: 0,
        contextLimit: defaultContextLimit,
        contextUsagePercent: 0,
        projectedInputPercent: 0,
        hasHeadroom: true,
      };
    }

    const contextLimit = getContextLimit(activeConversation.model);
    return calculateConversationContextUsage({
      messages: activeConversation.messages,
      persistedConversationTokens: activeConversation.total_tokens,
      contextLimit,
      inputText: inputValue,
      isStreaming,
      streamingEvents: streamingEvents as StreamingRenderEvent[],
      streamingContent,
    });
  }, [activeConversation, defaultContextLimit, getContextLimit, inputValue, isStreaming, streamingContent, streamingEvents]);

  const showWorkspaceConversationSelect = embedded && Boolean(workspaceId);
  const workspaceConversationOptions = conversations.filter((conv) => conv.title !== 'Untitled Chat');
  const shareableConversationUsers = useMemo(() => {
    if (!conversationShareableUserIds || conversationShareableUserIds.length === 0) {
      return allUsers;
    }
    const allowedIds = new Set(conversationShareableUserIds);
    if (conversationOwnerId) {
      allowedIds.add(conversationOwnerId);
    }
    return allUsers.filter((user) => allowedIds.has(user.id));
  }, [allUsers, conversationOwnerId, conversationShareableUserIds]);
  const showInlineToolSelector = canUseConversationTools;

  const renderConversationItem = (conv: Conversation) => {
    const metaText = `${conv.messages.length} messages | ${formatChatTimestamp(conv.updated_at)}`;
    const isActive = activeConversation?.id === conv.id;

    return (
      <div
        key={conv.id}
        className={`chat-conversation-item ${isActive ? 'active' : ''}`}
        onClick={() => selectConversation(conv)}
      >
        {editingTitle === conv.id ? (
          <input
            type="text"
            value={titleInput}
            onChange={(e) => setTitleInput(e.target.value)}
            onBlur={() => saveTitle(conv.id)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') saveTitle(conv.id);
              if (e.key === 'Escape') setEditingTitle(null);
            }}
            onClick={(e) => e.stopPropagation()}
            autoFocus
            className="chat-title-input"
          />
        ) : (
          <>
            <div className="chat-conversation-title">
              {conv.active_task_id && (
                <span className="chat-task-indicator" title="Processing in background">
                  <MiniLoadingSpinner variant="icon" size={12} />
                </span>
              )}
              <ChatTitle title={conv.title} />
            </div>
            <div className="chat-conversation-meta">
              {metaText}
            </div>
          </>
        )}
        <div className="chat-conversation-actions">
          {deleteConfirmId === conv.id ? (
            <>
              <button
                className="chat-action-btn confirm-delete"
                onClick={(e) => confirmDeleteConversation(conv.id, e)}
                title="Confirm delete"
              >
                <Check size={14} />
              </button>
              <button
                className="chat-action-btn cancel-delete"
                onClick={(e) => {
                  e.stopPropagation();
                  setDeleteConfirmId(null);
                }}
                title="Cancel"
              >
                <X size={14} />
              </button>
            </>
          ) : (
            <>
              <button
                className="chat-action-btn"
                onClick={(e) => startEditingTitle(conv, e)}
                title="Rename"
              >
                <Pencil size={14} />
              </button>
              <button
                className="chat-action-btn"
                onClick={(e) => deleteConversation(conv.id, e)}
                title="Delete"
              >
                <Trash2 size={14} />
              </button>
            </>
          )}
        </div>
      </div>
    );
  };

  const renderMessageBubbleSkeletons = () => (
    <div className="chat-message-skeleton-list">
      <div className="chat-message-skeleton-row chat-message-skeleton-row-user">
        <div className="chat-message-skeleton-bubble chat-message-skeleton-bubble-user">
          <div className="chat-skeleton-line chat-message-skeleton-line"></div>
          <div className="chat-skeleton-line chat-message-skeleton-line chat-message-skeleton-line-user-short"></div>
        </div>
      </div>
      <div className="chat-message-skeleton-row chat-message-skeleton-row-assistant">
        <div className="chat-message-skeleton-bubble chat-message-skeleton-bubble-assistant">
          <div className="chat-skeleton-line chat-message-skeleton-line"></div>
          <div className="chat-skeleton-line chat-message-skeleton-line chat-message-skeleton-line-short"></div>
        </div>
      </div>
      <div className="chat-message-skeleton-row chat-message-skeleton-row-assistant">
        <div className="chat-message-skeleton-bubble chat-message-skeleton-bubble-assistant">
          <div className="chat-skeleton-line chat-message-skeleton-line"></div>
          <div className="chat-skeleton-line chat-message-skeleton-line chat-message-skeleton-line-short"></div>
        </div>
      </div>
    </div>
  );

  const renderFullChatSkeleton = () => (
    <>
      {/* Skeleton header */}
      <div className={`chat-header ${embedded ? 'chat-header-embedded' : ''} chat-header-skeleton`} aria-hidden="true">
        <div className="chat-header-info">
          <div className="chat-skeleton-line chat-header-skeleton-title"></div>
        </div>
        <div className="chat-header-actions">
          <div className="chat-skeleton-line chat-header-skeleton-action"></div>
          <div className="chat-skeleton-line chat-header-skeleton-action"></div>
        </div>
      </div>
      {/* Skeleton messages */}
      <div className="chat-messages chat-messages-skeleton">
        {renderMessageBubbleSkeletons()}
      </div>
      {/* Skeleton input area */}
      <div className="chat-input-area manual-resize chat-input-area-skeleton" aria-hidden="true">
        <div className="chat-input-wrapper">
          <div className="chat-skeleton-line chat-input-skeleton-line"></div>
        </div>
      </div>
    </>
  );

  const panelStyle: CSSProperties | undefined = !embedded
    ? ({ ['--chat-sidebar-width' as '--chat-sidebar-width']: `${sidebarWidth}px` } as CSSProperties)
    : undefined;

  return (
    <div className={`chat-panel ${embedded ? 'chat-panel-embedded' : ''}${showWorkspaceConversationSelect ? ' chat-panel-workspace' : ''}${isFullscreen ? ' chat-panel-fullscreen' : ''}`} style={panelStyle}>
      {/* Conversations Sidebar */}
      {!embedded && showSidebar && <div className="chat-sidebar open">
        <div className="chat-sidebar-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h3>Conversations</h3>
            {conversations.some(c => c.active_task_id) && (
              <span title="Processing in background">
                <MiniLoadingSpinner variant="icon" size={14} />
              </span>
            )}
          </div>
        </div>

        <div className={`chat-conversation-list ${!isAdmin ? 'chat-conversation-list-non-admin' : ''}`} aria-busy={isConversationListLoading}>
          {isConversationListLoading ? (
            <div className="chat-conversation-skeleton-list" aria-hidden="true">
              {Array.from({ length: isAdmin ? 6 : 8 }).map((_, index) => (
                <div key={index} className="chat-conversation-skeleton">
                  <div className="chat-skeleton-line chat-conversation-skeleton-title"></div>
                  <div className="chat-skeleton-line chat-conversation-skeleton-meta"></div>
                </div>
              ))}
            </div>
          ) : conversations.length === 0 ? (
            <div className="chat-empty-state">
              <p>No conversations yet</p>
              <button className="btn" onClick={createNewConversation}>
                Start a conversation
              </button>
            </div>
          ) : isAdmin ? (
            groupedConversations.map(group => {
              const isCollapsed = collapsedGroups[group.key] ?? true;
              return (
                <div key={group.key} className="chat-conversation-group">
                  <button className="chat-group-header" onClick={() => toggleGroup(group.key)}>
                    <span className="chat-group-name">{group.label}</span>
                    <span className="chat-group-count">{group.conversations.length}</span>
                    <span className="chat-group-toggle">{isCollapsed ? '▶' : '▼'}</span>
                  </button>
                  {!isCollapsed && (
                    <div className="chat-group-list">
                      {group.conversations.map(renderConversationItem)}
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            conversations.map(renderConversationItem)
          )}
        </div>
      </div>}

      {!embedded && (
        <ResizeHandle
          direction="horizontal"
          className="resize-handle resize-handle-horizontal chat-resize-handle"
          onResize={handleResizeSidebar}
          collapsed={!showSidebar ? 'before' : undefined}
          onExpand={expandSidebar}
        />
      )}

      {/* Main Chat Area */}
      <div className="chat-main" ref={chatMainRef}>
        {isConversationListLoading ? (
          renderFullChatSkeleton()
        ) : activeConversation ? (
          <>
            {/* Chat Header */}
            <div className={`chat-header ${embedded ? 'chat-header-embedded' : ''}`}>
              <div className="chat-header-info">
                {!embedded && !isAdmin && (
                  <button
                    className="btn btn-secondary btn-sm btn-icon"
                    onClick={() => setShowSidebar((prev) => !prev)}
                    title={showSidebar ? 'Hide sidebar' : 'Show sidebar'}
                    style={{ marginRight: '8px' }}
                  >
                    {showSidebar ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
                  </button>
                )}
                {showWorkspaceConversationSelect ? (
                  <div
                    className="chat-workspace-conversation-picker"
                    ref={workspaceConversationDropdownRef}
                  >
                    <div
                      role="button"
                      tabIndex={0}
                      className="model-selector-trigger chat-workspace-conversation-trigger"
                      onClick={() => setIsWorkspaceConversationMenuOpen((open) => !open)}
                      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setIsWorkspaceConversationMenuOpen((open) => !open); } }}
                      title="Select a workspace chat"
                      aria-haspopup="listbox"
                      aria-expanded={isWorkspaceConversationMenuOpen}
                    >
                      <MessageSquare size={14} className="chat-workspace-conversation-icon" aria-hidden="true" />
                      <span className="model-selector-text chat-workspace-conversation-trigger-label">{activeConversation.title || 'Untitled Chat'}</span>
                      {(Boolean(activeTask) || Boolean(activeConversation.active_task_id)) && (
                        <MiniLoadingSpinner variant="icon" size={14} title="Processing in background" ariaHidden />
                      )}
                      {!(Boolean(activeTask) || Boolean(activeConversation.active_task_id)) && (Boolean(interruptedTask) || interruptedConversationIds.size > 0) && !interruptDismissed && (
                        <button
                          type="button"
                          className="chat-workspace-interrupt-dismiss"
                          title="A conversation was interrupted — click to dismiss"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (workspaceId) dismissInterruptAlert(currentUser.id, workspaceId);
                            setInterruptDismissed(true);
                          }}
                        >
                          <AlertCircle size={13} className="alert-icon" />
                          <Slash size={13} className="dismiss-icon" aria-hidden />
                        </button>
                      )}
                      <span className="model-selector-arrow chat-workspace-conversation-trigger-arrow">▾</span>
                    </div>

                    {isWorkspaceConversationMenuOpen && (
                      <div className="model-selector-dropdown chat-workspace-conversation-dropdown">
                        <div className="model-selector-dropdown-inner" role="listbox" aria-label="Workspace chats">
                          {workspaceConversationOptions.map((conversation) => {
                            const isSelected = conversation.id === activeConversation.id;
                            const isEditing = editingTitle === conversation.id;
                            const isLive = Boolean(conversation.active_task_id) || (isSelected && Boolean(activeTask));
                            const isInterruptedTask = !isLive && (isSelected ? Boolean(interruptedTask) : interruptedConversationIds.has(conversation.id));
                            const isRawInterrupted = isInterruptedTask;
                            const isInterrupted = isRawInterrupted && !interruptDismissed;

                            return (
                              <div
                                key={conversation.id}
                                role="option"
                                tabIndex={0}
                                aria-selected={isSelected}
                                className={`model-selector-item chat-workspace-conversation-item ${isSelected ? 'is-selected' : ''}`}
                                onClick={() => {
                                  setIsWorkspaceConversationMenuOpen(false);
                                  void selectConversation(conversation);
                                }}
                                onKeyDown={(event) => {
                                  if (event.key === 'Enter' || event.key === ' ') {
                                    event.preventDefault();
                                    setIsWorkspaceConversationMenuOpen(false);
                                    void selectConversation(conversation);
                                  }
                                }}
                              >
                                <div className="chat-workspace-conversation-content">
                                  {isEditing ? (
                                    <input
                                      type="text"
                                      value={titleInput}
                                      onChange={(e) => setTitleInput(e.target.value)}
                                      onBlur={() => void saveTitle(conversation.id)}
                                      onKeyDown={(e) => {
                                        if (e.key === 'Enter') {
                                          e.preventDefault();
                                          void saveTitle(conversation.id);
                                        }
                                        if (e.key === 'Escape') {
                                          e.preventDefault();
                                          setEditingTitle(null);
                                        }
                                      }}
                                      onClick={(e) => e.stopPropagation()}
                                      autoFocus
                                      className="chat-title-input chat-workspace-conversation-title-input"
                                    />
                                  ) : (
                                    <span className="model-selector-item-name chat-workspace-conversation-item-name">
                                      {conversation.title || 'Untitled Chat'}
                                    </span>
                                  )}
                                </div>

                                {!isEditing && (
                                  <div
                                    className="chat-workspace-conversation-actions"
                                    onClick={(event) => event.stopPropagation()}
                                  >
                                    {deleteConfirmId === conversation.id ? (
                                      <>
                                        <button
                                          type="button"
                                          className="chat-action-btn confirm-delete"
                                          onClick={(e) => void confirmDeleteConversation(conversation.id, e)}
                                          title="Confirm delete"
                                        >
                                          <Check size={14} />
                                        </button>
                                        <button
                                          type="button"
                                          className="chat-action-btn cancel-delete"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            setDeleteConfirmId(null);
                                          }}
                                          title="Cancel"
                                        >
                                          <X size={14} />
                                        </button>
                                      </>
                                    ) : (
                                      <>
                                        <button
                                          type="button"
                                          className="chat-action-btn"
                                          onClick={(e) => startEditingTitle(conversation, e)}
                                          title="Rename"
                                        >
                                          <Pencil size={14} />
                                        </button>
                                        <button
                                          type="button"
                                          className="chat-action-btn"
                                          onClick={(e) => void deleteConversation(conversation.id, e)}
                                          title="Delete"
                                        >
                                          <Trash2 size={14} />
                                        </button>
                                      </>
                                    )}
                                  </div>
                                )}

                                {isInterrupted && (
                                  <button
                                    type="button"
                                    className="chat-workspace-interrupt-dismiss is-inline"
                                    title="Conversation interrupted — click to dismiss"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      if (workspaceId) dismissInterruptAlert(currentUser.id, workspaceId);
                                      setInterruptDismissed(true);
                                    }}
                                  >
                                    <AlertCircle size={12} className="alert-icon chat-workspace-conversation-interrupted" />
                                    <Slash size={12} className="dismiss-icon" aria-hidden />
                                  </button>
                                )}
                                {isLive && (
                                  <MiniLoadingSpinner variant="icon" size={14} title="Processing in background" ariaHidden />
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="chat-header-title-row">
                    {editingTitle === activeConversation.id ? (
                      <input
                        type="text"
                        value={titleInput}
                        onChange={(e) => setTitleInput(e.target.value)}
                        onBlur={() => saveTitle(activeConversation.id)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') void saveTitle(activeConversation.id);
                          if (e.key === 'Escape') setEditingTitle(null);
                        }}
                        autoFocus
                        className="chat-title-input chat-header-title-input"
                      />
                    ) : (
                      <>
                        <h2>{activeConversation.title}</h2>
                        <button
                          className="chat-header-title-edit-btn"
                          onClick={() => {
                            setEditingTitle(activeConversation.id);
                            setTitleInput(activeConversation.title);
                          }}
                          title="Rename"
                          aria-label="Rename conversation"
                        >
                          <Pencil size={14} />
                        </button>
                      </>
                    )}
                  </div>
                )}
              </div>
              <div className="chat-header-actions">
                <ContextUsagePie
                  currentTokens={contextUsage.currentTokens}
                  totalTokens={contextUsage.totalTokens}
                  contextLimit={contextUsage.contextLimit}
                  loading={modelsLoading}
                />
                {canManageConversationMembers && (
                  <MemberManagementButton
                    className="btn btn-secondary btn-sm btn-icon"
                    onClick={handleOpenMembersModal}
                    title="Manage conversation members"
                  />
                )}
                {canUseConversationTools && (
                  <ToolSelectorDropdown
                    availableTools={availableTools}
                    selectedToolIds={resolvedConversationToolIdSet}
                    onToggleTool={handleToggleConversationTool}
                    selectedToolGroupIds={conversationToolGroupIdSet}
                    onToggleToolGroup={handleToggleConversationToolGroup}
                    toolGroups={toolGroups}
                    disabled={false}
                    readOnly={false}
                    saving={savingTools}
                    title="Conversation Tools"
                    showToolCalls={showToolCalls}
                    onToggleToolCalls={setShowToolCalls}
                  />
                )}
                <ModelSelector
                  models={availableModels}
                  selectedModelId={(() => {
                    const parsedActiveModel = parseStoredModelIdentifier(activeConversation.model || '');
                    const explicitProvider = normalizeProviderAlias(parsedActiveModel.provider);
                    const activeModelId = parsedActiveModel.modelId;

                    let selected = undefined;
                    if (activeModelId && explicitProvider) {
                      selected = availableModels.find((model) => (
                        model.id === activeModelId
                        && providersEquivalent(model.provider, explicitProvider)
                      ));
                    }
                    if (!selected && activeModelId) {
                      selected = availableModels.find((model) => model.id === activeModelId);
                    }

                    if (!selected && activeModelId.includes('/')) {
                      const slashIndex = activeModelId.indexOf('/');
                      const inferredProvider = normalizeProviderAlias(activeModelId.slice(0, slashIndex));
                      const providerModelId = activeModelId.slice(slashIndex + 1);
                      selected = availableModels.find((model) => (
                        model.id === providerModelId
                        && providersEquivalent(model.provider, explicitProvider || inferredProvider)
                      ));
                    }

                    return selected
                      ? toProviderScopedModelKey(selected.provider, selected.id)
                      : activeModelId;
                  })()}
                  onModelChange={changeModel}
                  getModelSelectionKey={(model) => toProviderScopedModelKey(model.provider, model.id)}
                  disabled={isStreaming || modelsLoading}
                  loading={modelsLoading}
                  triggerIcon={showWorkspaceConversationSelect ? <Bot size={14} /> : undefined}
                  triggerClassName={showWorkspaceConversationSelect ? 'chat-workspace-model-trigger' : undefined}
                />
                {!embedded && (
                  <button
                    className="btn btn-secondary btn-sm btn-icon"
                    onClick={toggleFullscreen}
                    title={isFullscreen ? 'Exit full screen' : 'Full screen'}
                  >
                    {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                  </button>
                )}
                <button className="btn btn-sm btn-secondary chat-new-chat-btn" onClick={startFreshConversation} title="Start a new conversation" disabled={isReadOnly}>
                  <MessageSquarePlus size={14} className="chat-new-chat-icon" aria-hidden="true" />
                  <span className="chat-new-chat-label">New Chat</span>
                </button>
              </div>
            </div>

            {/* Messages */}
            {!isMessagesCollapsed && (
            <div className="chat-messages" ref={chatMessagesRef} onScroll={handleScroll}>
              {isConversationSwitchLoading ? (
                renderMessageBubbleSkeletons()
              ) : activeConversation.messages.length === 0 && !isStreaming ? (
                <div className="chat-welcome">
                  <h3>Start a conversation</h3>
                  <p>Ask questions about your indexed code, query databases, or get help with your systems.</p>
                </div>
              ) : (
                <>
                  {activeConversation.messages.map((msg, idx) => {
                    const bp = branchPointsByIndex.get(idx);
                    // Branches are anchored on user messages only — never
                    // surface the nav on assistant rows even if a stale
                    // assistant-indexed branch exists in the DB.
                    const hasBranches = !!(bp && bp.branches.length > 0 && msg.role === 'user');
                    const msgKey = `msg-${idx}`;
                    return (
                    <div key={msgKey} className={`chat-branch-wrapper chat-branch-wrapper-${msg.role}${hasBranches ? ' chat-branch-wrapper-has-branches' : ''}`}>
                    <div className={`chat-message chat-message-${msg.role}`}>
                      <div className="chat-message-content" key={editingMessageIdx === idx ? 'editing' : 'viewing'}>
                        {editingMessageIdx === idx ? (
                          <>
                            <div
                              contentEditable
                              suppressContentEditableWarning
                              className="chat-message-text chat-message-user-text chat-edit-input"
                              onInput={(e) => {
                                // Convert innerHTML back to plain text with newlines
                                const el = e.currentTarget;
                                // Get text content but preserve line breaks
                                const text = el.innerText || '';
                                setEditMessageContent(text);
                              }}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                  e.preventDefault();
                                  submitEditMessage();
                                } else if (e.key === 'Escape') {
                                  cancelEditMessage();
                                }
                              }}
                              ref={(el) => {
                                if (el && el.innerHTML === '') {
                                  // Convert newlines to <br> for display
                                  el.innerHTML = editMessageContent
                                    .replace(/&/g, '&amp;')
                                    .replace(/</g, '&lt;')
                                    .replace(/>/g, '&gt;')
                                    .replace(/\n/g, '<br>');
                                  // Move cursor to end
                                  const range = document.createRange();
                                  range.selectNodeContents(el);
                                  range.collapse(false);
                                  const sel = window.getSelection();
                                  sel?.removeAllRanges();
                                  sel?.addRange(range);
                                  el.focus();
                                }
                              }}
                            />
                            <div className="chat-message-footer">
                              <span className="chat-message-time">
                                {formatChatTimestamp(msg.timestamp)}
                              </span>
                            </div>
                          </>
                        ) : (
                          <>
                            {/* Render chronological events if available */}
                            {msg.role === 'assistant' && msg.events && msg.events.length > 0 ? (
                              <>
                                {(() => {
                                  // Render events chronologically, merging only ADJACENT reasoning events
                                  const result: React.ReactNode[] = [];
                                  let pendingReasoning = '';
                                  let pendingReasoningParts: ReasoningPart[] = [];
                                  let pendingReasoningTools: ActiveToolCall[] = [];
                                  let pendingReasoningDurationSeconds: number | undefined;
                                  let reasoningBlockCount = 0;

                                  const flushReasoning = () => {
                                    if (!pendingReasoning) return;
                                    reasoningBlockCount++;
                                    result.push(
                                      <ReasoningDisplay
                                        key={`reasoning-${reasoningBlockCount}`}
                                        content={pendingReasoning}
                                        isComplete={true}
                                        durationSeconds={pendingReasoningDurationSeconds}
                                        parts={pendingReasoningParts.length > 0 ? pendingReasoningParts : undefined}
                                        toolCalls={pendingReasoningTools.length > 0 ? pendingReasoningTools : undefined}
                                        workspaceId={workspaceId}
                                        conversationId={activeConversation.id}
                                        onOpenWorkspaceFile={onOpenWorkspaceFile}
                                      />
                                    );
                                    pendingReasoning = '';
                                    pendingReasoningParts = [];
                                    pendingReasoningTools = [];
                                    pendingReasoningDurationSeconds = undefined;
                                  };

                                  for (let evIdx = 0; evIdx < msg.events.length; evIdx++) {
                                    const ev = msg.events[evIdx];
                                    if (ev.type === 'reasoning') {
                                      // Accumulate adjacent reasoning
                                      pendingReasoning += (pendingReasoning ? '\n\n' : '') + ev.content;
                                      if (typeof ev.duration_seconds === 'number') {
                                        pendingReasoningDurationSeconds = ev.duration_seconds;
                                      }
                                      const lastPart = pendingReasoningParts[pendingReasoningParts.length - 1];
                                      if (lastPart && lastPart.type === 'text') {
                                        lastPart.text = (lastPart.text || '') + (lastPart.text ? '\n\n' : '') + ev.content;
                                      } else {
                                        pendingReasoningParts.push({ type: 'text', text: ev.content });
                                      }
                                    } else if (ev.type === 'tool' && pendingReasoning) {
                                      // Tool immediately following reasoning — embed in current reasoning block
                                      const tc: ActiveToolCall = {
                                        tool: ev.tool,
                                        input: ev.input,
                                        output: ev.output,
                                        presentation: ev.presentation,
                                        connection: ev.connection,
                                        status: 'complete' as const,
                                      };
                                      pendingReasoningTools.push(tc);
                                      pendingReasoningParts.push({ type: 'tool', toolCall: tc });
                                    } else {
                                      // Content or standalone tool breaks reasoning adjacency
                                      flushReasoning();
                                      if (ev.type === 'tool' && showToolCalls) {
                                        result.push(
                                          <div key={`event-${evIdx}`} className="chat-tool-calls">
                                            <ToolCallDisplay
                                              toolCall={{
                                                tool: ev.tool,
                                                input: ev.input,
                                                output: ev.output,
                                                presentation: ev.presentation,
                                                connection: ev.connection,
                                                status: 'complete'
                                              }}
                                              defaultExpanded={false}
                                              conversationId={activeConversation.id}
                                              workspaceId={workspaceId}
                                              siblingEvents={msg.events}
                                              onOpenWorkspaceFile={onOpenWorkspaceFile}
                                            />
                                          </div>
                                        );
                                      } else if (ev.type === 'content') {
                                        result.push(
                                          <div key={`event-${evIdx}`} className="chat-message-text markdown-content">
                                            <MemoizedMarkdown content={ev.content} />
                                          </div>
                                        );
                                      }
                                    }
                                  }
                                  // Flush any trailing reasoning
                                  flushReasoning();

                                  return result;
                                })()}
                              </>
                            ) : (
                              <>
                                {msg.role === 'user' ? (
                                  <>
                                    {(() => {
                                      const { text, attachments } = parseMessageContent(msg.content);
                                      return (
                                        <>
                                          {attachments.length > 0 && <MessageAttachments attachments={attachments} onImageClick={setModalImageUrl} />}
                                          {text && (
                                            <div className="chat-message-text chat-message-user-text">
                                              <LinkifiedText text={text} />
                                            </div>
                                          )}
                                        </>
                                      );
                                    })()}
                                  </>
                                ) : (
                                  <div className="chat-message-text markdown-content">
                                    <MemoizedMarkdown content={parseMessageContent(msg.content).text} />
                                  </div>
                                )}
                              </>
                            )}
                            <div className="chat-message-footer">
                              <span className="chat-message-time">
                                {formatChatTimestamp(msg.timestamp)}
                              </span>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                    {/* Action bar — flat on pane background, outside the bubble */}
                    {(() => {
                      const isEditing = editingMessageIdx === idx;
                      const activeBranchId = activeConversation.active_branch_id;
                      const branchNav = hasBranches && bp ? (() => {
                        const livePathOptionId = `__current__:${bp.branch_point_index}`;
                        const hasLivePathOption = !activeBranchId && activeConversation.messages.length > bp.branch_point_index;
                        const allOptions = [
                          ...bp.branches.map(b => ({ id: b.id, label: b.created_by_username || 'Branch' })),
                          ...(hasLivePathOption ? [{ id: livePathOptionId, label: 'Current' }] : []),
                        ];
                        const newestBranch = bp.branches.length > 0 ? bp.branches[bp.branches.length - 1] : null;
                        const inferredCurrentBranchId = newestBranch?.parent_branch_id ?? null;
                        const lineageBranchId = findLineageBranchIdForPoint(branchesById, activeBranchId, bp.branch_point_index);
                        // 1. Active branch matches this point → use it
                        // 2. Active branch ancestry reaches this point → use that ancestor branch
                        // 3. If on the live path, show the synthetic current option
                        // 4. Remembered selection for this point → use it when on the live path
                        // 5. If on live path (active is null), infer from newest branch's parent
                        // 6. Fallback to last (most recent) branch
                        const bpIdx = bp.branch_point_index;
                        let matchIdx = lineageBranchId
                          ? allOptions.findIndex(o => o.id === lineageBranchId)
                          : -1;
                        if (matchIdx < 0 && hasLivePathOption) {
                          matchIdx = allOptions.findIndex(o => o.id === livePathOptionId);
                        }
                        if (matchIdx < 0 && branchSelections[bpIdx]) {
                          matchIdx = allOptions.findIndex(o => o.id === branchSelections[bpIdx]);
                        }
                        if (matchIdx < 0 && inferredCurrentBranchId) {
                          matchIdx = allOptions.findIndex(o => o.id === inferredCurrentBranchId);
                        }
                        const currentOptionIdx = matchIdx >= 0 ? matchIdx : allOptions.length - 1;
                        return (
                          <span className="chat-branch-nav">
                            <button className="chat-branch-nav-btn" onClick={() => { if (currentOptionIdx > 0 && !branchSwitching) switchBranch(allOptions[currentOptionIdx - 1].id); }} disabled={currentOptionIdx <= 0 || branchSwitching} aria-label="Previous branch">
                              <ChevronLeft size={12} />
                            </button>
                            <span className="chat-branch-nav-label">{currentOptionIdx + 1}/{allOptions.length}</span>
                            <button className="chat-branch-nav-btn" onClick={() => { if (currentOptionIdx < allOptions.length - 1 && !branchSwitching) switchBranch(allOptions[currentOptionIdx + 1].id); }} disabled={currentOptionIdx >= allOptions.length - 1 || branchSwitching} aria-label="Next branch">
                              <ChevronRight size={12} />
                            </button>
                          </span>
                        );
                      })() : null;

                      const isCopied = copiedMessageIdx === idx;
                      // Only show the restore banner on the branch-point message for the active branch
                      const showBanner = inputBanner && bp && activeBranchId && bp.branches.some(b => b.id === activeBranchId);

                      if (msg.role === 'user') {
                        return (
                          <>
                            {isEditing && editMessageAttachments.length > 0 && (
                              <div className="chat-edit-preview-list">
                                {editMessageAttachments.map(att => (
                                  <div key={att.id} className="attachment-item">
                                    {att.type === 'image' && att.preview ? (
                                      <div className="attachment-image-preview">
                                        <img src={att.preview} alt={att.name} />
                                      </div>
                                    ) : (
                                      <div className="attachment-file-preview">
                                        {att.filePath ? <Link size={20} /> : <FileText size={20} />}
                                      </div>
                                    )}
                                    <div className="attachment-info">
                                      <span className="attachment-name" title={att.name}>{att.name}</span>
                                      <span className="attachment-size">{formatAttachmentSize(att.size)}</span>
                                    </div>
                                    <button type="button" className="attachment-remove" onClick={() => setEditMessageAttachments(editMessageAttachments.filter(a => a.id !== att.id))}>
                                      <X size={16} />
                                    </button>
                                  </div>
                                ))}
                              </div>
                            )}
                            <div className={`chat-message-actions chat-message-actions-right${isEditing ? ' visible' : ''}`}>
                            {isEditing ? (
                              <>
                                <span className="chat-message-actions-spacer" />
                                <button className="chat-action-text-btn primary" onClick={submitEditMessage}>Send</button>
                                <button className="chat-action-text-btn" onClick={cancelEditMessage}>Cancel</button>
                                <div className="chat-edit-attachments-wrapper">
                                  <FileAttachment
                                    attachments={editMessageAttachments}
                                    onAttachmentsChange={setEditMessageAttachments}
                                  />
                                </div>
                              </>
                            ) : (
                              <span className="chat-message-hover-actions">
                                <button className="chat-action-icon-btn" onClick={() => copyMessageText(idx, msg.content)} title={isCopied ? 'Copied!' : 'Copy message'}>
                                  {isCopied ? <Check size={12} /> : <Copy size={12} />}
                                </button>
                                {!isStreaming && !isReadOnly && (
                                  <button className="chat-action-icon-btn" onClick={() => { const parsed = parseMessageContent(msg.content); startEditMessage(idx, parsed.text, parsed.attachments); }} title="Edit and resend">
                                    <Pencil size={12} />
                                  </button>
                                )}
                                {!isStreaming && !isReadOnly && (
                                  <button className="chat-action-icon-btn" onClick={() => replayFromMessage(idx)} title="Replay from this message">
                                    <RefreshCw size={12} />
                                  </button>
                                )}
                                {!isStreaming && !isReadOnly && (
                                  <button
                                    className="chat-action-icon-btn"
                                    onClick={() => setPendingDeleteIdx(pendingDeleteIdx === idx ? null : idx)}
                                    title={msg.snapshot_restore ? 'Delete message and restore workspace snapshot' : 'Delete message'}
                                  >
                                    <Trash2 size={12} />
                                  </button>
                                )}
                              </span>
                            )}
                            {branchNav}
                          </div>
                            {pendingDeleteIdx === idx && (
                              <div className="chat-message-banner-row chat-message-banner-row-right">
                                <div className="chat-branch-restore-banner">
                                  <span>{msg.snapshot_restore ? 'Delete message and restore workspace snapshot?' : 'Delete this message and all messages after it?'}</span>
                                  <button className="chat-branch-restore-btn confirm" onClick={() => { setPendingDeleteIdx(null); deleteFromMessage(idx); }}>Confirm</button>
                                  <button className="chat-branch-restore-btn dismiss" onClick={() => setPendingDeleteIdx(null)}>Cancel</button>
                                </div>
                              </div>
                            )}
                            {showBanner && (
                              <div className="chat-message-banner-row chat-message-banner-row-right">
                                {inputBanner}
                              </div>
                            )}
                          </>
                        );
                      } else {
                        return (
                          <>
                            <div className="chat-message-actions chat-message-actions-left">
                              <span className="chat-message-hover-actions">
                                <button className="chat-action-icon-btn" onClick={() => copyMessageText(idx, msg.content)} title={isCopied ? 'Copied!' : 'Copy message'}>
                                  {isCopied ? <Check size={12} /> : <Copy size={12} />}
                                </button>
                                {!isStreaming && !isReadOnly && (
                                  <button className="chat-action-icon-btn" onClick={() => replayFromMessage(idx)} title="Replay from this message">
                                    <RefreshCw size={12} />
                                  </button>
                                )}
                                {!isStreaming && !isReadOnly && (
                                  <button
                                    className="chat-action-icon-btn"
                                    onClick={() => setPendingDeleteIdx(pendingDeleteIdx === idx ? null : idx)}
                                    title={msg.snapshot_restore ? 'Delete reply and restore workspace snapshot' : 'Delete reply'}
                                  >
                                    <Trash2 size={12} />
                                  </button>
                                )}
                                {showPromptDebugButton && (
                                  <button className="chat-action-icon-btn" onClick={() => openPromptDebugForAssistantMessage(idx)} title="Open prompt debug for this assistant reply">
                                    <Bug size={12} />
                                  </button>
                                )}
                              </span>
                              {branchNav && <span className="chat-message-actions-spacer" />}
                              {branchNav}
                            </div>
                            {pendingDeleteIdx === idx && (
                              <div className="chat-message-banner-row chat-message-banner-row-left">
                                <div className="chat-branch-restore-banner">
                                  <span>{msg.snapshot_restore ? 'Delete reply and restore workspace snapshot?' : 'Delete this reply and all messages after it?'}</span>
                                  <button className="chat-branch-restore-btn confirm" onClick={() => { setPendingDeleteIdx(null); deleteFromMessage(idx); }}>Confirm</button>
                                  <button className="chat-branch-restore-btn dismiss" onClick={() => setPendingDeleteIdx(null)}>Cancel</button>
                                </div>
                              </div>
                            )}
                            {showBanner && (
                              <div className="chat-message-banner-row chat-message-banner-row-left">
                                {inputBanner}
                              </div>
                            )}
                          </>
                        );
                      }
                    })()}
                    </div>
                  );
                  })}

                  {/* Streaming assistant message - uses consolidated segments for performance */}
                  {isStreaming && consolidatedSegments.length > 0 && (
                    <div className="chat-message chat-message-assistant chat-message-streaming-active">
                      <div className="chat-message-content">
                        {consolidatedSegments.map((segment, idx) => (
                          <StreamingSegmentDisplay
                            key={`segment-${idx}-${segment.type}`}
                            segment={segment}
                            showToolCalls={showToolCalls}
                            workspaceId={workspaceId}
                            conversationId={activeConversation.id}
                            onOpenWorkspaceFile={onOpenWorkspaceFile}
                          />
                        ))}
                        <div className="chat-message-streaming">
                          {(() => {
                            const runningTool = consolidatedSegments.find(
                              seg => seg.type === 'tool' && seg.toolCall?.status === 'running'
                            );
                            if (runningTool && runningTool.type === 'tool') {
                              const lines = runningTool.toolCall?.generating_lines;
                              return lines
                                ? `Running tool... (${lines} lines)`
                                : 'Running tool...';
                            }
                            // Check for a tool that's being generated (has generating_lines but
                            // hasn't started executing yet - no input means still generating args)
                            const generatingTool = consolidatedSegments.find(
                              seg => seg.type === 'tool' && !seg.toolCall?.input && seg.toolCall?.generating_lines
                            );
                            if (generatingTool && generatingTool.type === 'tool') {
                              return `Generating... (${generatingTool.toolCall?.generating_lines} lines)`;
                            }
                            if (consolidatedSegments.some(seg => seg.type === 'reasoning' && !seg.isComplete)) {
                              return 'Reasoning...';
                            }
                            return 'Generating...';
                          })()}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Loading indicator - only when nothing is streaming yet */}
                  {isStreaming && consolidatedSegments.length === 0 && (
                    <div className="chat-message chat-message-assistant">
                      <div className="chat-message-content">
                        <div className="chat-typing-indicator">
                          <span></span><span></span><span></span>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Continue prompt - shows for max iterations, connection error, or interrupted task */}
              {!isStreaming && activeConversation && (
                // Show continue when:
                // 1. Last message is assistant AND (hitMaxIterations OR isConnectionError)
                // 2. OR there's an interrupted task (from server restart)
                ((activeConversation.messages.length > 0 &&
                  activeConversation.messages[activeConversation.messages.length - 1].role === 'assistant' &&
                  (hitMaxIterations || isConnectionError)) ||
                 interruptedTask) && !isReadOnly && (
                <div className="chat-continue-inline">
                  <span className="chat-continue-text">
                    Conversation interrupted, <button className="chat-continue-link" onClick={continueConversation}>continue?</button>
                  </span>
                </div>
              ))}

              <div ref={messagesEndRef} />
            </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="chat-error">
                {error}
                <div className="chat-error-actions">
                  {isConnectionError && lastSentMessage && (
                    <button
                      className="btn-resend"
                      onClick={resendMessage}
                      disabled={Boolean(sendReadinessBlockReason)}
                      title={sendReadinessBlockReason || 'Re-send'}
                    >
                      Re-send
                    </button>
                  )}
                  <button onClick={() => { setError(null); setIsConnectionError(false); }}>×</button>
                </div>
              </div>
            )}

            <ResizeHandle
              direction="vertical"
              className="resize-handle resize-handle-vertical chat-resize-handle"
              onResize={handleResizeInputArea}
              collapsed={isInputAreaCollapsed ? 'after' : isMessagesCollapsed ? 'before' : undefined}
              onExpand={isInputAreaCollapsed ? expandInputArea : isMessagesCollapsed ? expandMessages : undefined}
            />

            {/* Input Area */}
            {!isInputAreaCollapsed && (
            <div
              className={`chat-input-area ${isManualResize ? 'manual-resize' : ''} ${autoResizeState ? 'auto-resizing' : ''} ${autoResizeState === 'shrinking' ? 'shrinking' : ''}`.trim().replace(/\s+/g, ' ')}
              style={isMessagesCollapsed ? { flex: 1, minHeight: 'auto' } : { height: `${inputAreaHeight}px`, minHeight: `${inputAreaHeight}px` }}
            >
              {isReadOnly && (
                <div className="chat-readonly-note" role="status">
                  {effectiveReadOnlyMessage}
                </div>
              )}
              <div className="chat-input-wrapper">
                <FileAttachment
                  attachments={attachments}
                  onAttachmentsChange={setAttachments}
                  disabled={isReadOnly || isStreaming}
                />
                <textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  placeholder={isReadOnly ? effectiveReadOnlyMessage : 'Ask a question or paste an image (Ctrl+V)...'}
                  rows={1}
                  className="chat-input"
                  disabled={isReadOnly}
                />
                {isStreaming ? (
                  <div className="chat-input-inline-actions">
                    {showInlineToolSelector && (
                      <ToolSelectorDropdown
                        availableTools={effectiveAvailableTools}
                        selectedToolIds={resolvedEffectiveToolIdSet}
                        onToggleTool={handleToggleInlineTool}
                        selectedToolGroupIds={effectiveToolGroupIdSet}
                        onToggleToolGroup={handleToggleInlineToolGroup}
                        toolGroups={effectiveToolGroups}
                        disabled={effectiveSavingTools}
                        readOnly={false}
                        saving={effectiveSavingTools}
                        title="Workspace Tools"
                      />
                    )}
                    <button
                      type="button"
                      className="btn chat-stop-btn-inline"
                      onClick={stopStreaming}
                      title="Stop generating"
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <rect x="6" y="6" width="12" height="12" rx="2"></rect>
                      </svg>
                    </button>
                  </div>
                ) : (
                  !isReadOnly && (
                    <div className="chat-input-inline-actions">
                      {showInlineToolSelector && (
                        <ToolSelectorDropdown
                          availableTools={effectiveAvailableTools}
                          selectedToolIds={resolvedEffectiveToolIdSet}
                          onToggleTool={handleToggleInlineTool}
                          selectedToolGroupIds={effectiveToolGroupIdSet}
                          onToggleToolGroup={handleToggleInlineToolGroup}
                          toolGroups={effectiveToolGroups}
                          disabled={effectiveSavingTools}
                          readOnly={false}
                          saving={effectiveSavingTools}
                          title="Workspace Tools"
                        />
                      )}
                      {(inputValue.trim() || attachments.length > 0) && (
                        <button
                          type="button"
                          className="btn chat-send-btn-inline"
                          onClick={sendMessage}
                          disabled={!activeConversation || Boolean(sendReadinessBlockReason) || !contextUsage.hasHeadroom}
                          title={sendReadinessBlockReason
                            ? sendReadinessBlockReason
                            : contextUsage.hasHeadroom
                              ? 'Send message'
                              : `Context headroom too low (${contextUsage.projectedInputPercent}%)`}
                        >
                          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="12" y1="19" x2="12" y2="5"></line>
                            <polyline points="5 12 12 5 19 12"></polyline>
                          </svg>
                        </button>
                      )}
                    </div>
                  )
                )}
              </div>
            </div>
            )}
          </>
        ) : (
          <div className="chat-no-conversation">
            <h2>Welcome to Chat</h2>
            <p>Select a conversation or start a new one to begin.</p>
            <button className="btn" onClick={createNewConversation}>
              New Conversation
            </button>
          </div>
        )}
      </div>

      {showMembersModal && activeConversation && conversationOwnerId && (
        <MemberManagementModal
          isOpen={showMembersModal}
          onClose={() => setShowMembersModal(false)}
          members={conversationMembers}
          onSave={handleSaveMembers}
          allUsers={shareableConversationUsers}
          ownerId={conversationOwnerId}
          entityType="conversation"
          formatUserLabel={formatUserLabel}
          saving={savingMembers}
        />
      )}

      {showPromptDebugModal && (
        <div
          className="modal-overlay"
          onClick={closePromptDebugModal}
          role="presentation"
        >
          <div
            className="modal modal-large"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="prompt-debug-modal-title"
          >
            <div className="modal-header">
              <h3 id="prompt-debug-modal-title">Provider Prompt Debug</h3>
              <button className="modal-close" onClick={closePromptDebugModal} aria-label="Close prompt debug modal">
                &times;
              </button>
            </div>
            <div className="modal-body" style={{ maxHeight: '70vh', overflowY: 'auto' }}>
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: '0.9rem', opacity: 0.8 }}>
                  Captured provider input calls for this conversation.
                </div>
              </div>

              {promptDebugError && (
                <div className="chat-error" style={{ marginBottom: 12 }}>{promptDebugError}</div>
              )}

              {!promptDebugLoading && chronologicalPromptDebugRecords.length === 0 && !promptDebugError && (
                <div style={{ fontSize: '0.95rem', opacity: 0.8 }}>
                  No prompt-debug records yet for this conversation.
                </div>
              )}

              {chronologicalPromptDebugRecords.map((record, recordIdx) => {
                const createdAt = new Date(record.created_at).toLocaleString();
                const renderedMessages = Array.isArray(record.rendered_provider_messages)
                  ? record.rendered_provider_messages
                  : [];
                return (
                  <div key={record.id} style={{ marginBottom: 24 }}>
                    {recordIdx > 0 && (
                      <hr style={{ border: 'none', borderTop: '2px solid var(--color-border)', margin: '20px 0' }} />
                    )}

                    {/* Record metadata */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14, alignItems: 'center' }}>
                      <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.03em', color: '#2451a6', background: '#dbe7f7', padding: '4px 10px', borderRadius: 6 }}>
                        {record.provider}
                      </span>
                      <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text-primary)' }}>
                        {record.model}
                      </span>
                      <span style={{ fontSize: 12, color: '#2451a6', background: '#dbe7f7', padding: '3px 8px', borderRadius: 4 }}>
                        {record.prompt_token_count ?? 0} tokens
                      </span>
                      <span style={{ fontSize: 12, color: 'var(--color-text-muted)', background: 'var(--color-surface-hover)', padding: '3px 8px', borderRadius: 4 }}>
                        {record.mode}
                      </span>
                      <span style={{ fontSize: 12, color: 'var(--color-text-muted)', background: 'var(--color-surface-hover)', padding: '3px 8px', borderRadius: 4 }}>
                        {record.request_kind}
                      </span>
                      <span style={{ fontSize: 12, color: 'var(--color-text-muted)', marginLeft: 'auto' }}>
                        {createdAt}
                      </span>
                    </div>

                    {/* Provider messages — exact payload sent to LLM, in order */}
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: 'var(--color-text-muted)' }}>
                      Messages sent to provider ({renderedMessages.length})
                    </div>
                    {renderedMessages.map((message, messageIdx) => {
                      const messageRole = String((message as Record<string, unknown>)?.role ?? 'unknown').toLowerCase();
                      const messageLabel = messageRole.toUpperCase();
                      const messageContent = formatPromptMessageContent((message as Record<string, unknown>)?.content);
                      const lineCount = messageContent ? messageContent.split('\n').length : 0;
                      const messageKey = `${record.id}-${messageIdx}`;
                      const copied = copiedPromptMessageKey === messageKey;

                      const badgeColor = messageRole === 'system'
                        ? '#6b3fa0'
                        : messageRole === 'user'
                          ? '#2451a6'
                          : messageRole === 'assistant'
                            ? '#1d6a41'
                            : 'var(--color-text-muted)';
                      const borderColor = messageRole === 'system'
                        ? '#dccff0'
                        : messageRole === 'user'
                          ? '#c8d6ec'
                          : messageRole === 'assistant'
                            ? '#c4ddd0'
                            : 'var(--color-border)';
                      const headerBg = messageRole === 'system'
                        ? '#f5f0fa'
                        : messageRole === 'user'
                          ? '#eef2f8'
                          : messageRole === 'assistant'
                            ? '#edf7f1'
                            : 'var(--color-surface-hover)';

                      /* System and tool messages are collapsible to reduce noise. */
                      if (messageRole === 'system' || messageRole === 'tool') {
                        return (
                          <details key={messageKey} open={messageRole === 'system'} style={{ marginBottom: 10, border: `1px solid ${borderColor}`, borderRadius: 8, overflow: 'hidden' }}>
                            <summary style={{ listStyle: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 8px', background: headerBg }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.04em', color: badgeColor }}>
                                  {messageLabel}
                                </span>
                                <span style={{ fontWeight: 600, fontSize: 12 }}>Message {messageIdx + 1}</span>
                                <span style={{ opacity: 0.7, fontSize: 12 }}>({lineCount} lines)</span>
                              </div>
                              <button
                                style={{
                                  border: '1px solid var(--color-border)',
                                  background: 'transparent',
                                  borderRadius: 6,
                                  padding: '2px 6px',
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  gap: 4,
                                  fontSize: 12,
                                  lineHeight: 1.1,
                                  cursor: 'pointer',
                                }}
                                onClick={(event) => {
                                  event.preventDefault();
                                  event.stopPropagation();
                                  void copyPromptText(messageKey, messageContent);
                                }}
                                title="Copy message"
                                aria-label="Copy message"
                              >
                                {copied ? <Check size={12} /> : <Copy size={12} />}
                                <span>{copied ? 'Copied' : 'Copy'}</span>
                              </button>
                            </summary>
                            <pre style={{ whiteSpace: 'pre-wrap', margin: 0, padding: 12, fontSize: 13 }}>{messageContent}</pre>
                          </details>
                        );
                      }

                      /* User/assistant/other messages shown inline */
                      return (
                        <div key={messageKey} style={{ marginBottom: 10, border: `1px solid ${borderColor}`, borderRadius: 8, overflow: 'hidden' }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, padding: '4px 8px', background: headerBg }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.04em', color: badgeColor }}>
                                {messageLabel}
                              </span>
                              <span style={{ fontWeight: 600, fontSize: 12 }}>Message {messageIdx + 1}</span>
                            </div>
                            <button
                              style={{
                                border: '1px solid var(--color-border)',
                                background: 'transparent',
                                borderRadius: 6,
                                padding: '2px 6px',
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: 4,
                                fontSize: 12,
                                lineHeight: 1.1,
                                cursor: 'pointer',
                              }}
                              onClick={() => void copyPromptText(messageKey, messageContent)}
                              title="Copy message"
                              aria-label="Copy message"
                            >
                              {copied ? <Check size={12} /> : <Copy size={12} />}
                              <span>{copied ? 'Copied' : 'Copy'}</span>
                            </button>
                          </div>
                          <pre style={{ whiteSpace: 'pre-wrap', margin: 0, padding: 12, fontSize: 13 }}>{messageContent || '(empty)'}</pre>
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Image Modal */}
      {modalImageUrl && (
        <div
          className="image-modal-overlay"
          onClick={() => setModalImageUrl(null)}
          onKeyDown={(e) => e.key === 'Escape' && setModalImageUrl(null)}
        >
          <div className="image-modal-content" onClick={(e) => e.stopPropagation()}>
            <button
              className="image-modal-close"
              onClick={() => setModalImageUrl(null)}
              title="Close"
            >
              <X size={24} />
            </button>
            <img src={modalImageUrl} alt="Enlarged view" />
          </div>
        </div>
      )}
    </div>
  );
}
