import { useState, useEffect, useRef, useCallback, useMemo, memo, isValidElement, type ReactNode, type CSSProperties } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { diffLines } from 'diff';
import Chart from 'chart.js/auto';
import type { Chart as ChartInstance, ChartConfiguration } from 'chart.js';
import { Copy, Check, Pencil, Slash, Trash2, Maximize2, Minimize2, X, AlertCircle, RefreshCw, Play, FileText, Bug, ChevronDown, ChevronRight, ChevronLeft, Bot, MessageSquare, MessageSquarePlus, BrainCircuit, Clock, Diff, Wrench, Database, Search, Terminal, BarChart3, Globe, Code, FolderSearch, Image as ImageIcon, Link, Share2 } from 'lucide-react';
import { api } from '@/api';
import type { Conversation, ChatMessage, ChatTask, User, UserDirectoryEntry, ContentPart, ConversationMember, UserSpaceAvailableTool, ProviderPromptDebugRecord, MessageEvent, WorkspaceChatStateResponse, LlmProviderWire, UserSpaceFile, UserSpaceSnapshotFileDiff, ConversationBranchKind, ConversationBranchPointInfo, ConversationBranchSummary, AvailableModel, RetryVisualizationRequest, ToolConnectionRef } from '@/types';
import { FileAttachment, attachmentsToContentParts, formatAttachmentSize, resizeAttachmentImageDataUrl, type AttachmentFile } from './FileAttachment';
import { ModelSelector } from './ModelSelector';
import { ResizeHandle } from './ResizeHandle';
import { calculateConversationContextUsage } from '@/utils/contextUsage';
import { fetchUserSpaceToolCatalog, getUserSpaceGroupToolIds } from '@/utils/userSpaceTools';
import {
  KNOWN_PROVIDER_KEYS,
  modelIdentifierInList,
  resolvedModelSelectionKey,
  resolveProviderModelSelection,
  toProviderScopedModelKey,
} from '@/utils/modelProviders';
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
import type { FileDiffOverlayEntry } from './shared/FileDiffOverlay';
import { MemberManagementButton } from './shared/MemberManagementButton';
import { MemberManagementModal } from './shared/MemberManagementModal';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';
import { ToolSelectorDropdown, type ToolGroupInfo } from './shared/ToolSelectorDropdown';
import { UserSpaceFileDiffView, formatDiffStatus } from './shared/UserSpaceFileDiffView';
import { useAvailableModels } from '@/contexts/AvailableModelsContext';

const CHAT_DIAGNOSTIC_COMMAND_TOOL_ID = 'run_chat_diagnostic_command';

const CHAT_BUILT_IN_TOOLS: UserSpaceAvailableTool[] = [
  {
    id: CHAT_DIAGNOSTIC_COMMAND_TOOL_ID,
    name: 'Local terminal',
    tool_type: 'built-in',
    description: 'Run read-only diagnostic shell commands in a chat sandbox.',
  },
  {
    id: 'web_search',
    name: 'Web search',
    tool_type: 'built-in',
    description: 'Search the web from a chat sandbox.',
  },
  {
    id: 'web_read_pdf',
    name: 'Read PDF',
    tool_type: 'built-in',
    description: 'Extract targeted text from a web PDF.',
  },
  {
    id: 'web_browse',
    name: 'Web browse',
    tool_type: 'built-in',
    description: 'Browse a specific URL from a chat sandbox.',
  },
];

const CHAT_BUILT_IN_TOOL_IDS = CHAT_BUILT_IN_TOOLS.map((tool) => tool.id);
const CHAT_BUILT_IN_TOOL_ID_SET = new Set(CHAT_BUILT_IN_TOOL_IDS);
const WEB_BROWSE_TOOL_ID = 'web_browse';
const WEB_READ_PDF_TOOL_ID = 'web_read_pdf';
const WORKSPACE_BUILT_IN_TOOL_ID_SET = new Set(['web_search', 'web_read_pdf', 'web_browse']);
const WORKSPACE_BUILT_IN_TOOLS = CHAT_BUILT_IN_TOOLS.filter((tool) => WORKSPACE_BUILT_IN_TOOL_ID_SET.has(tool.id));
const VISIBLE_CHAT_BUILT_IN_TOOLS = CHAT_BUILT_IN_TOOLS.filter((tool) => tool.id !== WEB_READ_PDF_TOOL_ID);
const VISIBLE_WORKSPACE_BUILT_IN_TOOLS = WORKSPACE_BUILT_IN_TOOLS.filter((tool) => tool.id !== WEB_READ_PDF_TOOL_ID);

function getToolVisualName(toolId: string): string {
  return toolId === WEB_READ_PDF_TOOL_ID ? WEB_BROWSE_TOOL_ID : toolId;
}

function maskHiddenToolNames(text: string): string {
  return text.split(WEB_READ_PDF_TOOL_ID).join(WEB_BROWSE_TOOL_ID);
}

function normalizeDisabledBuiltInToolIds(disabledToolIds: string[]): string[] {
  const normalized = new Set(
    disabledToolIds.filter((id) => CHAT_BUILT_IN_TOOL_ID_SET.has(id)),
  );

  // Keep read_pdf as an internal implementation detail of web_browse when browse is enabled.
  if (!normalized.has(WEB_BROWSE_TOOL_ID)) {
    normalized.delete(WEB_READ_PDF_TOOL_ID);
  }

  return CHAT_BUILT_IN_TOOL_IDS.filter((id) => normalized.has(id));
}

interface CodeBlockProps {
  inline?: boolean;
  className?: string;
  children?: ReactNode | ReactNode[];
}

interface BranchRenderGroup {
  anchorIndex: number;
  selectionKey: string;
  sourceBranchPointIndex: number;
  branches: ConversationBranchSummary[];
}

function getConversationBranchAnchorIndex(
  branchPointIndex: number,
  branchKind: ConversationBranchKind | null | undefined,
  messageCount: number,
): number {
  if (messageCount <= 0) return 0;
  const target = branchKind === 'delete' ? branchPointIndex - 1 : branchPointIndex;
  return Math.max(0, Math.min(target, messageCount - 1));
}

function getConversationBranchSelectionKey(branchPointIndex: number, anchorIndex: number): string {
  return `${branchPointIndex}:${anchorIndex}`;
}

function resolveDefaultSelectedToolIds(
  selectedToolIds: string[],
  selectedToolGroupIds: string[],
  availableTools: UserSpaceAvailableTool[],
): string[] {
  if (selectedToolIds.length > 0 || selectedToolGroupIds.length > 0) {
    return selectedToolIds;
  }
  return availableTools.map((tool) => tool.id);
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

  const isLatexBlock = useMemo(() => {
    const normalized = language.toLowerCase();
    return normalized === 'latex' || normalized === 'tex' || normalized === 'katex' || normalized === 'math';
  }, [language]);

  const renderedLatex = useMemo(() => {
    if (!isLatexBlock || !codeText.trim()) {
      return { html: null as string | null, error: null as string | null };
    }

    try {
      const html = katex.renderToString(codeText, {
        displayMode: true,
        throwOnError: true,
        strict: 'warn',
      });

      return { html, error: null as string | null };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Invalid LaTeX expression.';

      const html = katex.renderToString(codeText, {
        displayMode: true,
        throwOnError: false,
        strict: 'ignore',
      });

      return {
        html,
        error: errorMessage,
      };
    }
  }, [codeText, isLatexBlock]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(codeText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code block:', err);
    }
  }, [codeText]);

  if (isLatexBlock && renderedLatex.html) {
    return (
      <div className="markdown-codeblock markdown-codeblock-latex">
        <div className="markdown-codeblock-header">
          <span className="markdown-codeblock-lang">latex</span>
          <button
            type="button"
            className={`markdown-codeblock-copy ${copied ? 'is-copied' : ''}`}
            onClick={handleCopy}
            aria-label={copied ? 'LaTeX copied' : 'Copy LaTeX'}
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
        <div className="markdown-latex-render" dangerouslySetInnerHTML={{ __html: renderedLatex.html }} />
        {renderedLatex.error && (
          <div className="markdown-latex-error">
            <p className="markdown-latex-error-message">Rendered with LaTeX parse warning: {renderedLatex.error}</p>
            <pre className="markdown-latex-source"><code>{codeText}</code></pre>
          </div>
        )}
      </div>
    );
  }

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

// Some models emit display math glued to surrounding text, e.g.
//   $$h(x,t) =
//   \begin{cases} ...
//   \end{cases}$$
// remark-math wants the `$$` fences on their own lines. We only split when a
// line contains exactly one `$$` (so an opener/closer of a multi-line block);
// inline `$$x$$` pairs and lines inside fenced code blocks are left alone.
function normalizeMarkdownMathFences(content: string): string {
  let inFence = false;
  return content
    .split('\n')
    .map((line) => {
      if (/^\s*(```|~~~)/.test(line)) {
        inFence = !inFence;
        return line;
      }
      if (inFence) return line;
      const count = (line.match(/\$\$/g) ?? []).length;
      if (count !== 1) return line;
      return line
        .replace(/^(\s*)\$\$(\S.*)$/, '$1$$$$\n$1$2')
        .replace(/^(.*\S)\$\$\s*$/, '$1\n$$$$');
    })
    .join('\n');
}

// Memoized markdown component to prevent re-parsing on every render
// Only re-renders when content actually changes
export const MemoizedMarkdown = memo(function MemoizedMarkdown({ content }: { content: string }) {
  const normalized = useMemo(() => normalizeMarkdownMathFences(content), [content]);
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={markdownComponents}
    >
      {normalized}
    </ReactMarkdown>
  );
});

// Tool call info for display during streaming
export interface ActiveToolCall {
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
  | { type: 'content'; channel?: 'final'; content: string }
  | { type: 'tool'; channel?: 'commentary'; toolCall: ActiveToolCall }
  | { type: 'reasoning'; channel?: 'analysis'; content: string; durationSeconds?: number };

type ChatEventChannel = 'analysis' | 'commentary' | 'final';

function getChatEventChannel(event: { type?: string; channel?: unknown }): ChatEventChannel {
  if (event.channel === 'analysis' || event.channel === 'commentary' || event.channel === 'final') {
    return event.channel;
  }
  if (event.type === 'reasoning') return 'analysis';
  if (event.type === 'tool') return 'commentary';
  return 'final';
}

function isVisualizationToolName(toolName?: string): boolean {
  return toolName === 'create_chart' || toolName === 'create_datatable';
}

function isVisualizationToolCall(toolCall?: Pick<ActiveToolCall, 'tool'>): boolean {
  return isVisualizationToolName(toolCall?.tool);
}

function normalizeStreamingToolEvent(event: any): StreamingRenderEvent {
  const nestedToolCall = event?.toolCall && typeof event.toolCall === 'object' ? event.toolCall : undefined;
  const hasTopLevelOutput = event && typeof event === 'object' && Object.prototype.hasOwnProperty.call(event, 'output');
  const hasNestedOutput = nestedToolCall && Object.prototype.hasOwnProperty.call(nestedToolCall, 'output');

  return {
    type: 'tool' as const,
    channel: 'commentary' as const,
    toolCall: {
      tool: event?.tool ?? nestedToolCall?.tool ?? '',
      input: event?.input ?? nestedToolCall?.input,
      output: event?.output ?? nestedToolCall?.output,
      presentation: event?.presentation ?? nestedToolCall?.presentation,
      connection: event?.connection ?? nestedToolCall?.connection,
      status: hasTopLevelOutput || hasNestedOutput ? 'complete' : 'running',
      generating_lines: event?.generating_lines ?? nestedToolCall?.generating_lines,
    },
  };
}

// Parse table metadata from SQL tool output
// Format: <!--TABLEDATA:{"columns":[...],"rows":[...]}-->
interface TableData {
  columns: string[];
  rows: unknown[][];
}

const RETRY_CONTEXT_EVENT_LIMIT = 12;
const RETRY_CONTEXT_OUTPUT_LIMIT = 12000;

type VisualizationRetrySiblingEvent = {
  type: string;
  tool?: string;
  input?: Record<string, unknown>;
  output?: string;
  connection?: ToolConnectionRef;
};

function truncateRetryContextOutput(output: string | undefined): string | undefined {
  if (!output || output.length <= RETRY_CONTEXT_OUTPUT_LIMIT) return output;
  return `${output.slice(0, RETRY_CONTEXT_OUTPUT_LIMIT)}\n...[truncated ${output.length - RETRY_CONTEXT_OUTPUT_LIMIT} chars]`;
}

function buildVisualizationRetryContextEvents(siblingEvents?: VisualizationRetrySiblingEvent[]): RetryVisualizationRequest['context_events'] {
  if (!siblingEvents) return [];
  return siblingEvents
    .filter((event) => event.type === 'tool')
    .slice(-RETRY_CONTEXT_EVENT_LIMIT)
    .map((event) => ({
      type: event.type,
      tool: event.tool,
      input: event.input,
      output: truncateRetryContextOutput(event.output),
      connection: event.connection,
    }));
}

function parseTableMetadata(output: string): { tableData: TableData | null; displayText: string } {
  const startMarker = '<!--TABLEDATA:';
  const endMarker = '-->';
  if (!output.startsWith(startMarker)) {
    return { tableData: null, displayText: output };
  }

  const lineEndIdx = output.indexOf('\n');
  const metadataSearchEnd = lineEndIdx === -1 ? output.length - 1 : lineEndIdx - 1;
  const endIdx = output.lastIndexOf(endMarker, metadataSearchEnd);
  if (endIdx < startMarker.length) {
    return { tableData: null, displayText: output };
  }

  const jsonStr = output.substring(startMarker.length, endIdx);

  try {
    const tableData = JSON.parse(jsonStr) as TableData;
    let contentStart = endIdx + endMarker.length;
    if (output.startsWith('\r\n', contentStart)) {
      contentStart += 2;
    } else if (output[contentStart] === '\r' || output[contentStart] === '\n') {
      contentStart += 1;
    }

    const displayText = output.slice(contentStart);
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

const MODEL_REMOVED_FROM_CHAT_MODELS_MESSAGE = 'Selected model has been removed from Chat Models. Select another model to continue.';

function branchStartsGeneration(branchKind: ConversationBranchKind): boolean {
  return branchKind === 'edit' || branchKind === 'replay';
}

const resolveConversationModelSelection = resolveProviderModelSelection<AvailableModel>;

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
  // Workspace state snapshots can occasionally arrive without message payloads;
  // never let an empty incoming list wipe a populated local conversation.
  if (current.messages.length > 0 && incoming.messages.length === 0) {
    return {
      ...current,
      ...incoming,
      model: incomingIsNewer ? (incoming.model || current.model) : (current.model || incoming.model),
      messages: current.messages,
    };
  }
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

// Helper component to parse URLs and render them as clickable links
export const LinkifiedText = memo(function LinkifiedText({ text }: { text: string }) {
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
  const chartInstanceRef = useRef<ChartInstance | null>(null);
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
      if (canvasRef.current && containerRef.current) {
        // Destroy previous chart if exists
        if (chartInstanceRef.current) {
          chartInstanceRef.current.destroy();
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

        chartInstanceRef.current = new Chart(canvasRef.current, config as ChartConfiguration);
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
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
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

interface ParsedUserSpaceWriteResult {
  toolName: 'upsert_userspace_file' | 'patch_userspace_file' | 'move_userspace_file' | 'delete_userspace_file';
  op: 'upsert' | 'patch' | 'move' | 'delete';
  status: string;
  path: string;
  oldPath: string | null;
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

interface ParsedUserSpaceWriteBatch {
  toolName: 'upsert_userspace_file' | 'patch_userspace_file' | 'move_userspace_file' | 'delete_userspace_file';
  isBatch: boolean;
  // The "primary" entry, used for legacy single-file rendering paths.
  // For a batched payload this is just the first entry; for a single-file
  // payload this is the only entry.
  primary: ParsedUserSpaceWriteResult;
  entries: ParsedUserSpaceWriteResult[];
  summary: {
    total: number;
    persisted: number;
    rejected: number;
    noChanges: number;
    withViolations: number;
  };
  aggregateMessage: string;
  aggregateStatus: string;
}

const USERSPACE_WRITE_TOOL_NAMES = new Set([
  'upsert_userspace_file',
  'patch_userspace_file',
  'move_userspace_file',
  'delete_userspace_file',
]);
const USERSPACE_DIFFABLE_TOOL_NAMES = new Set(['upsert_userspace_file', 'patch_userspace_file']);
const USERSPACE_WRITE_DIFF_CACHE_MAX_ENTRIES = 100;

interface ParsedUserSpaceReadResult {
  path: string;
  message: string;
  content: string | null;
  isRejected: boolean;
  status: string;
  startLine: number | null;
  endLine: number | null;
  totalLines: number | null;
}

interface ParsedUserSpaceListEntry {
  path: string;
  entryType: 'file' | 'directory';
  sizeBytes: number;
  updatedAt: string | null;
}

interface ParsedUserSpaceListResult {
  message: string;
  status: string;
  count: number;
  entries: ParsedUserSpaceListEntry[];
  fileCount: number;
  directoryCount: number;
  totalSizeBytes: number;
  truncated: boolean;
}

interface ParsedUserSpaceReadBatch {
  isBatch: boolean;
  primary: ParsedUserSpaceReadResult;
  entries: ParsedUserSpaceReadResult[];
  summary: {
    total: number;
    read: number;
    rejected: number;
  };
  aggregateMessage: string;
  aggregateStatus: string;
}

interface ParsedUserspaceToolPayload {
  toolName: string;
  status: string;
  path: string;
  message: string;
  error: string | null;
  rejected: boolean;
  persisted: boolean;
  persistedWithViolations: boolean;
  retryable: boolean;
  failureClass: string;
  nextBestTool: string;
  actionRequired: string | null;
  writeSignature: string | null;
  filePath: string | null;
  fileContent: string | null;
  startLine: number | null;
  endLine: number | null;
  totalLines: number | null;
  details: string[];
}

function findJsonKeyIndex(source: string, key: string, fromIndex = 0): number {
  return source.indexOf(`"${key}"`, fromIndex);
}

function findValueStart(source: string, key: string, fromIndex = 0): number {
  const keyIndex = findJsonKeyIndex(source, key, fromIndex);
  if (keyIndex === -1) return -1;
  const colonIndex = source.indexOf(':', keyIndex);
  return colonIndex === -1 ? -1 : colonIndex + 1;
}

function readJsonStringLiteral(source: string, fromIndex: number): { value: string; nextIndex: number } | null {
  const quoteIndex = source.indexOf('"', fromIndex);
  if (quoteIndex === -1) return null;

  let value = '';
  let escaped = false;

  for (let index = quoteIndex + 1; index < source.length; index += 1) {
    const ch = source[index];

    if (escaped) {
      switch (ch) {
        case 'n': value += '\n'; break;
        case 'r': value += '\r'; break;
        case 't': value += '\t'; break;
        case 'b': value += '\b'; break;
        case 'f': value += '\f'; break;
        case '"': value += '"'; break;
        case '\\': value += '\\'; break;
        case 'u': {
          const hex = source.slice(index + 1, index + 5);
          if (/^[0-9a-fA-F]{4}$/.test(hex)) {
            value += String.fromCharCode(Number.parseInt(hex, 16));
            index += 4;
          }
          break;
        }
        default:
          value += ch;
          break;
      }
      escaped = false;
      continue;
    }

    if (ch === '\\') {
      escaped = true;
      continue;
    }

    if (ch === '"') {
      return { value, nextIndex: index + 1 };
    }

    value += ch;
  }

  return { value, nextIndex: source.length };
}

function readJsonLiteral(source: string, fromIndex: number): { value: string; nextIndex: number } | null {
  let index = fromIndex;
  while (index < source.length && /\s/.test(source[index])) index += 1;
  if (index >= source.length) return null;

  const terminators = new Set([',', '}', '\n', '\r']);
  let value = '';
  while (index < source.length) {
    const ch = source[index];
    if (terminators.has(ch)) break;
    value += ch;
    index += 1;
  }

  return { value: value.trim(), nextIndex: index };
}

function readJsonBoolean(source: string, fromIndex: number): boolean | null {
  const literal = readJsonLiteral(source, fromIndex);
  if (!literal) return null;
  if (literal.value === 'true') return true;
  if (literal.value === 'false') return false;
  return null;
}

function readJsonNumber(source: string, fromIndex: number): number | null {
  const literal = readJsonLiteral(source, fromIndex);
  if (!literal) return null;
  const value = Number(literal.value);
  return Number.isFinite(value) ? value : null;
}

function extractJsonObjectSource(source: string, key: string): string | null {
  const startIndex = findValueStart(source, key);
  if (startIndex === -1) return null;

  const objectStart = source.indexOf('{', startIndex);
  if (objectStart === -1) return null;

  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = objectStart; index < source.length; index += 1) {
    const ch = source[index];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }

    if (ch === '"') {
      inString = true;
      continue;
    }

    if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        return source.slice(objectStart, index + 1);
      }
    }
  }

  return source.slice(objectStart);
}

function readJsonStringField(source: string, key: string): string | null {
  const valueStart = findValueStart(source, key);
  if (valueStart === -1) return null;
  const parsed = readJsonStringLiteral(source, valueStart);
  return parsed ? parsed.value : null;
}

function readJsonBooleanField(source: string, key: string): boolean | null {
  const valueStart = findValueStart(source, key);
  if (valueStart === -1) return null;
  return readJsonBoolean(source, valueStart);
}

function readJsonNumberField(source: string, key: string): number | null {
  const valueStart = findValueStart(source, key);
  if (valueStart === -1) return null;
  return readJsonNumber(source, valueStart);
}

function readJsonStringArrayField(source: string, key: string): string[] {
  const arrayStart = findValueStart(source, key);
  if (arrayStart === -1) return [];

  const bracketStart = source.indexOf('[', arrayStart);
  if (bracketStart === -1) return [];

  const values: string[] = [];
  let index = bracketStart + 1;

  while (index < source.length) {
    while (index < source.length && /[\s,]/.test(source[index])) index += 1;
    if (index >= source.length || source[index] === ']') break;

    if (source[index] === '"') {
      const parsed = readJsonStringLiteral(source, index);
      if (!parsed) break;
      if (parsed.value.trim().length > 0) values.push(parsed.value);
      index = parsed.nextIndex;
      continue;
    }

    const literal = readJsonLiteral(source, index);
    if (!literal) break;
    if (literal.value.trim().length > 0) values.push(literal.value.trim());
    index = literal.nextIndex;
  }

  return values;
}

function readJsonObjectArrayField(source: string, key: string): string[] {
  const arrayStart = findValueStart(source, key);
  if (arrayStart === -1) return [];

  const bracketStart = source.indexOf('[', arrayStart);
  if (bracketStart === -1) return [];

  const values: string[] = [];
  let objectStart = -1;
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = bracketStart + 1; index < source.length; index += 1) {
    const ch = source[index];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }

    if (ch === '"') {
      inString = true;
      continue;
    }

    if (ch === '{') {
      if (depth === 0) objectStart = index;
      depth += 1;
      continue;
    }

    if (ch === '}') {
      if (depth > 0) {
        depth -= 1;
        if (depth === 0 && objectStart !== -1) {
          values.push(source.slice(objectStart, index + 1));
          objectStart = -1;
        }
      }
      continue;
    }

    if (ch === ']' && depth === 0) {
      break;
    }
  }

  if (depth > 0 && objectStart !== -1) {
    values.push(source.slice(objectStart));
  }

  return values;
}

function isLikelyTruncatedToolOutput(source: string): boolean {
  return /\.\.\. \[[\d,]+ characters omitted\] \.\.\./.test(source)
    || source.includes('... (truncated)');
}

function parseUserspaceToolPayload(output?: string | null): ParsedUserspaceToolPayload | null {
  if (!output) return null;

  try {
    const parsed = JSON.parse(output) as Record<string, unknown>;
    if (!parsed || typeof parsed !== 'object') return null;

    const file = parsed.file && typeof parsed.file === 'object' ? parsed.file as Record<string, unknown> : null;
    const diagnostics = parsed.diagnostics && typeof parsed.diagnostics === 'object'
      ? parsed.diagnostics as Record<string, unknown>
      : null;
    const path = typeof parsed.path === 'string' && parsed.path.trim()
      ? parsed.path.trim()
      : typeof file?.path === 'string' && file.path.trim()
        ? file.path.trim()
        : '';

    return {
      toolName: typeof parsed.tool === 'string' ? parsed.tool : '',
      status: typeof parsed.status === 'string' ? parsed.status : '',
      path,
      message: typeof parsed.message === 'string' ? parsed.message.trim() : '',
      error: typeof parsed.error === 'string' && parsed.error.trim() ? parsed.error.trim() : null,
      rejected: parsed.rejected === true,
      persisted: parsed.persisted === true,
      persistedWithViolations: parsed.persisted_with_violations === true,
      retryable: parsed.retryable !== false,
      failureClass: typeof parsed.failure_class === 'string' ? parsed.failure_class : '',
      nextBestTool: typeof parsed.next_best_tool === 'string' ? parsed.next_best_tool : '',
      actionRequired: typeof parsed.action_required === 'string' && parsed.action_required.trim()
        ? parsed.action_required.trim()
        : null,
      writeSignature: typeof parsed.write_signature === 'string' && parsed.write_signature.trim()
        ? parsed.write_signature.trim()
        : null,
      filePath: typeof file?.path === 'string' && file.path.trim() ? file.path.trim() : null,
      fileContent: typeof file?.content === 'string' ? file.content : null,
      startLine: typeof diagnostics?.start_line === 'number' ? diagnostics.start_line : null,
      endLine: typeof diagnostics?.end_line === 'number' ? diagnostics.end_line : null,
      totalLines: typeof diagnostics?.total_lines === 'number' ? diagnostics.total_lines : null,
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
    // Fall through to tolerant field extraction for truncated payloads.
  }

  const parsedFile = extractJsonObjectSource(output, 'file');
  const parsedDiagnostics = extractJsonObjectSource(output, 'diagnostics');

  const path = readJsonStringField(output, 'path') || (parsedFile ? readJsonStringField(parsedFile, 'path') : null) || '';
  if (!path) return null;

  return {
    toolName: readJsonStringField(output, 'tool') || '',
    status: readJsonStringField(output, 'status') || '',
    path,
    message: readJsonStringField(output, 'message') || '',
    error: readJsonStringField(output, 'error'),
    rejected: readJsonBooleanField(output, 'rejected') === true,
    persisted: readJsonBooleanField(output, 'persisted') === true,
    persistedWithViolations: readJsonBooleanField(output, 'persisted_with_violations') === true,
    retryable: readJsonBooleanField(output, 'retryable') !== false,
    failureClass: readJsonStringField(output, 'failure_class') || '',
    nextBestTool: readJsonStringField(output, 'next_best_tool') || '',
    actionRequired: readJsonStringField(output, 'action_required'),
    writeSignature: readJsonStringField(output, 'write_signature'),
    filePath: parsedFile ? (readJsonStringField(parsedFile, 'path') ?? null) : null,
    fileContent: parsedFile ? (readJsonStringField(parsedFile, 'content') ?? null) : null,
    startLine: parsedDiagnostics ? readJsonNumberField(parsedDiagnostics, 'start_line') : null,
    endLine: parsedDiagnostics ? readJsonNumberField(parsedDiagnostics, 'end_line') : null,
    totalLines: parsedDiagnostics ? readJsonNumberField(parsedDiagnostics, 'total_lines') : null,
    details: [
      ...readJsonStringArrayField(output, 'contract_violations'),
      ...readJsonStringArrayField(output, 'warnings'),
    ],
  };
}

function parseUserspaceListToolResult(toolName: string, output?: string | null): ParsedUserSpaceListResult | null {
  if (toolName !== 'list_userspace_files') return null;
  if (!output) return null;

  const buildResult = (
    entries: ParsedUserSpaceListEntry[],
    message: string,
    status: string,
    count: number,
    truncated: boolean,
  ): ParsedUserSpaceListResult => {
    let fileCount = 0;
    let directoryCount = 0;
    let totalSizeBytes = 0;

    entries.forEach((entry) => {
      if (entry.entryType === 'directory') {
        directoryCount += 1;
      } else {
        fileCount += 1;
        totalSizeBytes += entry.sizeBytes;
      }
    });

    entries.sort((a, b) => {
      if (a.entryType !== b.entryType) {
        return a.entryType === 'directory' ? -1 : 1;
      }
      return a.path.localeCompare(b.path);
    });

    return {
      message,
      status,
      count,
      entries,
      fileCount,
      directoryCount,
      totalSizeBytes,
      truncated,
    };
  };

  let parsed: Record<string, unknown> | null = null;
  try {
    const candidate = JSON.parse(output);
    if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
      parsed = candidate as Record<string, unknown>;
    }
  } catch {
    parsed = null;
  }

  if (parsed) {
    const filesRaw = Array.isArray(parsed.files) ? (parsed.files as unknown[]) : [];
    const entries: ParsedUserSpaceListEntry[] = [];

    for (const item of filesRaw) {
      if (!item || typeof item !== 'object') continue;
      const rec = item as Record<string, unknown>;
      const path = typeof rec.path === 'string' ? rec.path.trim() : '';
      if (!path) continue;
      const rawType = typeof rec.entry_type === 'string' ? rec.entry_type : 'file';
      const entryType: 'file' | 'directory' = rawType === 'directory' ? 'directory' : 'file';
      const sizeBytes = typeof rec.size_bytes === 'number' && Number.isFinite(rec.size_bytes)
        ? Math.max(0, rec.size_bytes as number)
        : 0;
      const updatedAt = typeof rec.updated_at === 'string' ? rec.updated_at : null;
      entries.push({ path, entryType, sizeBytes, updatedAt });
    }

    const count = typeof parsed.count === 'number' ? (parsed.count as number) : entries.length;
    const message = typeof parsed.message === 'string' ? (parsed.message as string) : '';
    const status = typeof parsed.status === 'string' ? (parsed.status as string) : '';

    return buildResult(entries, message, status, count, false);
  }

  const entries = readJsonObjectArrayField(output, 'files')
    .map((itemSource): ParsedUserSpaceListEntry | null => {
      const path = readJsonStringField(itemSource, 'path');
      if (!path) return null;
      const rawType = readJsonStringField(itemSource, 'entry_type') || 'file';
      return {
        path: path.trim(),
        entryType: rawType === 'directory' ? 'directory' : 'file',
        sizeBytes: Math.max(0, readJsonNumberField(itemSource, 'size_bytes') ?? 0),
        updatedAt: readJsonStringField(itemSource, 'updated_at'),
      };
    })
    .filter((entry): entry is ParsedUserSpaceListEntry => Boolean(entry && entry.path));

  const count = readJsonNumberField(output, 'count') ?? entries.length;
  const message = readJsonStringField(output, 'message') || '';
  const status = readJsonStringField(output, 'status') || '';
  const truncated = isLikelyTruncatedToolOutput(output);

  if (entries.length === 0 && count === 0 && !message && !status) return null;

  return buildResult(entries, message, status, count, truncated);
}

function parseUserspaceReadToolResult(toolName: string, output?: string | null): ParsedUserSpaceReadResult | null {
  if (toolName !== 'read_userspace_file') return null;
  const parsed = parseUserspaceToolPayload(output);
  if (!parsed) return null;

  const fileContent = parsed.fileContent;
  const path = parsed.path || parsed.filePath || '';
  if (!path) return null;

  return {
    path,
    message: parsed.message,
    content: fileContent,
    isRejected: parsed.rejected || parsed.status.includes('rejected'),
    status: parsed.status,
    startLine: parsed.startLine,
    endLine: parsed.endLine,
    totalLines: parsed.totalLines,
  };
}

function parseUserspaceReadBatch(toolName: string, output?: string | null): ParsedUserSpaceReadBatch | null {
  if (toolName !== 'read_userspace_file') return null;
  if (!output) return null;

  let parsedJson: Record<string, unknown> | null = null;
  try {
    const candidate = JSON.parse(output);
    if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
      parsedJson = candidate as Record<string, unknown>;
    }
  } catch {
    parsedJson = null;
  }

  const isBatch = Boolean(parsedJson && parsedJson.batch === true && Array.isArray(parsedJson.files));

  if (!isBatch) {
    const single = parseUserspaceReadToolResult(toolName, output);
    if (!single) return null;
    return {
      isBatch: false,
      primary: single,
      entries: [single],
      summary: {
        total: 1,
        read: single.isRejected ? 0 : 1,
        rejected: single.isRejected ? 1 : 0,
      },
      aggregateMessage: single.message,
      aggregateStatus: single.status,
    };
  }

  const filesRaw = parsedJson!.files as unknown[];
  const entries: ParsedUserSpaceReadResult[] = [];
  for (const item of filesRaw) {
    if (!item || typeof item !== 'object') continue;
    const e = item as Record<string, unknown>;
    const fileObj = e.file && typeof e.file === 'object' ? (e.file as Record<string, unknown>) : null;
    const entryPath =
      (typeof e.path === 'string' && (e.path as string).trim()) ? (e.path as string).trim()
      : (typeof fileObj?.path === 'string' && (fileObj!.path as string).trim()) ? (fileObj!.path as string).trim()
      : '';
    if (!entryPath) continue;
    const status = typeof e.status === 'string' ? e.status : '';
    const fileContent = fileObj && typeof fileObj.content === 'string' ? (fileObj.content as string) : null;
    const diagnostics = e.diagnostics && typeof e.diagnostics === 'object' ? (e.diagnostics as Record<string, unknown>) : null;
    entries.push({
      path: entryPath,
      message: typeof e.message === 'string' ? (e.message as string).trim() : '',
      content: fileContent,
      isRejected: e.rejected === true || status.includes('rejected'),
      status,
      startLine: typeof diagnostics?.start_line === 'number' ? (diagnostics.start_line as number) : null,
      endLine: typeof diagnostics?.end_line === 'number' ? (diagnostics.end_line as number) : null,
      totalLines: typeof diagnostics?.total_lines === 'number' ? (diagnostics.total_lines as number) : null,
    });
  }

  if (entries.length === 0) return null;

  const summarySrc = parsedJson!.summary && typeof parsedJson!.summary === 'object'
    ? (parsedJson!.summary as Record<string, unknown>)
    : {};
  const numericOrCount = (key: string, fallback: () => number): number => {
    const v = summarySrc[key];
    return typeof v === 'number' ? v : fallback();
  };

  const summary = {
    total: numericOrCount('total', () => entries.length),
    read: numericOrCount('read', () => entries.filter((e) => !e.isRejected).length),
    rejected: numericOrCount('rejected', () => entries.filter((e) => e.isRejected).length),
  };

  const aggregateMessage = typeof parsedJson!.message === 'string' ? (parsedJson!.message as string) : '';
  const aggregateStatus = typeof parsedJson!.status === 'string' ? (parsedJson!.status as string) : '';

  return {
    isBatch: true,
    primary: entries[0],
    entries,
    summary,
    aggregateMessage,
    aggregateStatus,
  };
}

function normalizeUserspaceSnippetContent(content: string): string {
  const lines = content.split('\n');
  if (lines.length === 0) return content;

  const numberedLineCount = lines.filter((line) => /^\s*\d+\s*\|\s?/.test(line)).length;
  if (numberedLineCount === 0 || numberedLineCount < Math.ceil(lines.length / 2)) {
    return content;
  }

  return lines.map((line) => line.replace(/^\s*\d+\s*\|\s?/, '')).join('\n');
}

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
  if (!USERSPACE_WRITE_TOOL_NAMES.has(toolName)) {
    return null;
  }

  const parsed = parseUserspaceToolPayload(output);
  if (!parsed) {
    return null;
  }

  const status = parsed.status;
  const file = parsed.fileContent != null
    ? {
        path: parsed.filePath || parsed.path,
        content: parsed.fileContent,
        updated_at: parsed.writeSignature || '',
      } as UserSpaceFile
    : null;
  const path = parsed.path;
  if (!path) return null;

  const op: 'upsert' | 'patch' | 'move' | 'delete' =
    toolName === 'upsert_userspace_file' ? 'upsert'
    : toolName === 'patch_userspace_file' ? 'patch'
    : toolName === 'move_userspace_file' ? 'move'
    : 'delete';

  return {
    toolName: toolName as ParsedUserSpaceWriteResult['toolName'],
    op,
    status,
    path,
    oldPath: null,
    message: parsed.message,
    error: parsed.error,
    actionRequired: parsed.actionRequired,
    writeSignature: parsed.writeSignature,
    persisted: parsed.persisted || status.startsWith('persisted'),
    rejected: parsed.rejected || status.includes('rejected'),
    noChanges: status === 'no_changes',
    file,
    details: parsed.details,
  };
}

function entryStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
}

function parseUserspaceWriteBatch(toolName: string, output?: string | null): ParsedUserSpaceWriteBatch | null {
  if (!USERSPACE_WRITE_TOOL_NAMES.has(toolName)) return null;
  if (!output) return null;

  const opForTool: 'upsert' | 'patch' | 'move' | 'delete' =
    toolName === 'upsert_userspace_file' ? 'upsert'
    : toolName === 'patch_userspace_file' ? 'patch'
    : toolName === 'move_userspace_file' ? 'move'
    : 'delete';

  // Try strict JSON parse first to detect a batched `files` collection.
  let parsedJson: Record<string, unknown> | null = null;
  try {
    const candidate = JSON.parse(output);
    if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
      parsedJson = candidate as Record<string, unknown>;
    }
  } catch {
    parsedJson = null;
  }

  const isBatch = Boolean(parsedJson && parsedJson.batch === true && Array.isArray(parsedJson.files));

  if (!isBatch) {
    const single = parseUserspaceWriteToolResult(toolName, output);
    if (!single) return null;
    return {
      toolName: toolName as ParsedUserSpaceWriteBatch['toolName'],
      isBatch: false,
      primary: single,
      entries: [single],
      summary: {
        total: 1,
        persisted: single.persisted ? 1 : 0,
        rejected: single.rejected ? 1 : 0,
        noChanges: single.noChanges ? 1 : 0,
        withViolations: 0,
      },
      aggregateMessage: single.message,
      aggregateStatus: single.status,
    };
  }

  const filesRaw = parsedJson!.files as unknown[];
  const entries: ParsedUserSpaceWriteResult[] = [];
  for (const item of filesRaw) {
    if (!item || typeof item !== 'object') continue;
    const e = item as Record<string, unknown>;
    const entryOpRaw = typeof e.op === 'string' ? e.op : opForTool;
    const entryOp: 'upsert' | 'patch' | 'move' | 'delete' =
      entryOpRaw === 'upsert' || entryOpRaw === 'patch' || entryOpRaw === 'move' || entryOpRaw === 'delete'
        ? entryOpRaw
        : opForTool;
    const entryStatus = typeof e.status === 'string' ? e.status : '';
    const fileObj = e.file && typeof e.file === 'object' ? (e.file as Record<string, unknown>) : null;
    const entryPath =
      (typeof e.new_path === 'string' && e.new_path.trim()) ? (e.new_path as string).trim()
      : (typeof e.path === 'string' && e.path.trim()) ? (e.path as string).trim()
      : (typeof fileObj?.path === 'string' && (fileObj!.path as string).trim()) ? (fileObj!.path as string).trim()
      : '';
    if (!entryPath) continue;
    const writeSig = typeof e.write_signature === 'string' && e.write_signature.trim() ? e.write_signature.trim() : null;
    const fileForEntry: UserSpaceFile | null = fileObj && typeof fileObj.content === 'string'
      ? {
          path: typeof fileObj.path === 'string' && fileObj.path.trim() ? (fileObj.path as string).trim() : entryPath,
          content: fileObj.content as string,
          updated_at: writeSig || '',
        } as UserSpaceFile
      : null;
    const entryToolName: ParsedUserSpaceWriteResult['toolName'] =
      entryOp === 'upsert' ? 'upsert_userspace_file'
      : entryOp === 'patch' ? 'patch_userspace_file'
      : entryOp === 'move' ? 'move_userspace_file'
      : 'delete_userspace_file';
    entries.push({
      toolName: entryToolName,
      op: entryOp,
      status: entryStatus,
      path: entryPath,
      oldPath: typeof e.old_path === 'string' && (e.old_path as string).trim() ? (e.old_path as string).trim() : null,
      message: typeof e.message === 'string' ? (e.message as string).trim() : '',
      error: typeof e.error === 'string' && (e.error as string).trim() ? (e.error as string).trim() : null,
      actionRequired: typeof e.action_required === 'string' && (e.action_required as string).trim() ? (e.action_required as string).trim() : null,
      writeSignature: writeSig,
      persisted: e.persisted === true || entryStatus.startsWith('persisted'),
      rejected: e.rejected === true || entryStatus.includes('rejected'),
      noChanges: entryStatus === 'no_changes',
      file: fileForEntry,
      details: [
        ...entryStringList(e.contract_violations),
        ...entryStringList(e.warnings),
      ],
    });
  }

  if (entries.length === 0) return null;

  const summarySrc = parsedJson!.summary && typeof parsedJson!.summary === 'object'
    ? (parsedJson!.summary as Record<string, unknown>)
    : {};
  const numericOrCount = (key: string, fallback: () => number): number => {
    const v = summarySrc[key];
    return typeof v === 'number' ? v : fallback();
  };

  const summary = {
    total: numericOrCount('total', () => entries.length),
    persisted: numericOrCount('persisted', () => entries.filter((e) => e.persisted).length),
    rejected: numericOrCount('rejected', () => entries.filter((e) => e.rejected).length),
    noChanges: numericOrCount('no_changes', () => entries.filter((e) => e.noChanges).length),
    withViolations: numericOrCount('with_violations', () => entries.filter((e) => e.status === 'persisted_with_violations').length),
  };

  const aggregateMessage = typeof parsedJson!.message === 'string' ? (parsedJson!.message as string) : '';
  const aggregateStatus = typeof parsedJson!.status === 'string' ? (parsedJson!.status as string) : '';

  return {
    toolName: toolName as ParsedUserSpaceWriteBatch['toolName'],
    isBatch: true,
    primary: entries[0],
    entries,
    summary,
    aggregateMessage,
    aggregateStatus,
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
  allowRerun?: boolean;
  siblingEvents?: VisualizationRetrySiblingEvent[];
  messageId?: string;
  messageIndex?: number;
  eventIndex?: number;
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

const TERMINAL_TOOL_NAMES = new Set(['run_terminal_command', CHAT_DIAGNOSTIC_COMMAND_TOOL_ID]);
const TERMINAL_TOOL_CONNECTION_TYPES = new Set(['ssh_shell']);
const SQL_TOOL_CONNECTION_TYPES = new Set(['postgres', 'mysql', 'mssql', 'influxdb']);
const TERMINAL_PRESENTATION_KIND = 'terminal';
const USERSPACE_EXEC_RERUN_KIND = 'userspace_exec';
const CONVERSATION_TOOL_RERUN_KIND = 'conversation_tool';
const CHAT_DIAGNOSTIC_RERUN_KIND = 'chat_diagnostic';
const TERMINAL_RERUN_KINDS = new Set([
  USERSPACE_EXEC_RERUN_KIND,
  CONVERSATION_TOOL_RERUN_KIND,
  CHAT_DIAGNOSTIC_RERUN_KIND,
]);

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

function isSqlToolCall(toolCall: ActiveToolCall): boolean {
  const toolType = toolCall.connection?.tool_type?.trim().toLowerCase();
  return Boolean(toolType && SQL_TOOL_CONNECTION_TYPES.has(toolType));
}

function isConversationToolRerunnable(toolCall: ActiveToolCall): boolean {
  const toolType = toolCall.connection?.tool_type?.trim().toLowerCase();
  const connectionMode = normalizedPresentationValue(toolCall.connection?.connection_mode);
  return Boolean(
    toolCall.connection?.tool_config_id
    && (
      (toolType && TERMINAL_TOOL_CONNECTION_TYPES.has(toolType))
      || (toolType === 'odoo_shell' && connectionMode === 'ssh')
      || (toolType && SQL_TOOL_CONNECTION_TYPES.has(toolType))
    )
  );
}

function getPresentationRerunKind(toolCall: ActiveToolCall): string | null {
  const rerunKind = normalizedPresentationValue(toolCall.presentation?.rerun_kind);
  return TERMINAL_RERUN_KINDS.has(rerunKind) ? rerunKind : null;
}

function canRerunToolCall(toolCall: ActiveToolCall): boolean {
  if (getPresentationRerunKind(toolCall)) {
    return true;
  }
  if (toolCall.tool === CHAT_DIAGNOSTIC_COMMAND_TOOL_ID) {
    return true;
  }
  if (isConversationToolRerunnable(toolCall)) {
    return true;
  }
  return toolCall.tool === 'run_terminal_command';
}

function getTerminalRerunKind(toolCall: ActiveToolCall): string | null {
  const presentationRerunKind = getPresentationRerunKind(toolCall);
  if (presentationRerunKind) {
    return presentationRerunKind;
  }
  if (toolCall.tool === CHAT_DIAGNOSTIC_COMMAND_TOOL_ID) {
    return CHAT_DIAGNOSTIC_RERUN_KIND;
  }
  if (isConversationToolRerunnable(toolCall)) {
    return CONVERSATION_TOOL_RERUN_KIND;
  }
  return toolCall.tool === 'run_terminal_command' ? USERSPACE_EXEC_RERUN_KIND : null;
}

function decodeJsonStringFragment(fragment: string): string {
  let safeFragment = fragment.replace(/\.\.\. \(truncated\)\s*$/, '');
  safeFragment = safeFragment.replace(/\\(?:u[0-9a-fA-F]{0,3})?$/, '');
  try {
    return JSON.parse(`"${safeFragment}"`);
  } catch {
    return safeFragment
      .replace(/\\n/g, '\n')
      .replace(/\\r/g, '\r')
      .replace(/\\t/g, '\t')
      .replace(/\\"/g, '"')
      .replace(/\\\\/g, '\\');
  }
}

function extractJsonStringPropertyFragment(source: string, property: string): string | undefined {
  const marker = new RegExp(`"${property}"\\s*:\\s*"`);
  const match = marker.exec(source);
  if (!match) return undefined;

  const start = match.index + match[0].length;
  let escaped = false;
  let fragment = '';

  for (let index = start; index < source.length; index += 1) {
    const char = source[index];
    if (escaped) {
      fragment += char;
      escaped = false;
      continue;
    }
    if (char === '\\') {
      fragment += char;
      escaped = true;
      continue;
    }
    if (char === '"') {
      return decodeJsonStringFragment(fragment);
    }
    fragment += char;
  }

  return decodeJsonStringFragment(fragment);
}

function parseTruncatedTerminalOutput(output: string): ParsedTerminalOutput | null {
  const hasLegacyTruncationMarker = output.includes('... (truncated)');
  const hasOmittedCharsMarker = /\.\.\. \[[\d,]+ characters omitted\] \.\.\./.test(output);
  if (!hasLegacyTruncationMarker && !hasOmittedCharsMarker) return null;
  if (!/"exit_code"\s*:/.test(output)) return null;
  if (!/"(?:stdout|stderr|error)"\s*:/.test(output)) return null;

  const exitCodeMatch = /"exit_code"\s*:\s*(-?\d+)/.exec(output);
  if (!exitCodeMatch) return null;

  const status = extractJsonStringPropertyFragment(output, 'status') ?? 'unknown';
  const command = extractJsonStringPropertyFragment(output, 'command');
  const cwd = extractJsonStringPropertyFragment(output, 'cwd') ?? '.';
  const stdout = extractJsonStringPropertyFragment(output, 'stdout') ?? '';
  const stderr = extractJsonStringPropertyFragment(output, 'stderr') ?? '';
  const error = extractJsonStringPropertyFragment(output, 'error');

  return {
    status,
    command,
    cwd,
    exit_code: Number(exitCodeMatch[1]),
    stdout,
    stderr,
    error,
    timed_out: /"timed_out"\s*:\s*true/.test(output),
    truncated: true,
  };
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
    return parseTruncatedTerminalOutput(output);
  }
}

interface ParsedWebSearchResult {
  title: string;
  url: string;
  snippet: string;
  score?: number;
  favicon?: string;
  engine?: string;
}

interface ParsedWebSearchOutput {
  status: string;
  ok: boolean;
  blocked: boolean;
  error: string;
  query: string;
  provider: string;
  answer: string;
  results: ParsedWebSearchResult[];
  resultCount: number;
  engineUrl: string;
  durationMs?: number;
}

interface ParsedWebBrowseLink {
  url: string;
  text: string;
}

interface ParsedWebBrowseOutput {
  status: string;
  ok: boolean;
  error: string;
  url: string;
  requestedUrl: string;
  statusCode: number | null;
  title: string;
  text: string;
  textLength: number;
  truncated: boolean;
  links: ParsedWebBrowseLink[];
  consoleErrors: string[];
  durationMs?: number;
}

function parseWebSearchOutput(output: string | undefined | null): ParsedWebSearchOutput | null {
  if (!output) return null;
  try {
    const parsed = JSON.parse(output) as Record<string, unknown>;
    if (parsed.tool !== 'web_search') return null;

    const rawResults = Array.isArray(parsed.results) ? parsed.results : [];
    const results: ParsedWebSearchResult[] = [];
    for (const item of rawResults) {
      if (!item || typeof item !== 'object') continue;
      const record = item as Record<string, unknown>;
      const url = typeof record.url === 'string' ? record.url : '';
      const title = typeof record.title === 'string' ? record.title : '';
      if (!url || !title) continue;
      const snippet = typeof record.snippet === 'string' ? record.snippet : '';
      const result: ParsedWebSearchResult = { title, url, snippet };
      if (typeof record.score === 'number') result.score = record.score;
      if (typeof record.favicon === 'string' && record.favicon.trim()) {
        result.favicon = record.favicon;
      }
      if (typeof record.engine === 'string' && record.engine.trim()) {
        result.engine = record.engine;
      }
      results.push(result);
    }

    return {
      status: typeof parsed.status === 'string' ? parsed.status : 'unknown',
      ok: parsed.ok === true,
      blocked: parsed.blocked === true,
      error: typeof parsed.error === 'string' ? parsed.error : '',
      query: typeof parsed.query === 'string' ? parsed.query : '',
      provider: typeof parsed.provider === 'string' ? parsed.provider : '',
      answer: typeof parsed.answer === 'string' ? parsed.answer : '',
      results,
      resultCount: typeof parsed.result_count === 'number'
        ? parsed.result_count
        : results.length,
      engineUrl: typeof parsed.engine_url === 'string' ? parsed.engine_url : '',
      durationMs: typeof parsed.duration_ms === 'number' ? parsed.duration_ms : undefined,
    };
  } catch {
    return null;
  }
}

function parseWebBrowseOutput(output: string | undefined | null): ParsedWebBrowseOutput | null {
  if (!output) return null;
  try {
    const parsed = JSON.parse(output) as Record<string, unknown>;
    if (parsed.tool !== 'web_browse') return null;

    const rawLinks = Array.isArray(parsed.links) ? parsed.links : [];
    const links: ParsedWebBrowseLink[] = [];
    for (const item of rawLinks) {
      if (!item || typeof item !== 'object') continue;
      const record = item as Record<string, unknown>;
      const url = typeof record.url === 'string' ? record.url : '';
      if (!url) continue;
      const text = typeof record.text === 'string' ? record.text : '';
      links.push({ url, text });
    }

    const rawConsoleErrors = Array.isArray(parsed.console_errors) ? parsed.console_errors : [];
    const consoleErrors: string[] = [];
    for (const item of rawConsoleErrors) {
      if (typeof item === 'string' && item.trim()) consoleErrors.push(item);
    }

    const statusCodeRaw = parsed.status_code;
    let statusCode: number | null = null;
    if (typeof statusCodeRaw === 'number' && Number.isFinite(statusCodeRaw)) {
      statusCode = statusCodeRaw;
    }

    return {
      status: typeof parsed.status === 'string' ? parsed.status : 'unknown',
      ok: parsed.ok === true,
      error: typeof parsed.error === 'string' ? parsed.error : '',
      url: typeof parsed.url === 'string' ? parsed.url : '',
      requestedUrl: typeof parsed.requested_url === 'string' ? parsed.requested_url : '',
      statusCode,
      title: typeof parsed.title === 'string' ? parsed.title : '',
      text: typeof parsed.text === 'string' ? parsed.text : '',
      textLength: typeof parsed.text_length === 'number' ? parsed.text_length : 0,
      truncated: parsed.truncated === true,
      links,
      consoleErrors,
      durationMs: typeof parsed.duration_ms === 'number' ? parsed.duration_ms : undefined,
    };
  } catch {
    return null;
  }
}

interface ParsedWebReadPdfMatch {
  text: string;
  matchStart: number | null;
  matchEnd: number | null;
}

interface ParsedWebReadPdfOutput {
  ok: boolean;
  error: string;
  url: string;
  requestedUrl: string;
  query: string;
  text: string;
  textLength: number;
  textStartChar: number | null;
  textEndChar: number | null;
  truncated: boolean;
  matches: ParsedWebReadPdfMatch[];
  durationMs?: number;
}

function parsePartialWebReadPdfOutput(output: string): ParsedWebReadPdfOutput | null {
  const mode = readJsonStringField(output, 'mode');
  const tool = readJsonStringField(output, 'tool');
  if (mode !== 'pdf_read' && tool !== WEB_READ_PDF_TOOL_ID && tool !== WEB_BROWSE_TOOL_ID) return null;

  const text = readJsonStringField(output, 'text') ?? '';
  const textLength = readJsonNumberField(output, 'text_length');
  const durationMs = readJsonNumberField(output, 'duration_ms');
  const ok = readJsonBooleanField(output, 'ok');

  return {
    ok: ok !== false,
    error: readJsonStringField(output, 'error') ?? '',
    url: readJsonStringField(output, 'url') ?? '',
    requestedUrl: readJsonStringField(output, 'requested_url') ?? '',
    query: readJsonStringField(output, 'query') ?? '',
    text,
    textLength: textLength ?? text.length,
    textStartChar: readJsonNumberField(output, 'text_start_char'),
    textEndChar: readJsonNumberField(output, 'text_end_char'),
    truncated: readJsonBooleanField(output, 'truncated') ?? true,
    matches: [],
    durationMs: durationMs ?? undefined,
  };
}

function parseWebReadPdfOutput(output: string | undefined | null): ParsedWebReadPdfOutput | null {
  if (!output) return null;
  try {
    const parsed = JSON.parse(output) as Record<string, unknown>;
    if (parsed.tool !== WEB_READ_PDF_TOOL_ID && !(parsed.tool === WEB_BROWSE_TOOL_ID && parsed.mode === 'pdf_read')) return null;

    const rawMatches = Array.isArray(parsed.matches) ? parsed.matches : [];
    const matches: ParsedWebReadPdfMatch[] = [];
    for (const item of rawMatches) {
      if (!item || typeof item !== 'object') continue;
      const record = item as Record<string, unknown>;
      const text = typeof record.text === 'string' ? record.text : '';
      const matchStartRaw = record.match_start_char;
      const matchEndRaw = record.match_end_char;
      const matchStart = typeof matchStartRaw === 'number' && Number.isFinite(matchStartRaw) ? matchStartRaw : null;
      const matchEnd = typeof matchEndRaw === 'number' && Number.isFinite(matchEndRaw) ? matchEndRaw : null;
      if (!text && matchStart == null && matchEnd == null) continue;
      matches.push({ text, matchStart, matchEnd });
    }

    const textStartRaw = parsed.text_start_char;
    const textEndRaw = parsed.text_end_char;
    const textLengthRaw = parsed.text_length;
    const text = typeof parsed.text === 'string' ? parsed.text : '';

    return {
      ok: parsed.ok === true,
      error: typeof parsed.error === 'string' ? parsed.error : '',
      url: typeof parsed.url === 'string' ? parsed.url : '',
      requestedUrl: typeof parsed.requested_url === 'string' ? parsed.requested_url : '',
      query: typeof parsed.query === 'string' ? parsed.query : '',
      text,
      textLength: typeof textLengthRaw === 'number' && Number.isFinite(textLengthRaw) ? textLengthRaw : text.length,
      textStartChar: typeof textStartRaw === 'number' && Number.isFinite(textStartRaw) ? textStartRaw : null,
      textEndChar: typeof textEndRaw === 'number' && Number.isFinite(textEndRaw) ? textEndRaw : null,
      truncated: parsed.truncated === true,
      matches,
      durationMs: typeof parsed.duration_ms === 'number' ? parsed.duration_ms : undefined,
    };
  } catch {
    return parsePartialWebReadPdfOutput(output);
  }
}

export const ToolCallDisplay = memo(function ToolCallDisplay({
  toolCall,
  defaultExpanded = false,
  conversationId,
  workspaceId,
  allowRerun = true,
  siblingEvents,
  messageId,
  messageIndex,
  eventIndex,
  onRetrySuccess,
  onOpenWorkspaceFile,
}: ToolCallDisplayProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [copiedButtonKey, setCopiedButtonKey] = useState<string | null>(null);
  const copyFeedbackTimer = useRef<number | null>(null);
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryOutput, setRetryOutput] = useState<string | null>(null);
  const [retryError, setRetryError] = useState<string | null>(null);
  const [retryProgressText, setRetryProgressText] = useState<string | null>(null);
  const retryProgressTimers = useRef<number[]>([]);
  const visualToolName = useMemo(() => getToolVisualName(toolCall.tool), [toolCall.tool]);
  const [isRerunning, setIsRerunning] = useState(false);
  const [zoomedImage, setZoomedImage] = useState<string | null>(null);
  // Per-entry diff state keyed by snapshot+path+writeSignature so a batched
  // tool call can load each file's diff independently and a partial failure
  // does not blank the entire card.
  const [userspaceDiffMap, setUserspaceDiffMap] = useState<Map<string, UserSpaceSnapshotFileDiff>>(() => new Map());
  const [userspaceDiffLoadingMap, setUserspaceDiffLoadingMap] = useState<Map<string, boolean>>(() => new Map());
  const [userspaceDiffErrorMap, setUserspaceDiffErrorMap] = useState<Map<string, string>>(() => new Map());
  const [userspaceCurrentSnapshotId, setUserspaceCurrentSnapshotId] = useState<string | null>(null);
  const [showUserspaceDiffOverlay, setShowUserspaceDiffOverlay] = useState(false);
  const [userspaceOverlayActiveIndex, setUserspaceOverlayActiveIndex] = useState<number>(0);
  const [hydratedUserspaceListEntries, setHydratedUserspaceListEntries] = useState<ParsedUserSpaceListEntry[] | null>(null);
  const [isHydratingUserspaceList, setIsHydratingUserspaceList] = useState(false);
  const [hydratedUserspaceListError, setHydratedUserspaceListError] = useState<string | null>(null);
  const latestOutput = retryOutput || toolCall.output;
  const activeOutput = isRerunning ? retryOutput : latestOutput;
  const parsedTerminalOutput = useMemo(() => parseTerminalOutput(latestOutput), [latestOutput]);

  // Check if this is a visualization tool that can be retried
  const isVisualizationTool = toolCall.tool === 'create_chart' || toolCall.tool === 'create_datatable';
  const isTerminalCommand = isTerminalToolCall(toolCall) || Boolean(parsedTerminalOutput);
  const isSqlTool = isSqlToolCall(toolCall);
  const rerunKind = getTerminalRerunKind(toolCall);
  const canRerun = allowRerun && canRerunToolCall(toolCall);
  const hasRerunContext = rerunKind === USERSPACE_EXEC_RERUN_KIND
    ? Boolean(workspaceId)
    : rerunKind === CONVERSATION_TOOL_RERUN_KIND
      ? Boolean(conversationId && toolCall.connection?.tool_config_id)
      : rerunKind === CHAT_DIAGNOSTIC_RERUN_KIND
        ? Boolean(conversationId)
      : false;
  const activeTerminalOutput = activeOutput;

  // Parse terminal output for terminal-style rendering
  const terminalOutput = useMemo(() => {
    if (!isTerminalCommand) return null;
    return parseTerminalOutput(activeTerminalOutput);
  }, [activeTerminalOutput, isTerminalCommand]);

  // Check if this tool call failed based on output content
  const hasErrorInOutput = useMemo(() => {
    const output = activeOutput;
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
  }, [activeOutput]);

  // Effective output (use retry output if available)
  const effectiveOutput = activeOutput || '';

  const userspaceWriteBatch = useMemo(() => {
    if (hasErrorInOutput) {
      return parseUserspaceWriteBatch(toolCall.tool, toolCall.output);
    }
    return parseUserspaceWriteBatch(toolCall.tool, effectiveOutput);
  }, [effectiveOutput, hasErrorInOutput, toolCall.output, toolCall.tool]);

  // Legacy single-result alias for the (still single-file) read/render
  // paths. The full batched entries list is available on userspaceWriteBatch.
  const userspaceWriteResult = userspaceWriteBatch?.primary ?? null;

  // Only diffable ops (upsert/patch) drive the diff loader. Move/delete
  // entries are rendered as path rows without diff fetches.
  const diffableEntries = useMemo<ParsedUserSpaceWriteResult[]>(() => {
    if (!userspaceWriteBatch || !workspaceId) return [];
    return userspaceWriteBatch.entries.filter(
      (entry) =>
        USERSPACE_DIFFABLE_TOOL_NAMES.has(entry.toolName)
        && entry.persisted
        && !entry.rejected
        && !entry.noChanges,
    );
  }, [userspaceWriteBatch, workspaceId]);

  useEffect(() => {
    if (!expanded || !workspaceId || diffableEntries.length === 0) {
      return;
    }

    let cancelled = false;

    const loadOne = async (
      snapshotId: string,
      entry: ParsedUserSpaceWriteResult,
      cacheKey: string,
    ): Promise<UserSpaceSnapshotFileDiff> => {
      const cached = userspaceWriteDiffCache.get(cacheKey);
      if (cached) return cached;

      let finalDiff: UserSpaceSnapshotFileDiff;
      try {
        const baselineDiff = await api.getUserSpaceSnapshotFileDiff(workspaceId, snapshotId, entry.path);
        finalDiff = entry.file?.content != null
          ? mergeUserspaceWriteDiff(baselineDiff, entry)
          : baselineDiff;
      } catch {
        const currentFile = await api.getUserSpaceFile(workspaceId, entry.path);
        const afterContent = entry.file?.content ?? currentFile.content ?? '';
        const lines = afterContent.split('\n').length;
        finalDiff = {
          workspace_id: workspaceId,
          snapshot_id: snapshotId,
          path: entry.path,
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
      return finalDiff;
    };

    const loadAll = async () => {
      try {
        const timeline = await api.getUserSpaceSnapshotTimeline(workspaceId);
        const snapshotId = timeline.current_snapshot_id ?? null;
        if (!snapshotId) {
          throw new Error('No snapshot baseline is available for this workspace yet.');
        }

        if (!cancelled) {
          setUserspaceCurrentSnapshotId(snapshotId);
        }

        // Mark all entries loading up-front so the UI shows progress for
        // each card even before its individual fetch completes.
        if (!cancelled) {
          setUserspaceDiffLoadingMap(() => {
            const next = new Map<string, boolean>();
            for (const entry of diffableEntries) {
              next.set(buildUserspaceToolDiffCacheKey(snapshotId, entry), true);
            }
            return next;
          });
          setUserspaceDiffErrorMap(() => new Map());
        }

        // Capped concurrency: process entries in small parallel batches
        // so a 20-file batched call does not flood the snapshot API.
        const concurrency = 4;
        const queue = [...diffableEntries];
        const runners: Promise<void>[] = [];
        const runOne = async () => {
          while (queue.length > 0) {
            const entry = queue.shift();
            if (!entry) break;
            const cacheKey = buildUserspaceToolDiffCacheKey(snapshotId, entry);
            try {
              const diff = await loadOne(snapshotId, entry, cacheKey);
              if (cancelled) return;
              setUserspaceDiffMap((prev) => {
                const next = new Map(prev);
                next.set(cacheKey, diff);
                return next;
              });
            } catch (err) {
              if (cancelled) return;
              const message = err instanceof Error ? err.message : 'Failed to load file diff';
              setUserspaceDiffErrorMap((prev) => {
                const next = new Map(prev);
                next.set(cacheKey, message);
                return next;
              });
            } finally {
              if (!cancelled) {
                setUserspaceDiffLoadingMap((prev) => {
                  const next = new Map(prev);
                  next.delete(cacheKey);
                  return next;
                });
              }
            }
          }
        };
        for (let i = 0; i < Math.min(concurrency, diffableEntries.length); i += 1) {
          runners.push(runOne());
        }
        await Promise.all(runners);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : 'Failed to load file diffs';
        // Mark every entry errored so cards show the reason.
        setUserspaceDiffLoadingMap(() => new Map());
        setUserspaceDiffErrorMap(() => {
          const next = new Map<string, string>();
          for (const entry of diffableEntries) {
            // Without a snapshot id we can't form the canonical key, so
            // use the path as a fallback identifier.
            next.set(`__error__:${entry.path}`, message);
          }
          return next;
        });
      }
    };

    void loadAll();

    return () => {
      cancelled = true;
    };
  }, [expanded, workspaceId, diffableEntries]);

  const userspaceReadBatch = useMemo(() => {
    if (toolCall.tool !== 'read_userspace_file') return null;
    return parseUserspaceReadBatch(toolCall.tool, effectiveOutput);
  }, [effectiveOutput, toolCall.tool]);

  const userspaceListResult = useMemo(() => {
    if (toolCall.tool !== 'list_userspace_files') return null;
    return parseUserspaceListToolResult(toolCall.tool, effectiveOutput);
  }, [effectiveOutput, toolCall.tool]);

  useEffect(() => {
    if (toolCall.tool !== 'list_userspace_files' || !workspaceId || !expanded) {
      setHydratedUserspaceListEntries(null);
      setHydratedUserspaceListError(null);
      setIsHydratingUserspaceList(false);
      return;
    }
    if (!userspaceListResult?.truncated) {
      setHydratedUserspaceListEntries(null);
      setHydratedUserspaceListError(null);
      setIsHydratingUserspaceList(false);
      return;
    }

    let cancelled = false;
    setIsHydratingUserspaceList(true);
    setHydratedUserspaceListError(null);

    const hydrate = async () => {
      try {
        const files = await api.listUserSpaceFiles(workspaceId, { includeDirs: false });
        if (cancelled) return;
        const mapped = files
          .map((file) => ({
            path: file.path,
            entryType: file.entry_type === 'directory' ? 'directory' as const : 'file' as const,
            sizeBytes: Number.isFinite(file.size_bytes) ? Math.max(0, file.size_bytes) : 0,
            updatedAt: file.updated_at || null,
          }))
          .filter((entry) => entry.entryType === 'file' && entry.path.trim().length > 0)
          .sort((a, b) => a.path.localeCompare(b.path));
        setHydratedUserspaceListEntries(mapped);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : 'Failed to load full workspace file list.';
        setHydratedUserspaceListError(message);
        setHydratedUserspaceListEntries(null);
      } finally {
        if (!cancelled) {
          setIsHydratingUserspaceList(false);
        }
      }
    };

    void hydrate();

    return () => {
      cancelled = true;
    };
  }, [expanded, toolCall.tool, userspaceListResult?.truncated, workspaceId]);

  const userspaceReadResult = userspaceReadBatch?.primary ?? null;

  const userspaceReadDiff = useMemo((): UserSpaceSnapshotFileDiff | null => {
    if (!userspaceReadResult || userspaceReadResult.isRejected || userspaceReadResult.content == null) return null;
    const normalizedContent = normalizeUserspaceSnippetContent(userspaceReadResult.content);
    const lines = normalizedContent.split('\n').length;
    return {
      workspace_id: '',
      snapshot_id: '',
      path: userspaceReadResult.path,
      status: 'R',
      before_content: '',
      after_content: normalizedContent,
      additions: lines,
      deletions: 0,
      is_binary: false,
      is_deleted_in_current: false,
      is_untracked_in_current: false,
    };
  }, [userspaceReadResult]);

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

  // Parse web_search / web_browse structured payloads for pretty rendering
  const webSearchOutput = useMemo(() => {
    if (toolCall.tool !== 'web_search' || !effectiveOutput) return null;
    return parseWebSearchOutput(effectiveOutput);
  }, [toolCall.tool, effectiveOutput]);

  const webBrowseOutput = useMemo(() => {
    if (toolCall.tool !== 'web_browse' || !effectiveOutput) return null;
    return parseWebBrowseOutput(effectiveOutput);
  }, [toolCall.tool, effectiveOutput]);

  const webReadPdfOutput = useMemo(() => {
    if (toolCall.tool !== WEB_READ_PDF_TOOL_ID || !effectiveOutput) return null;
    return parseWebReadPdfOutput(effectiveOutput);
  }, [toolCall.tool, effectiveOutput]);

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
    if (!effectiveOutput) return { tableData: null, displayText: '' };
    return parseTableMetadata(effectiveOutput);
  }, [effectiveOutput]);
  const visibleDisplayText = toolCall.tool === WEB_READ_PDF_TOOL_ID
    ? maskHiddenToolNames(displayText)
    : displayText;

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

  const copyToClipboard = useCallback(async (text: string, type: 'query' | 'result', buttonId = 'default') => {
    try {
      await navigator.clipboard.writeText(text);
      const key = `${type}:${buttonId}`;
      setCopiedButtonKey(key);
      if (copyFeedbackTimer.current != null) {
        window.clearTimeout(copyFeedbackTimer.current);
      }
      copyFeedbackTimer.current = window.setTimeout(() => {
        setCopiedButtonKey((current) => (current === key ? null : current));
      }, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, []);

  const isCopiedButton = useCallback(
    (type: 'query' | 'result', buttonId: string) => copiedButtonKey === `${type}:${buttonId}`,
    [copiedButtonKey],
  );
  useEffect(() => () => {
    if (copyFeedbackTimer.current != null) {
      window.clearTimeout(copyFeedbackTimer.current);
    }
  }, []);

  const clearRetryProgressTimers = useCallback(() => {
    retryProgressTimers.current.forEach((timerId) => window.clearTimeout(timerId));
    retryProgressTimers.current = [];
  }, []);

  useEffect(() => () => clearRetryProgressTimers(), [clearRetryProgressTimers]);

  // Handle retry for visualization tools
  const handleRetry = useCallback(async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (!conversationId) {
      setRetryError('Cannot retry: missing conversation context');
      return;
    }

    // Find deterministic source data from nearby tool calls when available.
    let sourceData: { columns: string[]; rows: unknown[][] } | null = null;

    if (siblingEvents) {
      for (const event of [...siblingEvents].reverse()) {
        if (event.type === 'tool' && event.output) {
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

    setIsRetrying(true);
    setRetryError(null);
    setRetryProgressText('Checking source data...');
    clearRetryProgressTimers();
    retryProgressTimers.current = [
      window.setTimeout(() => setRetryProgressText('Re-running source query if needed...'), 1000),
      window.setTimeout(() => setRetryProgressText('Repairing visualization data...'), 2500),
      window.setTimeout(() => setRetryProgressText('Validating repaired output...'), 5000),
    ];

    try {
      const toolType = toolCall.tool === 'create_chart' ? 'chart' : 'datatable';
      const request: RetryVisualizationRequest = {
        tool_type: toolType,
        ...(sourceData ? { source_data: sourceData } : {}),
        title: toolType === 'chart' ? 'Chart' : 'Data',
        allow_ai_repair: true,
        allow_source_rerun: true,
        failed_tool_input: toolCall.input,
        failed_tool_output: toolCall.output,
        context_events: buildVisualizationRetryContextEvents(siblingEvents),
        ...(messageId ? { message_id: messageId } : {}),
        ...(typeof messageIndex === 'number' ? { message_index: messageIndex } : {}),
        ...(typeof eventIndex === 'number' ? { event_index: eventIndex } : {}),
      };

      const result = await api.retryVisualization(conversationId, request, workspaceId);

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
      clearRetryProgressTimers();
      setRetryProgressText(null);
      setIsRetrying(false);
    }
  }, [clearRetryProgressTimers, conversationId, eventIndex, messageId, messageIndex, siblingEvents, toolCall.input, toolCall.output, toolCall.tool, onRetrySuccess, workspaceId]);

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

      if (rerunKind === CONVERSATION_TOOL_RERUN_KIND || rerunKind === CHAT_DIAGNOSTIC_RERUN_KIND) {
        if (!conversationId) return;
        const toolConfigId = toolCall.connection?.tool_config_id;
        if (rerunKind === CONVERSATION_TOOL_RERUN_KIND && !toolConfigId) return;
        const result = await api.retryTerminalToolCall(
          conversationId,
          rerunKind === CONVERSATION_TOOL_RERUN_KIND ? {
            tool_config_id: toolConfigId,
            input: toolCall.input || {},
          } : {
            builtin_tool_id: CHAT_DIAGNOSTIC_COMMAND_TOOL_ID,
            input: toolCall.input || {},
          },
          workspaceId,
        );
        if (!result.success || !result.output) {
          throw new Error(result.error || 'Re-run failed');
        }
        setRetryOutput(result.output);
        return;
      }
    } catch (err) {
      setRetryError(err instanceof Error ? err.message : 'Re-run failed');
    } finally {
      setIsRerunning(false);
    }
  }, [canRerun, rerunKind, isRerunning, inputDisplay, workspaceId, conversationId, toolCall.connection, toolCall.input]);

  // Determine the tool-type icon (always visible)
  const getToolIcon = () => {
    if (toolCall.tool === WEB_READ_PDF_TOOL_ID) return <FileText size={14} />;
    const name = visualToolName.toLowerCase();
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

  // Helper to look up an entry's diff/loading/error from the per-entry maps.
  const getEntryDiffState = (entry: ParsedUserSpaceWriteResult) => {
    if (!userspaceCurrentSnapshotId) {
      return {
        cacheKey: '',
        diff: null as UserSpaceSnapshotFileDiff | null,
        loading: false,
        error: userspaceDiffErrorMap.get(`__error__:${entry.path}`) ?? null,
      };
    }
    const cacheKey = buildUserspaceToolDiffCacheKey(userspaceCurrentSnapshotId, entry);
    return {
      cacheKey,
      diff: userspaceDiffMap.get(cacheKey) ?? null,
      loading: Boolean(userspaceDiffLoadingMap.get(cacheKey)),
      error: userspaceDiffErrorMap.get(cacheKey) ?? userspaceDiffErrorMap.get(`__error__:${entry.path}`) ?? null,
    };
  };

  // Build the list of overlay entries (in entry order) for the batched
  // FileDiffOverlay. Non-diffable ops still appear in the nav so the
  // overlay reflects the full batch.
  const userspaceOverlayEntries = useMemo<FileDiffOverlayEntry[]>(() => {
    if (!userspaceWriteBatch) return [];
    return userspaceWriteBatch.entries.map((entry, index) => {
      const state = getEntryDiffState(entry);
      return {
        key: `${entry.op}:${entry.path}:${entry.writeSignature ?? index}`,
        path: entry.path,
        op: entry.op,
        diff: state.diff,
        loading: state.loading,
        error: state.error,
      };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userspaceWriteBatch, userspaceDiffMap, userspaceDiffLoadingMap, userspaceDiffErrorMap, userspaceCurrentSnapshotId]);

  // Special rendering for chart tool - show chart inline without collapsible.
  // Keep these returns after every hook in this component so tool-call output
  // transitions cannot change the hook count between renders.
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

  const openUserspaceOverlayAt = (index: number) => {
    setUserspaceOverlayActiveIndex(index);
    setShowUserspaceDiffOverlay(true);
  };

  const renderUserspaceWriteEntry = (
    entry: ParsedUserSpaceWriteResult,
    index: number,
    options: { showSummaryFallback: boolean; batched?: boolean },
  ): ReactNode => {
    const { diff, loading, error } = getEntryDiffState(entry);
    const isDiffable = USERSPACE_DIFFABLE_TOOL_NAMES.has(entry.toolName) || entry.op === 'upsert' || entry.op === 'patch';
    const showDiffCard = isDiffable && !loading && Boolean(diff);
    const fallbackText = entry.error
      ? entry.error
      : entry.message
        ? entry.message
        : (options.showSummaryFallback ? (userspaceWriteSummary || displayText || (toolCall.output ?? '')) : '');
    const statusGlyph = entry.op === 'delete'
      ? 'D'
      : entry.op === 'move'
        ? 'R'
        : (diff?.status ?? (entry.rejected ? 'X' : 'M'));
    const opLabel = entry.op === 'upsert'
      ? 'Upsert'
      : entry.op === 'patch'
        ? 'Patch'
        : entry.op === 'move'
          ? 'Move'
          : 'Delete';
    const batched = Boolean(options.batched);
    const sectionClass = batched
      ? 'tool-call-section tool-call-userspace-batched-entry'
      : 'tool-call-section';
    const metaText = batched
      ? (entry.rejected ? (entry.error || entry.message || 'Rejected') : entry.noChanges ? 'No changes' : '')
      : ((diff ? formatDiffStatus(diff.status) : null) || entry.message || (entry.rejected ? 'Rejected' : entry.noChanges ? 'No changes' : 'Updated'));
    const diffCounts = diff ? `+${diff.additions} -${diff.deletions}` : '';

    return (
      <div className={sectionClass} key={`${entry.op}:${entry.path}:${index}`}>
        {!batched && (
          <div className="tool-call-section-header">
            <span className="tool-call-section-label">{`${opLabel}:`}</span>
            {showDiffCard ? (
              <button
                type="button"
                className="tool-call-retry-btn"
                onClick={() => openUserspaceOverlayAt(index)}
                title="Open full diff"
              >
                <Diff size={12} />
                <span>Full Diff</span>
              </button>
            ) : (
              <button
                className="tool-call-copy-btn"
                onClick={() => copyToClipboard(fallbackText || entry.path, 'result', `userspace-write-${index}`)}
                title="Copy result"
              >
                {isCopiedButton('result', `userspace-write-${index}`) ? <Check size={12} /> : <Copy size={12} />}
              </button>
            )}
          </div>
        )}
        <div className="tool-call-userspace-write-summary">
          <div className="tool-call-userspace-write-summary-row">
            {batched && (
              <span className="tool-call-userspace-batched-op" title={opLabel}>{opLabel}</span>
            )}
            {onOpenWorkspaceFile ? (
              <button
                className="tool-call-userspace-write-path tool-call-userspace-write-path-link"
                title={entry.oldPath ? `${entry.oldPath} → ${entry.path}` : entry.path}
                onClick={() => onOpenWorkspaceFile(entry.path)}
              >
                {entry.oldPath ? `${entry.oldPath} → ${entry.path}` : entry.path}
              </button>
            ) : (
              <span
                className="tool-call-userspace-write-path"
                title={entry.oldPath ? `${entry.oldPath} → ${entry.path}` : entry.path}
              >
                {entry.oldPath ? `${entry.oldPath} → ${entry.path}` : entry.path}
              </span>
            )}
            {batched && diffCounts && (
              <span className="tool-call-userspace-batched-counts">{diffCounts}</span>
            )}
            <span className={`userspace-snapshot-diff-status userspace-snapshot-diff-status-${String(statusGlyph).toLowerCase()}`}>
              {statusGlyph}
            </span>
            {batched && showDiffCard && (
              <button
                type="button"
                className="tool-call-userspace-batched-action"
                onClick={() => openUserspaceOverlayAt(index)}
                title="Open full diff"
              >
                <Diff size={12} />
              </button>
            )}
          </div>
          {metaText && (
            <div className="tool-call-userspace-write-meta">
              {metaText}
              {!batched && diffCounts ? ` | ${diffCounts}` : ''}
            </div>
          )}
        </div>
        {loading && (
          <div className="tool-call-userspace-diff-loading">
            <MiniLoadingSpinner variant="icon" size={14} />
            <span>Loading diff from current snapshot...</span>
          </div>
        )}
        {!loading && !diff && fallbackText && !batched && (
          <pre className="tool-call-code">{fallbackText}</pre>
        )}
        {error && (
          <div className="tool-call-userspace-diff-error">{error}</div>
        )}
        {showDiffCard && diff && (
          <div className="tool-call-userspace-diff-card">
            <UserSpaceFileDiffView
              diff={diff}
              beforeLabel="Last Snapshot"
              afterLabel="Tool Result"
              compact
              maxLines={batched ? 14 : undefined}
            />
          </div>
        )}
      </div>
    );
  };

  const renderUserspaceResultBody = (): ReactNode => {
    switch (toolCall.tool) {
      case 'upsert_userspace_file':
      case 'patch_userspace_file':
      case 'move_userspace_file':
      case 'delete_userspace_file': {
        if (!userspaceWriteBatch || !userspaceWriteResult) return null;
        const { entries, isBatch, summary, aggregateMessage } = userspaceWriteBatch;

        if (!isBatch) {
          return renderUserspaceWriteEntry(entries[0], 0, { showSummaryFallback: true });
        }

        return (
          <>
            <div className="tool-call-section">
              <div className="tool-call-section-header">
                <span className="tool-call-section-label">Batched result:</span>
                <button
                  className="tool-call-copy-btn"
                  onClick={() => copyToClipboard(aggregateMessage || displayText || (toolCall.output ?? ''), 'result', 'userspace-write-batch-summary')}
                  title="Copy summary"
                >
                  {isCopiedButton('result', 'userspace-write-batch-summary') ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
              <div className="tool-call-userspace-write-meta">
                {aggregateMessage || `Batched ${userspaceWriteBatch.toolName} across ${summary.total} file(s).`}
              </div>
            </div>
            <div className="tool-call-userspace-batched-list">
              {entries.map((entry, index) => renderUserspaceWriteEntry(entry, index, { showSummaryFallback: false, batched: true }))}
            </div>
          </>
        );
      }

      case 'read_userspace_file': {
        if (!userspaceReadBatch) return null;
        const { entries, isBatch, summary, aggregateMessage } = userspaceReadBatch;

        const renderReadEntry = (entry: ParsedUserSpaceReadResult, idx: number, opts: { batched?: boolean } = {}): ReactNode => {
          const batched = Boolean(opts.batched);
          const normalizedContent = entry.content != null
            ? normalizeUserspaceSnippetContent(entry.content)
            : null;
          const lineCount = normalizedContent ? normalizedContent.split('\n').length : 0;
          const entryDiff: UserSpaceSnapshotFileDiff | null = (!entry.isRejected && normalizedContent != null)
            ? {
                workspace_id: '',
                snapshot_id: '',
                path: entry.path,
                status: 'R',
                before_content: '',
                after_content: normalizedContent,
                additions: lineCount,
                deletions: 0,
                is_binary: false,
                is_deleted_in_current: false,
                is_untracked_in_current: false,
                starting_before_line: undefined,
                starting_after_line: entry.startLine ?? undefined,
              }
            : null;
          const range = (entry.startLine != null && entry.endLine != null)
            ? (entry.totalLines != null
                ? `Lines ${entry.startLine}-${entry.endLine} of ${entry.totalLines}`
                : `Lines ${entry.startLine}-${entry.endLine}`)
            : null;
          const statusGlyph = entry.isRejected ? 'X' : 'R';
          const sectionClass = batched
            ? 'tool-call-section tool-call-userspace-batched-entry'
            : 'tool-call-section';
          const metaText = batched
            ? (entry.isRejected ? (entry.message || 'Rejected') : '')
            : (entry.message || (entry.isRejected ? 'Rejected' : 'Read'));
          return (
            <div className={sectionClass} key={`read:${entry.path}:${idx}`}>
              {!batched && (
                <div className="tool-call-section-header">
                  <span className="tool-call-section-label">Read:</span>
                  <button
                    className="tool-call-copy-btn"
                    onClick={() => copyToClipboard(entry.content ?? entry.message ?? entry.path, 'result', `userspace-read-${idx}`)}
                    title="Copy result"
                  >
                    {isCopiedButton('result', `userspace-read-${idx}`) ? <Check size={12} /> : <Copy size={12} />}
                  </button>
                </div>
              )}
              <div className="tool-call-userspace-write-summary">
                <div className="tool-call-userspace-write-summary-row">
                  {batched && (
                    <span className="tool-call-userspace-batched-op">Read</span>
                  )}
                  {onOpenWorkspaceFile ? (
                    <button
                      className="tool-call-userspace-write-path tool-call-userspace-write-path-link"
                      title={entry.path}
                      onClick={() => onOpenWorkspaceFile(entry.path)}
                    >
                      {entry.path}
                    </button>
                  ) : (
                    <span className="tool-call-userspace-write-path" title={entry.path}>{entry.path}</span>
                  )}
                  {batched && range && (
                    <span className="tool-call-userspace-batched-counts">{range}</span>
                  )}
                  <span className={`userspace-snapshot-diff-status userspace-snapshot-diff-status-${statusGlyph.toLowerCase()}`}>
                    {statusGlyph}
                  </span>
                  {batched && (
                    <button
                      type="button"
                      className="tool-call-userspace-batched-action"
                      onClick={() => copyToClipboard(entry.content ?? entry.message ?? entry.path, 'result', `userspace-read-batch-${idx}`)}
                      title="Copy snippet"
                    >
                      {isCopiedButton('result', `userspace-read-batch-${idx}`) ? <Check size={12} /> : <Copy size={12} />}
                    </button>
                  )}
                </div>
                {metaText && (
                  <div className="tool-call-userspace-write-meta">
                    {metaText}
                    {!batched && range ? ` | ${range}` : ''}
                  </div>
                )}
              </div>
              {entryDiff && (
                <div className="tool-call-userspace-diff-card">
                  <UserSpaceFileDiffView
                    diff={entryDiff}
                    beforeLabel="Read Snippet"
                    afterLabel="Read Snippet"
                    compact
                    highlightSingleColumnChanges={false}
                  />
                </div>
              )}
            </div>
          );
        };

        if (!isBatch) {
          if (!userspaceReadDiff || !userspaceReadResult) {
            return renderReadEntry(entries[0], 0);
          }
          return renderReadEntry(entries[0], 0);
        }

        return (
          <>
            <div className="tool-call-section">
              <div className="tool-call-section-header">
                <span className="tool-call-section-label">Batched read:</span>
                <button
                  className="tool-call-copy-btn"
                  onClick={() => copyToClipboard(aggregateMessage || displayText || (toolCall.output ?? ''), 'result', 'userspace-read-batch-summary')}
                  title="Copy summary"
                >
                  {isCopiedButton('result', 'userspace-read-batch-summary') ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
              <div className="tool-call-userspace-write-meta">
                {aggregateMessage || `Batched read across ${summary.total} file(s).`}
              </div>
            </div>
            <div className="tool-call-userspace-batched-list">
              {entries.map((entry, idx) => renderReadEntry(entry, idx, { batched: true }))}
            </div>
          </>
        );
      }

      case 'list_userspace_files': {
        if (!userspaceListResult) return null;
        const { entries, count, fileCount, totalSizeBytes, message, truncated } = userspaceListResult;
        const parsedFileEntries = entries.filter((entry) => entry.entryType === 'file');
        const visibleEntries = hydratedUserspaceListEntries ?? parsedFileEntries;
        const isHydrated = hydratedUserspaceListEntries != null;

        const formatSize = (bytes: number): string => {
          if (bytes <= 0) return '0 B';
          if (bytes < 1024) return `${bytes} B`;
          if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
          if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
          return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
        };

        const formatUpdatedAt = (iso: string | null): string => {
          if (!iso) return '';
          const date = new Date(iso);
          if (Number.isNaN(date.getTime())) return '';
          return date.toLocaleString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          });
        };

        const summaryParts: string[] = [];
        if (fileCount > 0) summaryParts.push(`${fileCount} file${fileCount === 1 ? '' : 's'}`);
        if (fileCount > 0) summaryParts.push(formatSize(totalSizeBytes));
        const summaryText = summaryParts.length > 0
          ? summaryParts.join(' \u00b7 ')
          : (message || 'Empty workspace');

        const copyAllPaths = visibleEntries.map((e) => e.path).join('\n');

        return (
          <div className="tool-call-section">
            <div className="tool-call-section-header">
              <span className="tool-call-section-label">Workspace files:</span>
              <button
                className="tool-call-copy-btn"
                onClick={() => copyToClipboard(copyAllPaths || (toolCall.output ?? ''), 'result', 'userspace-list-paths')}
                title="Copy all paths"
              >
                {isCopiedButton('result', 'userspace-list-paths') ? <Check size={12} /> : <Copy size={12} />}
              </button>
            </div>
            <div className="tool-call-userspace-list-summary">{summaryText}</div>
            {isHydratingUserspaceList && (
              <div className="tool-call-userspace-list-truncated">Loading full file list...</div>
            )}
            {hydratedUserspaceListError && (
              <div className="tool-call-userspace-list-truncated">{hydratedUserspaceListError}</div>
            )}
            {visibleEntries.length === 0 ? (
              <div className="tool-call-userspace-list-empty">
                {truncated
                  ? 'Tool output was truncated before any file entries could be parsed.'
                  : 'No files in this workspace.'}
              </div>
            ) : (
              <ul className="tool-call-userspace-list">
                {visibleEntries.map((entry) => {
                  const updated = formatUpdatedAt(entry.updatedAt);
                  return (
                    <li
                      key={`file:${entry.path}`}
                      className="tool-call-userspace-list-item tool-call-userspace-list-item-file"
                    >
                      <span className="tool-call-userspace-list-icon" aria-hidden="true">
                        <FileText size={12} />
                      </span>
                      {onOpenWorkspaceFile ? (
                        <button
                          type="button"
                          className="tool-call-userspace-write-path tool-call-userspace-write-path-link"
                          title={entry.path}
                          onClick={() => onOpenWorkspaceFile(entry.path)}
                        >
                          {entry.path}
                        </button>
                      ) : (
                        <span className="tool-call-userspace-write-path" title={entry.path}>
                          {entry.path}
                        </span>
                      )}
                      <span className="tool-call-userspace-list-size" title={`${entry.sizeBytes} bytes`}>
                        {formatSize(entry.sizeBytes)}
                      </span>
                      {updated && (
                        <span className="tool-call-userspace-list-updated" title={entry.updatedAt ?? ''}>
                          {updated}
                        </span>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
            {(truncated || count > entries.length) && !isHydrated && !isHydratingUserspaceList && (
              <div className="tool-call-userspace-list-truncated">
                Tool output was truncated. Some entries are omitted.
              </div>
            )}
          </div>
        );
      }

      case 'web_search': {
        if (!webSearchOutput) {
          return (
            <div className="tool-call-section">
              <div className="tool-call-section-header">
                <span className="tool-call-section-label">Result:</span>
                <button
                  className="tool-call-copy-btn"
                  onClick={() => copyToClipboard(displayText || effectiveOutput, 'result', 'web-search-fallback')}
                  title="Copy result"
                >
                  {isCopiedButton('result', 'web-search-fallback') ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
              <pre className="tool-call-code">{displayText}</pre>
            </div>
          );
        }

        const search = webSearchOutput;
        const isErrorState = !search.ok || search.blocked || Boolean(search.error);
        const metaParts: string[] = [];
        if (search.provider) metaParts.push(search.provider);
        metaParts.push(`${search.resultCount} result${search.resultCount === 1 ? '' : 's'}`);
        if (typeof search.durationMs === 'number') {
          metaParts.push(`${search.durationMs} ms`);
        }
        const queryText = search.query || (toolCall.input?.query as string) || '';

        return (
          <div className="tool-call-section tool-call-web-section">
            <div className="tool-call-section-header">
              <span className="tool-call-section-label">
                <Search size={12} />
                <span>Web search</span>
              </span>
              <div className="tool-call-terminal-header-actions">
              </div>
            </div>
            {queryText && (
              <div className="tool-call-web-query" title={queryText}>
                <Search size={12} aria-hidden="true" />
                <span>{queryText}</span>
                <button
                  className="tool-call-copy-btn"
                  onClick={() => copyToClipboard(queryText, 'query', 'web-search-query')}
                  title="Copy query"
                >
                  {isCopiedButton('query', 'web-search-query') ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
            )}
            <div className="tool-call-web-meta">{metaParts.join(' \u00b7 ')}</div>
            {isErrorState && (
              <div className="tool-call-web-error">
                <AlertCircle size={12} />
                <span>{search.error || (search.blocked ? 'Search was blocked.' : 'Search failed.')}</span>
              </div>
            )}
            {search.answer && (
              <div className="tool-call-web-answer">
                <div className="tool-call-web-answer-label">Answer</div>
                <div className="tool-call-web-answer-body">{search.answer}</div>
              </div>
            )}
            {search.results.length > 0 ? (
              <ol className="tool-call-web-results">
                {search.results.map((result, idx) => (
                  <li className="tool-call-web-result" key={`${result.url}-${idx}`}>
                    <div className="tool-call-web-result-head">
                      {result.favicon ? (
                        <img
                          src={result.favicon}
                          alt=""
                          className="tool-call-web-result-favicon"
                          loading="lazy"
                          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                        />
                      ) : (
                        <Globe size={12} className="tool-call-web-result-favicon-fallback" aria-hidden="true" />
                      )}
                      <a
                        className="tool-call-web-result-title"
                        href={result.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        title={result.url}
                      >
                        {result.title}
                      </a>
                      <button
                        className="tool-call-copy-btn"
                        onClick={() => copyToClipboard(result.url, 'result', `web-search-result-url-${idx}`)}
                        title="Copy result URL"
                      >
                        {isCopiedButton('result', `web-search-result-url-${idx}`) ? <Check size={12} /> : <Copy size={12} />}
                      </button>
                    </div>
                    <a
                      className="tool-call-web-result-url"
                      href={result.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={result.url}
                    >
                      {result.url}
                    </a>
                    {result.snippet && (
                      <p className="tool-call-web-result-snippet">{result.snippet}</p>
                    )}
                    {result.engine && (
                      <div className="tool-call-web-result-engine">{result.engine}</div>
                    )}
                  </li>
                ))}
              </ol>
            ) : !isErrorState ? (
              <div className="tool-call-web-empty">No results returned.</div>
            ) : null}
          </div>
        );
      }

      case 'web_browse': {
        if (!webBrowseOutput) {
          return (
            <div className="tool-call-section">
              <div className="tool-call-section-header">
                <span className="tool-call-section-label">Result:</span>
                <button
                  className="tool-call-copy-btn"
                  onClick={() => copyToClipboard(displayText || effectiveOutput, 'result', 'web-browse-fallback')}
                  title="Copy result"
                >
                  {isCopiedButton('result', 'web-browse-fallback') ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
              <pre className="tool-call-code">{displayText}</pre>
            </div>
          );
        }

        const browse = webBrowseOutput;
        const isErrorState = !browse.ok || Boolean(browse.error);
        const finalUrl = browse.url || browse.requestedUrl;
        const requestedUrl = browse.requestedUrl || (toolCall.input?.url as string) || '';
        const showResolvedUrl = Boolean(finalUrl && finalUrl !== requestedUrl);
        const metaParts: string[] = [];
        if (browse.statusCode != null) metaParts.push(`HTTP ${browse.statusCode}`);
        if (browse.textLength > 0) metaParts.push(`${browse.textLength.toLocaleString()} chars`);
        if (browse.truncated) metaParts.push('truncated');
        if (typeof browse.durationMs === 'number') metaParts.push(`${browse.durationMs} ms`);

        return (
          <div className="tool-call-section tool-call-web-section">
            <div className="tool-call-section-header">
              <span className="tool-call-section-label">
                <Globe size={12} />
                <span>Web page</span>
              </span>
              <div className="tool-call-terminal-header-actions">
                {finalUrl && (
                  <a
                    className="tool-call-copy-btn tool-call-web-open-btn"
                    href={finalUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    title="Open in new tab"
                  >
                    <Link size={12} />
                  </a>
                )}
                {browse.text && (
                  <button
                    className="tool-call-copy-btn"
                    onClick={() => copyToClipboard(browse.text, 'result', 'web-browse-page-text')}
                    title="Copy page text"
                  >
                    {isCopiedButton('result', 'web-browse-page-text') ? <Check size={12} /> : <Copy size={12} />}
                  </button>
                )}
              </div>
            </div>
            {requestedUrl && (
              <div className="tool-call-web-query" title={requestedUrl}>
                <a
                  className="tool-call-web-query-link"
                  href={requestedUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={requestedUrl}
                >
                  <Globe size={12} aria-hidden="true" />
                  <span>{requestedUrl}</span>
                </a>
                <button
                  className="tool-call-copy-btn"
                  onClick={() => copyToClipboard(requestedUrl, 'query', 'web-browse-requested-url')}
                  title="Copy URL"
                >
                  {isCopiedButton('query', 'web-browse-requested-url') ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
            )}
            <div className="tool-call-web-page-head">
              {browse.title && (
                <div className="tool-call-web-page-title" title={browse.title}>
                  {browse.title}
                </div>
              )}
              {showResolvedUrl && finalUrl && (
                <a
                  className="tool-call-web-page-url"
                  href={finalUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={finalUrl}
                >
                  {finalUrl}
                </a>
              )}
              {metaParts.length > 0 && (
                <div className="tool-call-web-meta">{metaParts.join(' \u00b7 ')}</div>
              )}
            </div>
            {isErrorState && (
              <div className="tool-call-web-error">
                <AlertCircle size={12} />
                <span>{browse.error || 'Browse failed.'}</span>
              </div>
            )}
            {browse.text && (
              <div className="tool-call-web-page-text-wrap">
                <pre className="tool-call-web-page-text">{browse.text}</pre>
              </div>
            )}
            {browse.links.length > 0 && (
              <details className="tool-call-web-links">
                <summary>{browse.links.length} link{browse.links.length === 1 ? '' : 's'}</summary>
                <ul className="tool-call-web-links-list">
                  {browse.links.map((link, idx) => (
                    <li key={`${link.url}-${idx}`} className="tool-call-web-link-item tool-call-web-link-row">
                      <a
                        href={link.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        title={link.url}
                      >
                        {link.text || link.url}
                      </a>
                      <button
                        className="tool-call-copy-btn"
                        onClick={() => copyToClipboard(link.url, 'result', `web-browse-link-${idx}`)}
                        title="Copy link URL"
                      >
                        {isCopiedButton('result', `web-browse-link-${idx}`) ? <Check size={12} /> : <Copy size={12} />}
                      </button>
                    </li>
                  ))}
                </ul>
              </details>
            )}
            {browse.consoleErrors.length > 0 && (
              <details className="tool-call-web-console-errors">
                <summary>{browse.consoleErrors.length} console error{browse.consoleErrors.length === 1 ? '' : 's'}</summary>
                <ul>
                  {browse.consoleErrors.map((err, idx) => (
                    <li key={idx}>{err}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        );
      }

      case WEB_READ_PDF_TOOL_ID: {
        if (!webReadPdfOutput) {
          return (
            <div className="tool-call-section">
              <div className="tool-call-section-header">
                <span className="tool-call-section-label">Result:</span>
                <button
                  className="tool-call-copy-btn"
                  onClick={() => copyToClipboard(visibleDisplayText || maskHiddenToolNames(effectiveOutput), 'result', 'web-read-pdf-fallback')}
                  title="Copy result"
                >
                  {isCopiedButton('result', 'web-read-pdf-fallback') ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
              <pre className="tool-call-code">{visibleDisplayText}</pre>
            </div>
          );
        }

        const pdf = webReadPdfOutput;
        const isErrorState = !pdf.ok || Boolean(pdf.error);
        const finalUrl = pdf.url || pdf.requestedUrl;
        const requestedUrl = pdf.requestedUrl || (toolCall.input?.url as string) || finalUrl;
        const showResolvedUrl = Boolean(finalUrl && finalUrl !== requestedUrl);
        const metaParts: string[] = [];
        if (pdf.textLength > 0) {
          if (pdf.textStartChar != null && pdf.textEndChar != null) {
            metaParts.push(
              `chars ${pdf.textStartChar.toLocaleString()}\u2013${pdf.textEndChar.toLocaleString()} of ${pdf.textLength.toLocaleString()}`,
            );
          } else {
            metaParts.push(`${pdf.textLength.toLocaleString()} chars`);
          }
        }
        if (pdf.truncated) metaParts.push('truncated');
        if (pdf.matches.length > 0) {
          metaParts.push(`${pdf.matches.length} match${pdf.matches.length === 1 ? '' : 'es'}`);
        }
        if (typeof pdf.durationMs === 'number') metaParts.push(`${pdf.durationMs} ms`);

        return (
          <div className="tool-call-section tool-call-web-section">
            <div className="tool-call-section-header">
              <span className="tool-call-section-label">
                <FileText size={12} />
                <span>Web page</span>
              </span>
              <div className="tool-call-terminal-header-actions">
                {finalUrl && (
                  <a
                    className="tool-call-copy-btn tool-call-web-open-btn"
                    href={finalUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    title="Open in new tab"
                  >
                    <Link size={12} />
                  </a>
                )}
                {pdf.text && (
                  <button
                    className="tool-call-copy-btn"
                    onClick={() => copyToClipboard(pdf.text, 'result', 'web-read-pdf-text')}
                    title="Copy page text"
                  >
                    {isCopiedButton('result', 'web-read-pdf-text') ? <Check size={12} /> : <Copy size={12} />}
                  </button>
                )}
              </div>
            </div>
            {requestedUrl && (
              <div className="tool-call-web-query" title={requestedUrl}>
                <a
                  className="tool-call-web-query-link"
                  href={requestedUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={requestedUrl}
                >
                  <FileText size={12} aria-hidden="true" />
                  <span>{requestedUrl}</span>
                </a>
                <button
                  className="tool-call-copy-btn"
                  onClick={() => copyToClipboard(requestedUrl, 'query', 'web-read-pdf-requested-url')}
                  title="Copy URL"
                >
                  {isCopiedButton('query', 'web-read-pdf-requested-url') ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
            )}
            {pdf.query && (
              <div className="tool-call-web-page-head">
                <div className="tool-call-web-page-title" title={pdf.query}>
                  <Search size={12} aria-hidden="true" />
                  <span>{pdf.query}</span>
                </div>
              </div>
            )}
            {showResolvedUrl && finalUrl && (
              <div className="tool-call-web-page-head">
                <a
                  className="tool-call-web-page-url"
                  href={finalUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={finalUrl}
                >
                  {finalUrl}
                </a>
              </div>
            )}
            {metaParts.length > 0 && (
              <div className="tool-call-web-meta">{metaParts.join(' \u00b7 ')}</div>
            )}
            {isErrorState && (
              <div className="tool-call-web-error">
                <AlertCircle size={12} />
                <span>{pdf.error || 'Read failed.'}</span>
              </div>
            )}
            {pdf.matches.length > 0 && (
              <details className="tool-call-web-links" open>
                <summary>
                  {pdf.matches.length} match{pdf.matches.length === 1 ? '' : 'es'}
                </summary>
                <ul className="tool-call-web-links-list">
                  {pdf.matches.map((match, idx) => (
                    <li
                      key={`${match.matchStart ?? idx}-${idx}`}
                      className="tool-call-web-link-item"
                    >
                      {match.matchStart != null && (
                        <div className="tool-call-web-meta">
                          char {match.matchStart.toLocaleString()}
                          {match.matchEnd != null
                            ? `\u2013${match.matchEnd.toLocaleString()}`
                            : ''}
                        </div>
                      )}
                      {match.text && (
                        <pre className="tool-call-web-page-text">{match.text}</pre>
                      )}
                    </li>
                  ))}
                </ul>
              </details>
            )}
            {pdf.matches.length === 0 && pdf.text && (
              <div className="tool-call-web-page-text-wrap">
                <pre className="tool-call-web-page-text">{pdf.text}</pre>
              </div>
            )}
          </div>
        );
      }

      default: {
        return (
          <div className="tool-call-section">
            <div className="tool-call-section-header">
              <span className="tool-call-section-label">Result:</span>
              <button
                className="tool-call-copy-btn"
                onClick={() => copyToClipboard(visibleDisplayText || maskHiddenToolNames(effectiveOutput), 'result', 'tool-default-result')}
                title="Copy result"
              >
                {isCopiedButton('result', 'tool-default-result') ? <Check size={12} /> : <Copy size={12} />}
              </button>
            </div>
            {tableData ? (
              <DataTable data={tableData} />
            ) : (
              <pre className="tool-call-code">{visibleDisplayText}</pre>
            )}
          </div>
        );
      }
    }
  };

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
            <span className="tool-call-name">{visualToolName}</span>
          )}
          {toolCall.status === 'running' && toolCall.generating_lines ? (
            <span className="tool-call-progress">{toolCall.generating_lines} lines</span>
          ) : null}
          <span className="tool-call-toggle" aria-hidden="true">
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </span>
        </button>
        {allowRerun && isFailed && isVisualizationTool && !isRetrying && (
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
          <span
            className="tool-call-retrying"
            title="Retry checks existing source data first, then may rerun the source query and use AI to repair malformed visualization data."
          >
            <MiniLoadingSpinner variant="icon" size={12} />
            <span>{retryProgressText || 'Retrying...'}</span>
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
          {inputDisplay && !userspaceWriteResult && !userspaceReadResult && toolCall.tool !== 'list_userspace_files' && !isTerminalCommand && toolCall.tool !== 'web_search' && toolCall.tool !== 'web_browse' && toolCall.tool !== WEB_READ_PDF_TOOL_ID && (
            <div className="tool-call-section">
              <div className="tool-call-section-header">
                <span className="tool-call-section-label">Query:</span>
                <div className="tool-call-terminal-header-actions">
                  <button
                    className="tool-call-copy-btn"
                    onClick={() => copyToClipboard(inputDisplay, 'query', 'generic-query')}
                    title="Copy query"
                  >
                    {isCopiedButton('query', 'generic-query') ? <Check size={12} /> : <Copy size={12} />}
                  </button>
                  {isSqlTool && canRerun && hasRerunContext && (
                    <button
                      className="tool-call-copy-btn tool-call-terminal-rerun-btn"
                      onClick={handleRerunCommand}
                      title="Replay query"
                      disabled={isRerunning}
                    >
                      {isRerunning ? <MiniLoadingSpinner variant="icon" size={12} /> : <Play size={12} />}
                    </button>
                  )}
                </div>
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
                      onClick={() => copyToClipboard(inputDisplay, 'query', 'terminal-copy-command-complete')}
                      title="Copy command"
                    >
                      {isCopiedButton('query', 'terminal-copy-command-complete') ? <Check size={12} /> : <Terminal size={12} />}
                    </button>
                    <button
                      className="tool-call-copy-btn"
                      onClick={() => copyToClipboard(
                        [terminalOutput.stdout, terminalOutput.stderr].filter(Boolean).join('\n') || inputDisplay,
                        'result',
                        'terminal-copy-output-complete'
                      )}
                      title="Copy output"
                    >
                      {isCopiedButton('result', 'terminal-copy-output-complete') ? <Check size={12} /> : <Copy size={12} />}
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
                <div className="tool-call-terminal-header-bar">
                  <span className="tool-call-terminal-cwd">~</span>
                  <div className="tool-call-terminal-header-actions">
                    <button
                      className="tool-call-copy-btn"
                      onClick={() => copyToClipboard(inputDisplay, 'query', 'terminal-copy-command-running')}
                      title="Copy command"
                    >
                      {isCopiedButton('query', 'terminal-copy-command-running') ? <Check size={12} /> : <Terminal size={12} />}
                    </button>
                    <button
                      className="tool-call-copy-btn"
                      onClick={() => copyToClipboard(activeTerminalOutput || inputDisplay, 'result', 'terminal-copy-output-running')}
                      title="Copy output"
                    >
                      {isCopiedButton('result', 'terminal-copy-output-running') ? <Check size={12} /> : <Copy size={12} />}
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
                      onClick={() => copyToClipboard(inputDisplay, 'query', 'terminal-copy-command-empty')}
                      title="Copy command"
                    >
                      {isCopiedButton('query', 'terminal-copy-command-empty') ? <Check size={12} /> : <Terminal size={12} />}
                    </button>
                    <button
                      className="tool-call-copy-btn"
                      onClick={() => copyToClipboard(activeTerminalOutput || inputDisplay, 'result', 'terminal-copy-output-empty')}
                      title="Copy output"
                    >
                      {isCopiedButton('result', 'terminal-copy-output-empty') ? <Check size={12} /> : <Copy size={12} />}
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
          {effectiveOutput && !isTerminalCommand && (
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
            ) : renderUserspaceResultBody()
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
    {showUserspaceDiffOverlay && userspaceWriteBatch && (
      <FileDiffOverlay
        key="userspace-file-diff-overlay"
        entries={userspaceOverlayEntries}
        activeIndex={userspaceOverlayActiveIndex}
        title={userspaceWriteBatch.isBatch ? 'Userspace Batch Diff' : 'Userspace File Diff'}
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
  durationSeconds,
  visibility = 'compact',
  showToolCalls = true,
  workspaceId,
  conversationId,
  onOpenWorkspaceFile,
}: {
  content: string;
  isComplete: boolean;
  parts?: ReasoningPart[];
  durationSeconds?: number;
  visibility?: 'compact' | 'expanded' | 'hidden';
  showToolCalls?: boolean;
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
  const shouldAutoScrollContentRef = useRef(true);
  const startTimeRef = useRef<number>(Date.now());
  const [elapsed, setElapsed] = useState(0);

  const handleToggle = useCallback(() => {
    userToggledRef.current = true;
    setIsExpanded(prev => !prev);
  }, []);

  // Strip generated tool-call markup from reasoning text. Executed non-artifact
  // tools can render inline from structured commentary-channel events.
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
  const hasInlineToolCalls = showToolCalls && renderedParts.some((part) => part.type === 'tool' && !!part.toolCall);

  // Elapsed timer while streaming
  useEffect(() => {
    if (isComplete) return;
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [isComplete]);

  const handleContentScroll = useCallback(() => {
    if (!contentRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = contentRef.current;
    shouldAutoScrollContentRef.current = scrollHeight - scrollTop - clientHeight < 24;
  }, []);

  // Auto-scroll reasoning content while streaming
  useEffect(() => {
    if (!isComplete && isExpanded && contentRef.current && shouldAutoScrollContentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [content, isComplete, isExpanded]);

  useEffect(() => {
    if (!isComplete && isExpanded) {
      shouldAutoScrollContentRef.current = true;
    }
  }, [isComplete, isExpanded]);

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
  const toolCount = renderedParts.filter((part) => part.type === 'tool' && !!part.toolCall).length;

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

  const normalizeReasoningListIndentation = useCallback((text: string) => {
    const lines = text.split('\n');
    let inIndentedList = false;

    return lines.map((line) => {
      if (!line.trim()) {
        return '';
      }

      if (/^ {4,}(?:[*+-]\s|\d+[.)]\s)/.test(line)) {
        inIndentedList = true;
        return line.replace(/^ {4}/, '');
      }

      if (inIndentedList) {
        if (/^ {4,}\S/.test(line)) {
          return line.replace(/^ {4}/, '');
        }
        inIndentedList = false;
      }

      return line;
    }).join('\n');
  }, []);

  // Convert standalone **title** lines in reasoning text to ### markdown headers
  // so the model's section headings are visually distinct rather than showing raw asterisks.
  const formatReasoningText = useCallback((text: string) => {
    let result = text.replace(/\r\n?/g, '\n');
    result = normalizeReasoningListIndentation(result);
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
  }, [normalizeReasoningListIndentation]);

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
          {showToolCalls && toolCount > 0 && (
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
        <div className="reasoning-content-inner" ref={contentRef} onScroll={handleContentScroll}>
          {hasInlineToolCalls ? renderedParts.map((part, index) => {
            if (part.type === 'text') {
              return (
                <div key={index} className="markdown-content">
                  <MemoizedMarkdown content={formatReasoningText(part.text ?? '')} />
                </div>
              );
            }
            if (!part.toolCall) return null;
            return (
              <div key={index} className="chat-tool-calls reasoning-embedded-tool">
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
        parts={segment.reasoningParts}
        durationSeconds={segment.durationSeconds}
        showToolCalls={showToolCalls}
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
export function parseMessageContent(content: string | ContentPart[]): { text: string; attachments: ContentPart[] } {
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

// Component to display message attachments
interface MessageAttachmentsProps {
  attachments: ContentPart[];
  onImageClick?: (url: string) => void;
}

export const MessageAttachments = memo(function MessageAttachments({ attachments, onImageClick }: MessageAttachmentsProps) {
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

// --- Conversation search + archive helpers ----------------------------------
// Per-user preference: how many days back the sidebar shows by default.
// Conversations older than this cutoff are "archived" and lazy-loaded only
// when the user opens the archive modal or types a search query.
const CHAT_ARCHIVE_AGE_DAYS_KEY_PREFIX = 'chat-archive-age-days:';
const CHAT_ARCHIVE_AGE_DAYS_DEFAULT = 30;
const CHAT_ARCHIVE_AGE_PRESET_DAYS = [1, 7, 30, 365] as const;

function normalizeArchiveAgePreset(days: number): number {
  if (!Number.isFinite(days) || days <= 1) return 1;
  if (days <= 7) return 7;
  if (days <= 30) return 30;
  return 365;
}

function getArchiveAgeLabel(days: number): string {
  const normalized = normalizeArchiveAgePreset(days);
  if (normalized === 1) return '1d';
  if (normalized === 7) return '1wk';
  if (normalized === 30) return '1mo';
  return '1yr';
}

function getNextArchiveAgePreset(days: number): number {
  const normalized = normalizeArchiveAgePreset(days);
  const index = CHAT_ARCHIVE_AGE_PRESET_DAYS.indexOf(normalized as (typeof CHAT_ARCHIVE_AGE_PRESET_DAYS)[number]);
  const nextIndex = index >= 0 ? (index + 1) % CHAT_ARCHIVE_AGE_PRESET_DAYS.length : 0;
  return CHAT_ARCHIVE_AGE_PRESET_DAYS[nextIndex];
}

function getArchiveAgeStorageKey(userId: string): string {
  return `${CHAT_ARCHIVE_AGE_DAYS_KEY_PREFIX}${userId}`;
}

function readStoredArchiveAgeDays(userId: string): number {
  try {
    const raw = localStorage.getItem(getArchiveAgeStorageKey(userId));
    if (raw === null) return CHAT_ARCHIVE_AGE_DAYS_DEFAULT;
    const parsed = parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed < 0) return CHAT_ARCHIVE_AGE_DAYS_DEFAULT;
    return normalizeArchiveAgePreset(parsed);
  } catch {
    return CHAT_ARCHIVE_AGE_DAYS_DEFAULT;
  }
}

function archiveCutoffIso(daysAgo: number): string | null {
  if (!Number.isFinite(daysAgo) || daysAgo <= 0) return null;
  const cutoff = new Date(Date.now() - daysAgo * 24 * 60 * 60 * 1000);
  return cutoff.toISOString();
}

function isConversationOlderThanWindow(conversation: Conversation, daysAgo: number): boolean {
  const cutoffIso = archiveCutoffIso(daysAgo);
  if (!cutoffIso) return false;
  const updatedAtMs = Date.parse(conversation.updated_at);
  const cutoffMs = Date.parse(cutoffIso);
  if (!Number.isFinite(updatedAtMs) || !Number.isFinite(cutoffMs)) {
    return false;
  }
  return updatedAtMs < cutoffMs;
}

type ConversationArchiveCursor = {
  updatedAt: string;
  id: string;
};

const ARCHIVE_PAGE_SIZE = 50;

function getConversationWorkspaceId(conversation: Conversation): string | null {
  const camelWorkspaceId = (conversation as Conversation & { workspaceId?: string | null }).workspaceId;
  return conversation.workspace_id ?? camelWorkspaceId ?? null;
}

function getConversationArchiveCursor(conversations: Conversation[]): ConversationArchiveCursor | null {
  const lastConversation = conversations[conversations.length - 1];
  if (!lastConversation) return null;
  return {
    updatedAt: lastConversation.updated_at,
    id: lastConversation.id,
  };
}

function getConversationMessageCount(conversation: Conversation): number {
  return conversation.message_count ?? conversation.messages.length;
}

function mergeConversationPages(current: Conversation[], incoming: Conversation[]): Conversation[] {
  if (current.length === 0) return incoming;
  if (incoming.length === 0) return current;
  const seen = new Set(current.map((conversation) => conversation.id));
  const merged = [...current];
  incoming.forEach((conversation) => {
    if (seen.has(conversation.id)) return;
    seen.add(conversation.id);
    merged.push(conversation);
  });
  return merged;
}

// Best-effort plain-text view of a message for content searching. Mirrors
// what the user sees on screen, but flattens to a single string. Tool I/O
// blobs are intentionally excluded — they're noisy and rarely searchable.
function getConversationSearchText(conversation: Conversation): string {
  const parts: string[] = [conversation.title || ''];
  for (const msg of conversation.messages) {
    try {
      const { text } = parseMessageContent(msg.content);
      if (text) parts.push(text);
    } catch {
      // Ignore message parse failures — we don't want a single bad
      // message to make the whole conversation un-searchable.
    }
  }
  return parts.join('\n');
}

function conversationMatchesQuery(conversation: Conversation, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return getConversationSearchText(conversation).toLowerCase().includes(needle);
}

function buildConversationSnippet(conversation: Conversation, query: string, radius = 60): string | null {
  const needle = query.trim().toLowerCase();
  if (!needle) return null;
  // Skip if the title alone matches — the title is highlighted separately.
  if ((conversation.title || '').toLowerCase().includes(needle)) return null;
  for (const msg of conversation.messages) {
    let text: string;
    try {
      text = parseMessageContent(msg.content).text;
    } catch {
      continue;
    }
    if (!text) continue;
    const idx = text.toLowerCase().indexOf(needle);
    if (idx === -1) continue;
    const start = Math.max(0, idx - radius);
    const end = Math.min(text.length, idx + needle.length + radius);
    const prefix = start > 0 ? '…' : '';
    const suffix = end < text.length ? '…' : '';
    return `${prefix}${text.slice(start, end).replace(/\s+/g, ' ').trim()}${suffix}`;
  }
  return null;
}

const HighlightedText = memo(function HighlightedText({ text, query }: { text: string; query: string }) {
  if (!text) return null;
  const needle = query.trim();
  if (!needle) return <>{text}</>;
  const lowerText = text.toLowerCase();
  const lowerNeedle = needle.toLowerCase();
  const segments: ReactNode[] = [];
  let cursor = 0;
  let keyId = 0;
  while (cursor < text.length) {
    const matchIndex = lowerText.indexOf(lowerNeedle, cursor);
    if (matchIndex === -1) {
      segments.push(<span key={`s-${keyId++}`}>{text.slice(cursor)}</span>);
      break;
    }
    if (matchIndex > cursor) {
      segments.push(<span key={`s-${keyId++}`}>{text.slice(cursor, matchIndex)}</span>);
    }
    segments.push(
      <mark key={`m-${keyId++}`} className="chat-search-highlight">
        {text.slice(matchIndex, matchIndex + lowerNeedle.length)}
      </mark>,
    );
    cursor = matchIndex + lowerNeedle.length;
  }
  return <>{segments}</>;
});

interface ChatPanelProps {
  currentUser: User;
  debugMode?: boolean;
  initialConversationId?: string | null;
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
  onBranchSwitch?: (branchId: string | null, associatedSnapshotId: string | null) => void | Promise<void>;
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
  onOpenShareModal?: () => void;
  canShareConversation?: boolean;
  onShareConversationAtMessage?: (messageIdx: number) => void;
}

export function ChatPanel({
  currentUser,
  debugMode = false,
  initialConversationId,
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
  onOpenShareModal,
  canShareConversation = false,
  onShareConversationAtMessage,
}: ChatPanelProps) {
  const MIN_INPUT_AREA_HEIGHT = 96;
  const INPUT_AREA_COLLAPSE_THRESHOLD = 80;
  const chatLayoutCookieName = getChatLayoutCookieName(currentUser.id);

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [isConversationSwitchLoading, setIsConversationSwitchLoading] = useState(false);
  const [isCreatingFreshConversation, setIsCreatingFreshConversation] = useState(false);
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
  // Separate state for the chat-header rename input so its `autoFocus` does not
  // steal focus from the sidebar rename input when both would otherwise mount
  // simultaneously for the active conversation.
  const [editingHeaderTitle, setEditingHeaderTitle] = useState<string | null>(null);
  const [titleInput, setTitleInput] = useState('');
  const [editingMessageIdx, setEditingMessageIdx] = useState<number | null>(null);
  const [editMessageContent, setEditMessageContent] = useState('');
  const [editMessageAttachments, setEditMessageAttachments] = useState<AttachmentFile[]>([]);
  const [hitMaxIterations, setHitMaxIterations] = useState(false);

  // Chat branching state
  const [branchPoints, setBranchPoints] = useState<ConversationBranchPointInfo[]>([]);
  const [branchSwitching, setBranchSwitching] = useState(false);
  const [branchSelections, setBranchSelections] = useState<Record<string, string>>({});
  const [copiedMessageIdx, setCopiedMessageIdx] = useState<number | null>(null);
  const [pendingDeleteIdx, setPendingDeleteIdx] = useState<number | null>(null);
  const activeConversationId = activeConversation?.id ?? null;
  const branchGroupsByIndex = useMemo(() => {
    const messageCount = activeConversation?.messages.length ?? 0;
    const grouped = new Map<number, BranchRenderGroup[]>();

    for (const point of branchPoints) {
      // Skip branches whose anchor message no longer exists in the current
      // (possibly optimistically truncated) message list. Without this,
      // getConversationBranchAnchorIndex's Math.min clamp would pin a
      // deeper-message branch group onto the last visible user message
      // during the brief window between edit/replay/delete creating the
      // new branch and the streamed assistant reply re-extending the
      // conversation. See the "Oddly the anchor from messages further on
      // attaches to the user message until streaming begins" report.
      const anchorTarget = point.branch_point_index;
      if (anchorTarget >= messageCount) continue;

      const groupsForPoint = new Map<number, ConversationBranchSummary[]>();
      for (const branch of point.branches) {
        const anchorIndex = getConversationBranchAnchorIndex(
          point.branch_point_index,
          branch.branch_kind,
          messageCount,
        );
        const existing = groupsForPoint.get(anchorIndex);
        if (existing) {
          existing.push(branch);
        } else {
          groupsForPoint.set(anchorIndex, [branch]);
        }
      }

      for (const [anchorIndex, branches] of groupsForPoint.entries()) {
        const next = grouped.get(anchorIndex) ?? [];
        next.push({
          anchorIndex,
          selectionKey: getConversationBranchSelectionKey(point.branch_point_index, anchorIndex),
          sourceBranchPointIndex: point.branch_point_index,
          branches,
        });
        grouped.set(anchorIndex, next);
      }
    }

    for (const groups of grouped.values()) {
      groups.sort((left, right) => left.sourceBranchPointIndex - right.sourceBranchPointIndex);
    }

    return grouped;
  }, [branchPoints, activeConversation?.messages.length]);
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
  // Sidebar conversation search + archive lazy-load state. Archive holds
  // conversations older than the active cutoff; loaded only on demand to
  // keep ?view=chat boot fast for users with long histories.
  const [conversationSearchQuery, setConversationSearchQuery] = useState('');
  const [archiveAgeDays, setArchiveAgeDays] = useState<number>(() => readStoredArchiveAgeDays(currentUser.id));
  const [archivedConversations, setArchivedConversations] = useState<Conversation[]>([]);
  const [archivedConversationCount, setArchivedConversationCount] = useState<number | null>(null);
  const [archiveCountLoaded, setArchiveCountLoaded] = useState(false);
  const [archiveCountLoading, setArchiveCountLoading] = useState(false);
  const [archiveLoaded, setArchiveLoaded] = useState(false);
  const [archiveFullyLoaded, setArchiveFullyLoaded] = useState(false);
  const [archiveLoading, setArchiveLoading] = useState(false);
  const [archiveError, setArchiveError] = useState<string | null>(null);
  const [archiveCursor, setArchiveCursor] = useState<ConversationArchiveCursor | null>(null);
  const [workspaceArchivedConversations, setWorkspaceArchivedConversations] = useState<Conversation[]>([]);
  const [workspaceArchiveLoaded, setWorkspaceArchiveLoaded] = useState(false);
  const [workspaceArchiveLoading, setWorkspaceArchiveLoading] = useState(false);
  const [showArchiveModal, setShowArchiveModal] = useState(false);
  const [archiveSearchQuery, setArchiveSearchQuery] = useState('');
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
    meta: modelsMeta,
    readiness: modelsReadiness,
    refresh: refreshModels,
    awaitReady: awaitModelsReady,
  } = useAvailableModels();
  const [isWorkspaceConversationMenuOpen, setIsWorkspaceConversationMenuOpen] = useState(false);
  const [workspaceConversationSearchQuery, setWorkspaceConversationSearchQuery] = useState('');
  const workspaceConversationSearchInputRef = useRef<HTMLInputElement | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [conversationMembers, setConversationMembers] = useState<ConversationMember[]>([]);
  const [conversationToolIds, setConversationToolIds] = useState<string[]>([]);
  const [conversationToolGroupIds, setConversationToolGroupIds] = useState<string[]>([]);
  const [conversationDisabledBuiltInToolIds, setConversationDisabledBuiltInToolIds] = useState<string[]>([]);
  const [availableTools, setAvailableTools] = useState<UserSpaceAvailableTool[]>([]);
  const [toolGroups, setToolGroups] = useState<ToolGroupInfo[]>([]);
  const [showMembersModal, setShowMembersModal] = useState(false);
  const [memberPickerUsers, setMemberPickerUsers] = useState<UserDirectoryEntry[]>([]);
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

  const initialConversationIdRef = useRef<string | null>(
    initialConversationId && initialConversationId.trim() ? initialConversationId.trim() : null,
  );

  useEffect(() => {
    if (initialConversationId && initialConversationId.trim()) {
      initialConversationIdRef.current = initialConversationId.trim();
    }
  }, [initialConversationId]);

  const effectiveAvailableTools = useWorkspaceToolSource
    ? (workspaceAvailableTools ?? [])
    : availableTools;
  const effectiveToolIds = useWorkspaceToolSource
    ? (workspaceSelectedToolIds ?? [])
    : conversationToolIds;
  const resolvedConversationToolIds = useMemo(
    () => resolveDefaultSelectedToolIds(conversationToolIds, conversationToolGroupIds, effectiveAvailableTools),
    [conversationToolIds, conversationToolGroupIds, effectiveAvailableTools]
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
  const hasWorkspaceConversationContext = Boolean(
    workspaceId || activeConversation?.workspace_id || activeConversation?.workspaceId,
  );
  const conversationBuiltInTools = hasWorkspaceConversationContext
    ? VISIBLE_WORKSPACE_BUILT_IN_TOOLS
    : VISIBLE_CHAT_BUILT_IN_TOOLS;
  const selectedConversationBuiltInToolIdSet = useMemo(() => {
    const disabledIds = new Set(conversationDisabledBuiltInToolIds);
    return new Set(CHAT_BUILT_IN_TOOL_IDS.filter((id) => !disabledIds.has(id)));
  }, [conversationDisabledBuiltInToolIds]);
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
  const userspaceConversationIdsRef = useRef<Set<string>>(new Set());
  const shouldAutoScrollRef = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const processingTaskRef = useRef<string | null>(null);
  // Tracks which streaming `create_userspace_snapshot` tool_end events have
  // already triggered an `onSnapshotsMaybeChanged` notification, keyed by
  // a stable string per (taskId, eventIndex). Cleared when the task ends.
  const notifiedSnapshotEventKeysRef = useRef<Set<string>>(new Set());
  const titleSourceRef = useRef<Map<string, EventSource>>(new Map());
  // Forward-reference to connectTaskStream so the conversation event SSE
  // handler (defined earlier in the component) can trigger task streaming
  // as soon as a task_started event arrives, without waiting for the
  // periodic task-state poll.
  const connectTaskStreamRef = useRef<((taskId: string) => void) | null>(null);
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
    if (!isWorkspaceConversationMenuOpen) {
      setWorkspaceConversationSearchQuery('');
      return;
    }
    const handle = setTimeout(() => {
      workspaceConversationSearchInputRef.current?.focus();
    }, 0);
    return () => clearTimeout(handle);
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
          next[key] = conv.user_id !== currentUser.id; // current user's group expanded by default
          changed = true;
        }
      });

      return changed ? next : prev;
    });
  }, [collapsedGroups, conversations, currentUser.id, getOwnerKey, isAdmin]);

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

  // Memoized consolidated segments for streaming. Ordinary tool calls can stay
  // inline inside active reasoning; visualization artifacts render standalone.
  const consolidatedSegments = useMemo((): StreamingSegment[] => {
    if (!streamingEvents.length) return [];

    const segments: StreamingSegment[] = [];
    let currentContent = '';
    let currentReasoning = '';
    let currentReasoningParts: ReasoningPart[] = [];
    let currentReasoningDurationSeconds: number | undefined;

    // Flush accumulated reasoning into a NEW reasoning segment (adjacent reasoning merges, non-adjacent stays separate)
    const flushReasoning = (isComplete: boolean) => {
      if (!currentReasoning) return;
      const seg: StreamingSegment = {
        type: 'reasoning',
        content: currentReasoning,
        isComplete,
        durationSeconds: currentReasoningDurationSeconds,
        reasoningParts: currentReasoningParts.length > 0 ? [...currentReasoningParts] : [{ type: 'text', text: currentReasoning }],
      };
      segments.push(seg);
      currentReasoning = '';
      currentReasoningParts = [];
      currentReasoningDurationSeconds = undefined;
    };

    const flushContent = () => {
      if (!currentContent) return;
      segments.push({ type: 'content', content: currentContent });
      currentContent = '';
    };

    for (const ev of streamingEvents) {
      const channel = getChatEventChannel(ev);
      if (channel === 'analysis' && ev.type === 'reasoning') {
        // Flush any pending content first — content breaks reasoning adjacency
        flushContent();
        // Accumulate reasoning (adjacent reasoning events merge)
        currentReasoning += ev.content;
        const lastPart = currentReasoningParts[currentReasoningParts.length - 1];
        if (lastPart && lastPart.type === 'text') {
          lastPart.text = (lastPart.text || '') + ev.content;
        } else {
          currentReasoningParts.push({ type: 'text', text: ev.content });
        }
        if (typeof ev.durationSeconds === 'number') {
          currentReasoningDurationSeconds = ev.durationSeconds;
        }
      } else if (channel === 'final' && ev.type === 'content') {
        // Flush any pending reasoning — it's now complete since content follows
        flushReasoning(true);
        // Accumulate content
        currentContent += ev.content;
      } else if (channel === 'commentary' && ev.type === 'tool') {
        if (currentReasoning && !isVisualizationToolCall(ev.toolCall)) {
          currentReasoningParts.push({ type: 'tool', toolCall: ev.toolCall });
        } else {
          flushReasoning(true);
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
    setArchivedConversations([]);
    setArchivedConversationCount(null);
    setArchiveCountLoaded(false);
    setArchiveCountLoading(false);
    setArchiveLoaded(false);
    setArchiveFullyLoaded(false);
    setArchiveLoading(false);
    setArchiveCursor(null);
    setWorkspaceArchivedConversations([]);
    setWorkspaceArchiveLoaded(false);
    setWorkspaceArchiveLoading(false);
    setArchiveError(null);
    setShowArchiveModal(false);
    setArchiveSearchQuery('');
    setConversationSearchQuery('');
    userspaceConversationIdsRef.current = new Set();
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
    const visibleConversations = nextWorkspaceState.conversations.filter((conversation) => {
      if (conversation.id === nextWorkspaceState.selected_conversation_id) {
        return true;
      }
      return !isConversationOlderThanWindow(conversation, archiveAgeDays);
    });

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
          || existing.message_count !== conversation.message_count
          || existing.updated_at !== conversation.updated_at
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

    setIsConversationListLoading(false);

    if (activeT && (activeT.status === 'pending' || activeT.status === 'running')) {
      setActiveTask(activeT);
      setInterruptedTask(null);
      syncConversationActiveTaskId(selectedConversationId, activeT.id);
      return;
    }

    setActiveTask(null);
    setInterruptedTask(interruptedT ?? null);
    syncConversationActiveTaskId(selectedConversationId, null);
  }, [archiveAgeDays, syncConversationActiveTaskId]);

  const loadConversations = async () => {
    setIsConversationListLoading(true);
    try {
      const workspaceState = workspaceId
        ? (workspaceChatState ?? await api.getWorkspaceChatState(workspaceId, activeConversationRef.current?.id ?? null))
        : null;
      const sinceIso = archiveCutoffIso(archiveAgeDays);
      const [data, workspacePage] = await Promise.all([
        workspaceState?.conversations
          ? Promise.resolve(workspaceState.conversations)
          : api.listConversations(workspaceId, sinceIso ? { since: sinceIso } : undefined),
        !workspaceId
          ? api.listUserSpaceWorkspaces(0, 200).catch((workspaceErr) => {
              console.warn('Failed to load userspace workspaces for conversation filtering:', workspaceErr);
              return null;
            })
          : Promise.resolve(null),
      ]);
      let userspaceConversationIds = new Set<string>();

      if (workspacePage) {
        userspaceConversationIds = new Set(
          workspacePage.items.flatMap((workspace) => workspace.conversation_ids || [])
        );
      }
      userspaceConversationIdsRef.current = userspaceConversationIds;

      const visibleConversations = data.filter((conversation) => {
        const linkedWorkspaceId = getConversationWorkspaceId(conversation);
        if (workspaceId) {
          if (linkedWorkspaceId !== workspaceId) return false;
          if (conversation.id === activeConversationRef.current?.id) return true;
          return !isConversationOlderThanWindow(conversation, archiveAgeDays);
        }
        return !linkedWorkspaceId && !userspaceConversationIds.has(conversation.id);
      });

      setConversations(visibleConversations);
      setActiveConversation((current) => {
        const preferredConversationId = initialConversationIdRef.current;
        if (preferredConversationId) {
          const preferredConversation = visibleConversations.find((conversation) => conversation.id === preferredConversationId);
          if (preferredConversation) {
            initialConversationIdRef.current = null;
            return preferredConversation;
          }
        }

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

  const filterStandaloneConversations = useCallback((items: Conversation[]): Conversation[] => {
    const userspaceConversationIds = userspaceConversationIdsRef.current;
    return items.filter((conversation) => {
      const linkedWorkspaceId = getConversationWorkspaceId(conversation);
      return !linkedWorkspaceId && !userspaceConversationIds.has(conversation.id);
    });
  }, []);

  // Lazy-load global conversations older than the active cutoff.
  const loadArchivedConversationCount = useCallback(async () => {
    if (workspaceId) return;
    if (archiveCountLoaded || archiveCountLoading) return;
    setArchiveCountLoading(true);
    try {
      const cutoff = archiveCutoffIso(archiveAgeDays);
      const response = cutoff
        ? await api.countConversations(undefined, { until: cutoff })
        : { count: 0 };
      setArchivedConversationCount(response.count);
      setArchiveCountLoaded(true);
    } catch (err) {
      console.warn('Failed to load archived conversation count:', err);
    } finally {
      setArchiveCountLoading(false);
    }
  }, [workspaceId, archiveCountLoaded, archiveCountLoading, archiveAgeDays]);

  const loadArchivedConversations = useCallback(async () => {
    if (workspaceId) return;
    if (archiveLoading) return;
    if (archiveLoaded && archiveFullyLoaded) return;

    const isResetLoad = !archiveLoaded;
    setArchiveLoading(true);
    if (isResetLoad) {
      setArchiveError(null);
    }
    try {
      const cutoff = archiveCutoffIso(archiveAgeDays);
      const data = cutoff
        ? await api.listConversations(undefined, {
            until: cutoff,
            limit: ARCHIVE_PAGE_SIZE,
            cursorUpdatedAt: isResetLoad ? null : archiveCursor?.updatedAt ?? null,
            cursorId: isResetLoad ? null : archiveCursor?.id ?? null,
          })
        : [];
      const filtered = filterStandaloneConversations(data);
      const nextArchivedConversations = isResetLoad
        ? filtered
        : mergeConversationPages(archivedConversations, filtered);

      setArchivedConversations(nextArchivedConversations);
      setArchiveCursor(getConversationArchiveCursor(data));
      setArchiveLoaded(true);
      setArchiveFullyLoaded(data.length < ARCHIVE_PAGE_SIZE);
      setArchivedConversationCount((current) => {
        if (current === null) {
          return data.length < ARCHIVE_PAGE_SIZE ? nextArchivedConversations.length : null;
        }
        return Math.max(current, nextArchivedConversations.length);
      });
    } catch (err) {
      console.error('Failed to load archived conversations:', err);
      setArchiveError(err instanceof Error ? err.message : 'Failed to load older chats');
    } finally {
      setArchiveLoading(false);
    }
  }, [workspaceId, archiveLoading, archiveLoaded, archiveFullyLoaded, archiveAgeDays, archiveCursor, archivedConversations, filterStandaloneConversations]);

  // Lazy-load full workspace conversations on search so the picker can still
  // match message bodies without making initial workspace load carry every transcript.
  const loadArchivedWorkspaceConversations = useCallback(async () => {
    if (!workspaceId) return;
    if (workspaceArchiveLoaded || workspaceArchiveLoading) return;
    setWorkspaceArchiveLoading(true);
    try {
      const data = await api.listConversations(workspaceId);
      const filtered = data.filter((conversation) => {
        const linkedWorkspaceId = conversation.workspace_id
          ?? (conversation as Conversation & { workspaceId?: string | null }).workspaceId
          ?? null;
        return linkedWorkspaceId === workspaceId;
      });
      setWorkspaceArchivedConversations(filtered);
      setWorkspaceArchiveLoaded(true);
    } catch (err) {
      console.error('Failed to load archived workspace conversations:', err);
    } finally {
      setWorkspaceArchiveLoading(false);
    }
  }, [workspaceId, workspaceArchiveLoaded, workspaceArchiveLoading]);

  // Preload the standalone archive in the background once the main list is
  // ready so the sidebar badge can show the current count without waiting for
  // the archive modal to open.
  useEffect(() => {
    if (workspaceId || embedded) return;
    if (isConversationListLoading) return;
    if (archiveCountLoaded || archiveCountLoading) return;
    void loadArchivedConversationCount();
  }, [
    workspaceId,
    embedded,
    isConversationListLoading,
    archiveCountLoaded,
    archiveCountLoading,
    loadArchivedConversationCount,
  ]);

  // Auto-load archive when the user starts searching so results from older
  // chats appear without an extra click. Debounced through the search input.
  useEffect(() => {
    if (workspaceId) return;
    if (!conversationSearchQuery.trim()) return;
    if (archiveLoading) return;
    if (archiveLoaded && archiveFullyLoaded) return;
    void loadArchivedConversations();
  }, [conversationSearchQuery, workspaceId, archiveLoaded, archiveFullyLoaded, archiveLoading, loadArchivedConversations]);

  useEffect(() => {
    if (!workspaceId) return;
    if (!workspaceConversationSearchQuery.trim()) return;
    if (workspaceArchiveLoaded || workspaceArchiveLoading) return;
    void loadArchivedWorkspaceConversations();
  }, [
    workspaceConversationSearchQuery,
    workspaceId,
    workspaceArchiveLoaded,
    workspaceArchiveLoading,
    loadArchivedWorkspaceConversations,
  ]);

  // Persist the archive cutoff per-user.
  useEffect(() => {
    try {
      localStorage.setItem(getArchiveAgeStorageKey(currentUser.id), String(archiveAgeDays));
    } catch {
      // localStorage may be unavailable (private mode) — ignore.
    }
  }, [archiveAgeDays, currentUser.id]);

  // When the user changes the cutoff, mark the existing archive lists as
  // stale so search/modal trigger a fresh fetch. We intentionally keep the
  // current items in state so the UI does not flash empty during refetch.
  useEffect(() => {
    setArchivedConversationCount(null);
    setArchiveCountLoaded(false);
    setArchiveLoaded(false);
    setArchiveFullyLoaded(false);
    setArchiveCursor(null);
    setWorkspaceArchiveLoaded(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [archiveAgeDays]);

  // While the archive modal is open, refetch whenever the cached data is
  // stale (e.g. after toggling the archive window).
  useEffect(() => {
    if (!showArchiveModal) return;
    if (workspaceId) return;
    if (archiveLoaded || archiveLoading) return;
    void loadArchivedConversations();
  }, [showArchiveModal, archiveLoaded, archiveLoading, workspaceId, loadArchivedConversations]);

  useEffect(() => {
    if (!showArchiveModal) return;
    if (workspaceId) return;
    if (!archiveSearchQuery.trim()) return;
    if (archiveLoading) return;
    if (archiveLoaded && archiveFullyLoaded) return;
    void loadArchivedConversations();
  }, [showArchiveModal, archiveSearchQuery, workspaceId, archiveLoaded, archiveFullyLoaded, archiveLoading, loadArchivedConversations]);

  const handleArchiveModalScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    if (archiveLoading || !archiveLoaded || archiveFullyLoaded) return;
    const target = event.currentTarget;
    const remaining = target.scrollHeight - target.scrollTop - target.clientHeight;
    if (remaining > 160) return;
    void loadArchivedConversations();
  }, [archiveLoading, archiveLoaded, archiveFullyLoaded, loadArchivedConversations]);

  const cycleArchiveAgePreset = useCallback(() => {
    setArchiveAgeDays((current) => getNextArchiveAgePreset(current));
  }, []);

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
      setConversationDisabledBuiltInToolIds(
        normalizeDisabledBuiltInToolIds(data.disabled_builtin_tool_ids),
      );
    } catch (err) {
      console.error('Failed to fetch conversation tools:', err);
      setConversationToolIds([]);
      setConversationToolGroupIds([]);
      setConversationDisabledBuiltInToolIds([]);
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
      const catalog = await fetchUserSpaceToolCatalog();
      if (catalog.toolsError) {
        console.error('Failed to fetch available tools:', catalog.toolsError);
      }
      if (catalog.toolGroupsError) {
        console.error('Failed to fetch tool groups:', catalog.toolGroupsError);
      }
      setAvailableTools(catalog.availableTools);
      setToolGroups(catalog.toolGroups);
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
      setConversationDisabledBuiltInToolIds([]);
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
      const users = await api.listUsersDirectory();
      setMemberPickerUsers(users);
    } catch {
      setMemberPickerUsers([]);
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

  const persistConversationBuiltInToolSelection = useCallback(async (nextDisabledBuiltInToolIds: string[]) => {
    if (!activeConversation) return;

    const normalizedDisabledBuiltInToolIds = normalizeDisabledBuiltInToolIds(nextDisabledBuiltInToolIds);

    setSavingTools(true);
    try {
      await api.updateConversationTools(activeConversation.id, {
        tool_config_ids: resolvedConversationToolIds,
        tool_group_ids: conversationToolGroupIds,
        disabled_builtin_tool_ids: normalizedDisabledBuiltInToolIds,
      });
      setConversationDisabledBuiltInToolIds(normalizedDisabledBuiltInToolIds);
      const updatedConversation: Conversation = {
        ...activeConversation,
        disabled_builtin_tool_ids: normalizedDisabledBuiltInToolIds,
      };
      setActiveConversation(updatedConversation);
      setConversations(prev => prev.map(c => c.id === updatedConversation.id ? updatedConversation : c));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update built-in tool selection');
    } finally {
      setSavingTools(false);
    }
  }, [activeConversation, conversationToolGroupIds, resolvedConversationToolIds]);

  const handleToggleConversationBuiltInTool = useCallback(async (toolId: string) => {
    if (!activeConversation || isConversationViewer || hasWorkspaceConversationContext) return;
    if (!CHAT_BUILT_IN_TOOL_ID_SET.has(toolId)) return;

    const nextDisabled = new Set(conversationDisabledBuiltInToolIds);
    if (nextDisabled.has(toolId)) {
      nextDisabled.delete(toolId);
      if (toolId === WEB_BROWSE_TOOL_ID) {
        nextDisabled.delete(WEB_READ_PDF_TOOL_ID);
      }
    } else {
      nextDisabled.add(toolId);
      if (toolId === WEB_BROWSE_TOOL_ID) {
        nextDisabled.add(WEB_READ_PDF_TOOL_ID);
      }
    }

    await persistConversationBuiltInToolSelection(
      CHAT_BUILT_IN_TOOL_IDS.filter((id) => nextDisabled.has(id)),
    );
  }, [activeConversation, conversationDisabledBuiltInToolIds, hasWorkspaceConversationContext, isConversationViewer, persistConversationBuiltInToolSelection]);

  const handleToggleConversationTool = useCallback(async (toolId: string) => {
    if (!activeConversation || isConversationViewer) return;
    const targetTool = effectiveAvailableTools.find((tool) => tool.id === toolId);
    const currentGroupIds = new Set(conversationToolGroupIds);

    if (targetTool?.group_id && currentGroupIds.has(targetTool.group_id)) {
      const nextGroupIds = new Set(currentGroupIds);
      nextGroupIds.delete(targetTool.group_id);
      const nextSelected = new Set(
        getUserSpaceGroupToolIds(effectiveAvailableTools, targetTool.group_id).filter((id) => id !== toolId)
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
  }, [activeConversation, conversationToolGroupIds, effectiveAvailableTools, isConversationViewer, persistConversationToolSelection, resolvedConversationToolIds]);

  const handleToggleConversationToolGroup = useCallback(async (groupId: string) => {
    if (!activeConversation || isConversationViewer) return;
    const groupToolIds = getUserSpaceGroupToolIds(effectiveAvailableTools, groupId);
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
  }, [activeConversation, conversationToolGroupIds, effectiveAvailableTools, isConversationViewer, persistConversationToolSelection, resolvedConversationToolIds]);

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
    const selection = resolveConversationModelSelection(
      storedModel,
      availableModels,
    );
    if (!selection.modelId) {
      return defaultContextLimit;
    }

    if (selection.matchedModel) {
      return selection.matchedModel.context_limit;
    }

    return defaultContextLimit;
  }, [availableModels, defaultContextLimit, modelsMeta]);

  const applyCreatedConversation = useCallback((conversation: Conversation) => {
    setConversations(prev => [conversation, ...prev]);
    setActiveConversation(conversation);
    setConversationToolIds([]);
    setConversationToolGroupIds([]);
    setConversationDisabledBuiltInToolIds(
      normalizeDisabledBuiltInToolIds(conversation.disabled_builtin_tool_ids || []),
    );
  }, []);

  const createNewConversation = async () => {
    if (isReadOnly || isCreatingFreshConversation) return;
    try {
      setIsCreatingFreshConversation(true);
      shouldAutoScrollRef.current = true;
      const conversation = await api.createConversation(undefined, workspaceId);
      applyCreatedConversation(conversation);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create conversation');
    } finally {
      setIsCreatingFreshConversation(false);
    }
  };


  // Listen for conversation events (auto-generated titles + chat task
  // lifecycle) per active conversation using SSE.
  // IMPORTANT: We only open ONE SSE connection for the active conversation to
  // avoid exhausting the browser's HTTP/1.1 connection limit (6 per origin).
  // Opening SSE streams for every conversation would saturate the
  // connection pool and lock up the UI.
  const stopTitleStreamFor = useCallback((conversationId: string) => {
    const es = titleSourceRef.current.get(conversationId);
    if (es) {
      es.close();
    }
    titleSourceRef.current.delete(conversationId);
  }, []);

  const startTitleStreamFor = useCallback((conversationId: string, _title: string) => {
    if (titleSourceRef.current.has(conversationId)) return;

    try {
      const url = api.getConversationEventsUrl(conversationId, workspaceId);
      const es = new EventSource(url, { withCredentials: true });

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Auto-generated title arrived
          if (data.type === 'title_update' && data.title) {
            setConversations(prev => prev.map(c => {
               if (c.id === conversationId) {
                 return { ...c, title: data.title };
               }
               return c;
            }));
            setActiveConversation(prev => {
              if (prev && prev.id === conversationId) {
                return { ...prev, title: data.title };
              }
              return prev;
            });
            return;
          }

          // Background chat task lifecycle events. These let us pick up a
          // newly-running task immediately instead of waiting for the
          // periodic /task-state poll, which makes streaming feel instant.
          if (data.event === 'task_started' && data.task_id) {
            connectTaskStreamRef.current?.(data.task_id);
            return;
          }

          if (data.event === 'task_completed') {
            // The per-task SSE stream already drives streaming UI cleanup;
            // refresh the conversation here so persisted assistant messages
            // and any post-completion state (e.g. interrupted_task) are
            // reflected even if the task SSE was not active.
            void api.getConversation(conversationId, workspaceId)
              .then(fresh => {
                setActiveConversation(prev => (
                  prev && prev.id === conversationId ? fresh : prev
                ));
                setConversations(prev => prev.map(c => (
                  c.id === conversationId ? { ...c, ...fresh } : c
                )));
              })
              .catch(() => {});
            return;
          }
        } catch (e) {
          console.error("Failed to parse conversation event", e);
        }
      };

      es.onerror = () => {
        es.close();
        titleSourceRef.current.delete(conversationId);
      };

      titleSourceRef.current.set(conversationId, es);
    } catch (e) {
      console.error("Failed to start conversation event stream", e);
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
                    const channel = getChatEventChannel(ev);
                    if (channel === 'final' && ev.type === 'content') return { type: 'content' as const, channel: 'final' as const, content: ev.content || '' };
                    if (ev.type === 'reasoning') {
                      return {
                        type: 'reasoning' as const,
                        channel: 'analysis' as const,
                        content: ev.content || '',
                        durationSeconds: typeof ev.duration_seconds === 'number' ? ev.duration_seconds : undefined,
                      };
                    }
                    return normalizeStreamingToolEvent(ev);
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
       let messagePersisted = false;
       if (!startedTaskId) {
         try {
           const refreshed = await api.getConversation(conversationId, workspaceId);
           const previousMessageCount = previousConversation?.messages.length ?? 0;
           messagePersisted = refreshed.messages.length > previousMessageCount;
           setActiveConversation(refreshed);
           setConversations(prev => prev.map(c => c.id === refreshed.id ? refreshed : c));
           syncConversationActiveTaskId(conversationId, refreshed.active_task_id ?? null);
         } catch (refreshErr) {
           console.error('Failed to refresh conversation after task start error:', refreshErr);
         }
       }
       if (!startedTaskId && !messagePersisted && previousConversation) {
         setActiveConversation(previousConversation);
         setConversations(prev => prev.map(c => c.id === conversationId ? previousConversation : c));
         syncConversationActiveTaskId(conversationId, previousConversation.active_task_id ?? null);
       }
       clearActiveStreamingUi();
       if (messagePersisted && err && typeof err === 'object') {
         (err as { messagePersisted?: boolean }).messagePersisted = true;
       }
       throw err;
    }
  }, [activeConversation, clearActiveStreamingUi, connectTaskStream, syncConversationActiveTaskId, workspaceId]);

  // Keep task streaming in sync when workspace aggregate state sets activeTask.
  useEffect(() => {
    if (!activeTask) return;
    if (activeTask.status !== 'pending' && activeTask.status !== 'running') return;
    void connectTaskStream(activeTask.id);
  }, [activeTask?.id, activeTask?.status, connectTaskStream]);

  // Expose connectTaskStream via ref so the conversation event SSE handler
  // (declared earlier in the component) can trigger task streaming the moment
  // a task_started event is received.
  useEffect(() => {
    connectTaskStreamRef.current = (taskId: string) => { void connectTaskStream(taskId); };
    return () => { connectTaskStreamRef.current = null; };
  }, [connectTaskStream]);

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
    // Conversation-scoped SSE (`/conversations/{id}/events`) drives immediate
    // task pickup via `task_started` events; this interval is a long
    // safety-net fallback in case the SSE connection is briefly dropped.
    const interval = setInterval(() => {
      void checkTasks();
    }, 30000);

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
      setEditingHeaderTitle(null);
      return;
    }

    try {
      const updated = await api.updateConversationTitle(conversationId, titleInput.trim(), workspaceId);
      setConversations(prev => prev.map(c => c.id === conversationId ? { ...c, title: updated.title, updated_at: updated.updated_at } : c));
      if (activeConversation?.id === conversationId) {
        setActiveConversation(prev => prev?.id === conversationId ? { ...prev, title: updated.title, updated_at: updated.updated_at } : prev);
      }
      setEditingTitle(null);
      setEditingHeaderTitle(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update title');
    }
  };

  const changeModel = async (newModel: string) => {
    if (!activeConversation || isStreaming) return;

    try {
      const selection = resolveConversationModelSelection(
        newModel,
        availableModels,
      );
      const requestedModelId = selection.modelId || newModel.trim();
      const requestedProvider = selection.explicitProvider || selection.inferredProvider;
      const requestedProviderForApi = requestedProvider && KNOWN_PROVIDER_KEYS.has(requestedProvider)
        ? (requestedProvider as LlmProviderWire)
        : undefined;
      const selected = selection.matchedModel;

      const updated = await api.updateConversationModel(
        activeConversation.id,
        selected?.id || requestedModelId,
        workspaceId,
        selected?.provider || requestedProviderForApi,
      );
      setActiveConversation(prev => prev?.id === updated.id ? { ...prev, model: updated.model, updated_at: updated.updated_at } : prev);
      setConversations(prev => prev.map(c => c.id === updated.id ? { ...c, model: updated.model, updated_at: updated.updated_at } : c));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to change model');
    }
  };

  const startFreshConversation = async () => {
    if (isReadOnly || isCreatingFreshConversation) return;
    // Start a fresh conversation and detach from any currently streaming one.
    shouldAutoScrollRef.current = true;
    try {
      setIsCreatingFreshConversation(true);
      setIsConversationSwitchLoading(true);
      clearActiveStreamingUi();
      const conversation = await api.createConversation(undefined, workspaceId);
      applyCreatedConversation(conversation);
      setInterruptedTask(null);
      setHitMaxIterations(false);
      setIsConnectionError(false);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create conversation');
    } finally {
      setIsCreatingFreshConversation(false);
      setIsConversationSwitchLoading(false);
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

  const isModelsLoading = Boolean(
    modelsLoading || modelsReadiness?.models_loading || modelsReadiness?.copilot_refresh_in_progress,
  );
  const getBranchSendBlockReason = useCallback(async (
    branchKind: ConversationBranchKind,
    conversation: Conversation | null,
  ): Promise<string | null> => {
    if (!branchStartsGeneration(branchKind) || !conversation) {
      return null;
    }

    const modelState = isModelsLoading
      ? await awaitModelsReady()
      : { models: availableModels, meta: modelsMeta };
    const allowedModels = modelState.meta?.allowed_models ?? [];
    if (!allowedModels.length) {
      return null;
    }

    const selection = resolveConversationModelSelection(
      conversation.model || '',
      modelState.models,
    );
    if (!selection.modelId) {
      return null;
    }

    const scopedIdentifier = resolvedModelSelectionKey(selection);

    if (!modelIdentifierInList(scopedIdentifier, allowedModels)) {
      return MODEL_REMOVED_FROM_CHAT_MODELS_MESSAGE;
    }

    return null;
  }, [availableModels, awaitModelsReady, isModelsLoading, modelsMeta]);

  // Direct message send - bypasses inputValue state for programmatic sending
  const sendMessageDirect = async (message: string) => {
    if (!message.trim() || !activeConversation || isStreaming || isReadOnly) return;
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
      return true;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message';

      // Check for connection errors
      const isConnError = errorMessage.toLowerCase().includes('502') ||
                          errorMessage.toLowerCase().includes('503') ||
                          errorMessage.toLowerCase().includes('connection') ||
                          errorMessage.toLowerCase().includes('network') ||
                          errorMessage.toLowerCase().includes('fetch');
      const messagePersisted = Boolean(
        err
        && typeof err === 'object'
        && 'messagePersisted' in err
        && (err as { messagePersisted?: boolean }).messagePersisted,
      );

      setIsConnectionError(!messagePersisted && isConnError);
      setError(messagePersisted ? null : errorMessage);

      setIsStreaming(false);
      setStreamingContent('');
      setStreamingEvents([]);
      return messagePersisted;
    }
  };

  const sendMessage = async () => {
    if ((!inputValue.trim() && attachments.length === 0) || !activeConversation || isStreaming || isReadOnly) return;

    const userMessage = inputValue.trim();
    const messageAttachments = [...attachments];

    setInputValue('');
    setAttachments([]);

    // Auto-collapse sidebar when user starts chatting
    setShowSidebar(false);

    // Convert attachments to content parts if present
    if (messageAttachments.length > 0) {
      const contentParts = attachmentsToContentParts(userMessage, messageAttachments);
      const sent = await sendMessageDirect(JSON.stringify(contentParts));
      if (!sent) {
        setInputValue(userMessage);
        setAttachments(messageAttachments);
      }
    } else {
      const sent = await sendMessageDirect(userMessage);
      if (!sent) {
        setInputValue(userMessage);
      }
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
          size: (p as any).size_bytes || 0,
          mimeType: (p as any).mime_type || 'application/octet-stream',
          filePath: (p as any).file_path,
          attachmentId: (p as any).attachment_id,
          attachmentSource: (p as any).attachment_source,
          expiresAt: (p as any).expires_at,
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
    branchKind: ConversationBranchKind,
  ) => {
    if (branchPointIndex < 0 || branchPointIndex >= messageCount) {
      return null;
    }

    const blockReason = await getBranchSendBlockReason(branchKind, activeConversation);
    if (blockReason) {
      throw new Error(blockReason);
    }

    const createdBranch = await api.createConversationBranch(
      conversationId,
      {
        from_message_index: branchPointIndex,
        branch_kind: branchKind,
        auto_snapshot: Boolean(workspaceId),
      },
      workspaceId,
    );

    // The backend always clears active_branch_id when a new branch is
    // created (the conversation moves to the live path). Mirror that in
    // local state immediately so the branch nav lineage walk does not
    // still resolve to the prior branch and show e.g. "5/11" when the
    // user is actually on the new "11/11" branch. This applies uniformly
    // to edit, replay, and delete flows.
    setActiveConversation((prev) => (
      prev && prev.id === conversationId && prev.active_branch_id !== null
        ? { ...prev, active_branch_id: null }
        : prev
    ));
    setConversations((prev) => prev.map((c) => (
      c.id === conversationId && c.active_branch_id !== null
        ? { ...c, active_branch_id: null }
        : c
    )));

    if (createdBranch.parent_branch_id) {
      const anchorIndex = getConversationBranchAnchorIndex(
        branchPointIndex,
        createdBranch.branch_kind ?? branchKind,
        messageCount,
      );
      const selectionKey = getConversationBranchSelectionKey(branchPointIndex, anchorIndex);
      setBranchSelections((prev) => ({ ...prev, [selectionKey]: createdBranch.parent_branch_id! }));
    }

    // Inject the newly created branch into UI state using the actual server
    // response from the POST request. This avoids waiting for a secondary
    // GET request (refreshBranchPoints) to finish before the UI updates,
    // ensuring the X/N nav count bumps instantaneously before the streaming
    // task even starts.
    //
    // Also mirror the backend's sibling-absorption: when a new branch is
    // created at bp=K, the server deletes sibling branches at bp>K that
    // share the same parentBranchId (see repository.create_conversation_branch).
    // Dropping those siblings client-side makes the injected count match the
    // final server state, avoiding a visible flicker when the refresh GET
    // lands.
    setBranchPoints((prev) => {
      const parentBranchId = createdBranch.parent_branch_id ?? null;
      // 1) Remove entries whose branches are fully absorbed siblings.
      const filtered = prev
        .map((p) => {
          if (p.branch_point_index <= branchPointIndex) return p;
          const kept = p.branches.filter(
            (b) => (b.parent_branch_id ?? null) !== parentBranchId || b.id === createdBranch.id,
          );
          if (kept.length === p.branches.length) return p;
          return { ...p, branches: kept };
        })
        .filter((p) => p.branches.length > 0);
      // 2) Insert/merge the newly created branch at its own bp.
      const existingIdx = filtered.findIndex((p) => p.branch_point_index === branchPointIndex);
      if (existingIdx >= 0) {
        const next = [...filtered];
        next[existingIdx] = {
          ...next[existingIdx],
          branches: [...next[existingIdx].branches, createdBranch],
        };
        return next;
      }
      return [
        ...filtered,
        {
          branch_point_index: branchPointIndex,
          branches: [createdBranch],
          active_branch_id: null,
        },
      ];
    });

    // Refresh branch points immediately so the X/N nav reflects the new
    // branch on the very next render. Fire-and-forget: this drives only
    // the visual count and must not block the caller's optimistic UI
    // update (message truncation, snapshot restore, etc.).
    void refreshBranchPoints(conversationId);

    // Branch creation with auto_snapshot may have created a new userspace
    // snapshot. Notify parent so the snapshots panel can refresh.
    if (workspaceId) {
      try {
        onSnapshotsMaybeChanged?.();
      } catch (notifyErr) {
        console.warn('onSnapshotsMaybeChanged threw:', notifyErr);
      }
    }

    return createdBranch;
  }, [activeConversation, workspaceId, onSnapshotsMaybeChanged, refreshBranchPoints, getBranchSendBlockReason]);

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

  const getDeleteBranchPointIndex = useCallback((messageIdx: number): number => {
    return Math.max(messageIdx, 0);
  }, []);

  const switchBranch = useCallback(async (branchId: string) => {
    if (!activeConversation || branchSwitching) return;
    const conversationId = activeConversation.id;
    const isLiveTarget = branchId.startsWith('__current__:');
    setBranchSwitching(true);
    try {
      // Remember the old active branch's position before switching
      const oldActiveBranchId = activeConversation.active_branch_id;
      if (oldActiveBranchId) {
        const oldBranch = branchesById.get(oldActiveBranchId);
        if (oldBranch) {
          const anchorIndex = getConversationBranchAnchorIndex(
            oldBranch.branch_point_index,
            oldBranch.branch_kind,
            activeConversation.messages.length,
          );
          const selectionKey = getConversationBranchSelectionKey(oldBranch.branch_point_index, anchorIndex);
          setBranchSelections(prev => ({ ...prev, [selectionKey]: oldActiveBranchId }));
        }
      }

      let updated: Conversation;
      if (isLiveTarget) {
        // No active branch to release → UI is already on live; no-op.
        if (!oldActiveBranchId) {
          setBranchSwitching(false);
          return;
        }
        updated = await api.releaseConversationBranch(conversationId, workspaceId);
      } else {
        updated = await api.switchConversationBranch(conversationId, branchId, workspaceId);
      }
      setActiveConversation(updated);
      setConversations(prev => prev.map(c => c.id === updated.id ? updated : c));
      const refreshedPoints = await refreshBranchPoints(conversationId);

      // Notify parent (UserSpacePanel) about the branch switch with associated snapshot
      if (onBranchSwitch) {
        if (isLiveTarget) {
          onBranchSwitch(null, null);
        } else {
          const allBranches = refreshedPoints.flatMap(bp => bp.branches);
          const targetBranch = allBranches.find(b => b.id === branchId);
          if (targetBranch && !targetBranch.branch_kind) {
            onBranchSwitch(null, null);
          } else {
            onBranchSwitch(branchId, targetBranch?.associated_snapshot_id ?? null);
          }
        }
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
    const previousConversation = activeConversation;
    let replayBranch: ConversationBranchSummary | null = null;
    try {
      replayBranch = await createBranchForMessageMutation(
        conversationId,
        truncateAt,
        activeConversation.messages.length,
        'replay',
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
      // The backend cleared active_branch_id when it created the new branch
      // (we are now on the live path). Mirror that here so the branch nav
      // lineage walk does not still resolve to the prior branch and show
      // e.g. "5/11" when the user is actually on the new "11/11" branch.
      const optimisticConv: Conversation = {
        ...activeConversation,
        messages: [...messagesToKeep, optimisticMsg],
        active_branch_id: null,
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
      setIsStreaming(false);
      setStreamingContent('');
      setStreamingEvents([]);
      if (replayBranch) {
        try {
          const restored = await api.switchConversationBranch(conversationId, replayBranch.id, workspaceId);
          setActiveConversation(restored);
          setConversations(prev => prev.map(c => c.id === restored.id ? restored : c));
          syncConversationActiveTaskId(conversationId, restored.active_task_id ?? null);
          void refreshBranchPoints(conversationId);
          return;
        } catch (restoreErr) {
          console.error('Failed to restore replay branch after replay error:', restoreErr);
        }
      }
      setActiveConversation(previousConversation);
      setConversations(prev => prev.map(c => c.id === previousConversation.id ? previousConversation : c));
      syncConversationActiveTaskId(conversationId, previousConversation.active_task_id ?? null);
    }
  }, [activeConversation, isStreaming, isReadOnly, createBranchForMessageMutation, findUserMessageIndexAtOrBefore, workspaceId, refreshBranchPoints, connectTaskStream, syncConversationActiveTaskId]);

  const deleteFromMessage = useCallback(async (messageIdx: number) => {
    if (!activeConversation || isStreaming || isReadOnly) return;
    const conversationId = activeConversation.id;
    const selectedMessage = activeConversation.messages[messageIdx];
    if (!selectedMessage) return;
    const selectedMessageId = selectedMessage.message_id;

    // Anchor the branch at the deleted message itself so the backend
    // preserves the truncated tail and truncates the live path to
    // [:messageIdx]. The UI surfaces the restore nav on that same row.
    const branchPointIndex = getDeleteBranchPointIndex(messageIdx);
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
        'delete',
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

    let createdBranch = false;

    try {
      // 1. Create a branch to preserve the original messages
      await createBranchForMessageMutation(
        conversationId,
        truncateAt,
        activeConversation.messages.length,
        'edit',
      );
      createdBranch = true;

      // Clear the edit state after branch creation succeeds.
      setEditingMessageIdx(null);
      setEditMessageContent('');
      setEditMessageAttachments([]);
      setError(null);

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

      // The backend cleared active_branch_id when it created the new branch
      // (we are now on the live path). Mirror that here so the branch nav
      // lineage walk does not still resolve to the prior branch and show
      // e.g. "5/11" when the user is actually on the new "11/11" branch.
      const optimisticConv: Conversation = {
        ...activeConversation,
        messages: [...messagesToKeep, optimisticMsg],
        active_branch_id: null,
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

      if (createdBranch) {
        // Restore authoritative state from server on error after branch mutation.
        try {
          const refreshed = await api.getConversation(conversationId, workspaceId);
          setActiveConversation(refreshed);
          setConversations(prev => prev.map(c => c.id === refreshed.id ? refreshed : c));
        } catch (refreshErr) {
          console.error('Failed to refresh conversation after edit error:', refreshErr);
        }
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
  const workspaceArchivedConversationOptions = workspaceArchivedConversations.filter((conv) => conv.title !== 'Untitled Chat');
  const archivedConversationDisplayCount = archivedConversationCount ?? (archiveLoaded ? archivedConversations.length : null);
  const shareableConversationUsers = useMemo(() => {
    if (!conversationShareableUserIds || conversationShareableUserIds.length === 0) {
      return memberPickerUsers;
    }
    const allowedIds = new Set(conversationShareableUserIds);
    if (conversationOwnerId) {
      allowedIds.add(conversationOwnerId);
    }
    return memberPickerUsers.filter((user) => allowedIds.has(user.id));
  }, [memberPickerUsers, conversationOwnerId, conversationShareableUserIds]);
  const showInlineToolSelector = canUseConversationTools;

  const renderConversationItem = (conv: Conversation, options?: { searchQuery?: string; onClickOverride?: () => void }) => {
    const searchQuery = options?.searchQuery ?? '';
    const messageCount = getConversationMessageCount(conv);
    const metaMessageCount = `${messageCount} msg${messageCount === 1 ? '' : 's'}`;
    const metaTimestamp = formatChatTimestamp(conv.updated_at);
    const isActive = activeConversation?.id === conv.id;
    const snippet = searchQuery.trim() ? buildConversationSnippet(conv, searchQuery) : null;
    const handleItemClick = options?.onClickOverride ?? (() => selectConversation(conv));

    return (
      <div
        key={conv.id}
        className={`chat-conversation-item ${isActive ? 'active' : ''}${editingTitle === conv.id ? ' is-renaming' : ''}`}
        onClick={handleItemClick}
      >
        {editingTitle === conv.id ? (
          <textarea
            ref={(el) => { if (el) { el.style.height = 'auto'; el.style.height = `${el.scrollHeight}px`; } }}
            value={titleInput}
            onChange={(e) => {
              setTitleInput(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = `${e.target.scrollHeight}px`;
            }}
            onBlur={() => saveTitle(conv.id)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveTitle(conv.id); }
              if (e.key === 'Escape') setEditingTitle(null);
            }}
            onClick={(e) => e.stopPropagation()}
            autoFocus
            rows={1}
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
              {searchQuery.trim()
                ? <HighlightedText text={conv.title || 'Untitled Chat'} query={searchQuery} />
                : <ChatTitle title={conv.title} />
              }
            </div>
            <div className="chat-conversation-meta">
              <span className="chat-meta-count">{metaMessageCount}</span>
              <span className="chat-meta-time">{metaTimestamp}</span>
            </div>
            {snippet && (
              <div className="chat-conversation-snippet" title={snippet}>
                <HighlightedText text={snippet} query={searchQuery} />
              </div>
            )}
          </>
        )}
        {editingTitle !== conv.id && (
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
        )}
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
        {!workspaceId && (
          <div className="chat-conversation-search">
            <input
              type="text"
              className="chat-conversation-search-input"
              placeholder="Search chats..."
              value={conversationSearchQuery}
              onChange={(e) => setConversationSearchQuery(e.target.value)}
              aria-label="Search conversations by title or content"
            />
            {conversationSearchQuery && (
              <button
                type="button"
                className="chat-conversation-search-clear"
                onClick={() => setConversationSearchQuery('')}
                title="Clear search"
                aria-label="Clear search"
              >
                <X size={12} />
              </button>
            )}
            {(archiveLoading && conversationSearchQuery) || conversations.some(c => c.active_task_id) ? (
              <span className="chat-conversation-search-spinner" title={archiveLoading && conversationSearchQuery ? 'Loading older chats' : 'Processing in background'}>
                <MiniLoadingSpinner variant="icon" size={12} />
              </span>
            ) : null}
          </div>
        )}

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
          ) : (() => {
            const trimmedQuery = conversationSearchQuery.trim();
            // While searching, fold archived hits into the visible list so
            // matches across older chats surface inline.
            const baseConversations = trimmedQuery && !workspaceId
              ? [
                  ...conversations,
                  ...archivedConversations.filter((archived) => !conversations.some((c) => c.id === archived.id)),
                ]
              : conversations;
            const filteredConversations = trimmedQuery
              ? baseConversations.filter((c) => conversationMatchesQuery(c, trimmedQuery))
              : baseConversations;

            if (filteredConversations.length === 0) {
              if (trimmedQuery) {
                return (
                  <div className="chat-empty-state chat-empty-state-search">
                    <p>No chats match "{trimmedQuery}".</p>
                    {!workspaceId && !archiveLoaded && !archiveLoading && (
                      <button className="btn btn-secondary btn-sm" onClick={() => void loadArchivedConversations()}>
                        Search older chats
                      </button>
                    )}
                  </div>
                );
              }
              return (
                <div className="chat-empty-state">
                  <p>No conversations yet</p>
                  <button className="btn" onClick={createNewConversation}>
                    Start a conversation
                  </button>
                </div>
              );
            }

            const renderItem = (conv: Conversation) => renderConversationItem(conv, { searchQuery: trimmedQuery });

            if (isAdmin) {
              // Re-group filtered list so admin grouping still works while searching.
              const groups = new Map<string, { key: string; label: string; conversations: Conversation[]; isCurrentUserGroup: boolean }>();
              for (const conv of filteredConversations) {
                const key = getOwnerKey(conv);
                const label = getOwnerLabel(conv);
                const existing = groups.get(key);
                if (existing) {
                  existing.conversations.push(conv);
                  existing.isCurrentUserGroup = existing.isCurrentUserGroup || conv.user_id === currentUser.id;
                } else {
                  groups.set(key, {
                    key,
                    label,
                    conversations: [conv],
                    isCurrentUserGroup: conv.user_id === currentUser.id,
                  });
                }
              }
              const groupList = Array.from(groups.values()).sort((a, b) => {
                if (a.isCurrentUserGroup !== b.isCurrentUserGroup) return a.isCurrentUserGroup ? -1 : 1;
                return a.label.localeCompare(b.label);
              });
              return groupList.map((group) => {
                // When searching, expand groups so matches are visible.
                const isCollapsed = trimmedQuery ? false : (collapsedGroups[group.key] ?? !group.isCurrentUserGroup);
                return (
                  <div key={group.key} className="chat-conversation-group">
                    <button className="chat-group-header" onClick={() => toggleGroup(group.key)}>
                      <span className="chat-group-name">{group.label}</span>
                      <span className="chat-group-count">{group.conversations.length}</span>
                      <span className="chat-group-toggle">{isCollapsed ? '▶' : '▼'}</span>
                    </button>
                    {!isCollapsed && (
                      <div className="chat-group-list">
                        {group.conversations.map(renderItem)}
                      </div>
                    )}
                  </div>
                );
              });
            }
            return filteredConversations.map(renderItem);
          })()}
        </div>

        {!workspaceId && !isConversationListLoading && (
          <div className="chat-sidebar-footer">
            <button
              type="button"
              className="chat-show-older-btn"
              onClick={() => {
                setShowArchiveModal(true);
                setArchiveSearchQuery('');
                if (!archiveLoaded && !archiveLoading) void loadArchivedConversations();
              }}
              title={`Show chats older than ${getArchiveAgeLabel(archiveAgeDays)}`}
            >
              <Clock size={13} aria-hidden="true" />
              <span>Show Older</span>
              {archivedConversationDisplayCount !== null && archivedConversationDisplayCount > 0 && (
                <span className="chat-show-older-count">{archivedConversationDisplayCount}</span>
              )}
            </button>
          </div>
        )}
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
                          onPointerDown={(e) => {
                            e.stopPropagation();
                          }}
                          onMouseDown={(e) => {
                            e.stopPropagation();
                          }}
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
                        {workspaceConversationOptions.length > 1 && (
                        <div className="model-selector-search">
                          <input
                            ref={workspaceConversationSearchInputRef}
                            type="text"
                            className="model-selector-search-input"
                            placeholder="Search chats..."
                            value={workspaceConversationSearchQuery}
                            onChange={(e) => setWorkspaceConversationSearchQuery(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Escape') {
                                e.preventDefault();
                                if (workspaceConversationSearchQuery) {
                                  setWorkspaceConversationSearchQuery('');
                                } else {
                                  setIsWorkspaceConversationMenuOpen(false);
                                }
                              }
                            }}
                            aria-label="Filter workspace chats"
                          />
                          {workspaceConversationSearchQuery && (
                            <button
                              type="button"
                              className="model-selector-search-clear"
                              onClick={() => {
                                setWorkspaceConversationSearchQuery('');
                                workspaceConversationSearchInputRef.current?.focus();
                              }}
                              title="Clear search"
                              aria-label="Clear search"
                            >
                              <X size={12} />
                            </button>
                          )}
                        </div>
                        )}
                        <div className="model-selector-dropdown-inner" role="listbox" aria-label="Workspace chats">
                          {(() => {
                            const needle = workspaceConversationSearchQuery.trim().toLowerCase();
                            const hydratedWorkspaceConversationsById = new Map(
                              workspaceArchivedConversationOptions.map((conversation) => [conversation.id, conversation]),
                            );
                            const searchableWorkspaceConversationOptions = workspaceConversationOptions.map((conversation) => {
                              const hydrated = hydratedWorkspaceConversationsById.get(conversation.id);
                              return hydrated && hydrated.messages.length > conversation.messages.length
                                ? hydrated
                                : conversation;
                            });
                            const searchBase = needle
                              ? [
                                  ...searchableWorkspaceConversationOptions,
                                  ...workspaceArchivedConversationOptions.filter(
                                    (archived) => !workspaceConversationOptions.some((c) => c.id === archived.id),
                                  ),
                                ]
                              : workspaceConversationOptions;
                            const filtered = needle
                              ? searchBase.filter((c) => conversationMatchesQuery(c, needle))
                              : searchBase;
                            if (filtered.length === 0) {
                              return (
                                <div className="model-selector-empty">
                                  {needle ? `No chats match "${workspaceConversationSearchQuery.trim()}"` : 'No chats yet'}
                                </div>
                              );
                            }
                            return filtered.map((conversation) => {
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
                                  if (isEditing) {
                                    return;
                                  }
                                  if (event.key === 'Enter' || event.key === ' ') {
                                    event.preventDefault();
                                    setIsWorkspaceConversationMenuOpen(false);
                                    void selectConversation(conversation);
                                  }
                                }}
                              >
                                <div className="chat-workspace-conversation-content">
                                  {isEditing ? (
                                    <textarea
                                      ref={(el) => { if (el) { el.style.height = 'auto'; el.style.height = `${el.scrollHeight}px`; } }}
                                      value={titleInput}
                                      onChange={(e) => {
                                        setTitleInput(e.target.value);
                                        e.target.style.height = 'auto';
                                        e.target.style.height = `${e.target.scrollHeight}px`;
                                      }}
                                      onBlur={() => void saveTitle(conversation.id)}
                                      onKeyDown={(e) => {
                                        e.stopPropagation();
                                        if (e.key === 'Enter' && !e.shiftKey) {
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
                                      rows={1}
                                      className="chat-title-input chat-workspace-conversation-title-input"
                                    />
                                  ) : (
                                    <>
                                      <span className="model-selector-item-name chat-workspace-conversation-item-name">
                                        {workspaceConversationSearchQuery.trim()
                                          ? <HighlightedText text={conversation.title || 'Untitled Chat'} query={workspaceConversationSearchQuery} />
                                          : (conversation.title || 'Untitled Chat')}
                                      </span>
                                      {(() => {
                                        const snippet = workspaceConversationSearchQuery.trim()
                                          ? buildConversationSnippet(conversation, workspaceConversationSearchQuery)
                                          : null;
                                        if (!snippet) return null;
                                        return (
                                          <div className="chat-conversation-snippet chat-workspace-conversation-snippet" title={snippet}>
                                            <HighlightedText text={snippet} query={workspaceConversationSearchQuery} />
                                          </div>
                                        );
                                      })()}
                                    </>
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
                          });
                          })()}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className={`chat-header-title-row${editingHeaderTitle === activeConversation.id ? ' is-editing' : ''}`}>
                    {editingHeaderTitle === activeConversation.id ? (
                      <textarea
                        ref={(el) => { if (el) { el.style.height = 'auto'; el.style.height = `${el.scrollHeight}px`; } }}
                        value={titleInput}
                        onChange={(e) => {
                          setTitleInput(e.target.value);
                          e.target.style.height = 'auto';
                          e.target.style.height = `${e.target.scrollHeight}px`;
                        }}
                        onBlur={() => saveTitle(activeConversation.id)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            void saveTitle(activeConversation.id);
                          }
                          if (e.key === 'Escape') setEditingHeaderTitle(null);
                        }}
                        autoFocus
                        rows={1}
                        className="chat-title-input chat-header-title-input"
                      />
                    ) : (
                      <>
                        <h2>{activeConversation.title}</h2>
                        <button
                          className="chat-header-title-edit-btn"
                          onClick={() => {
                            setEditingHeaderTitle(activeConversation.id);
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
                  loading={isModelsLoading}
                />
                {canShareConversation && !embedded && activeConversation && (
                  <button
                    className="btn btn-secondary btn-sm btn-icon"
                    onClick={onOpenShareModal}
                    title="Share conversation"
                    aria-label="Share conversation"
                    type="button"
                  >
                    <Link size={14} />
                  </button>
                )}
                {!embedded && canManageConversationMembers && (
                  <MemberManagementButton
                    className="btn btn-secondary btn-sm btn-icon"
                    onClick={handleOpenMembersModal}
                    title="Manage conversation members"
                  />
                )}
                {canUseConversationTools && (
                  <ToolSelectorDropdown
                    availableTools={effectiveAvailableTools}
                    selectedToolIds={resolvedConversationToolIdSet}
                    onToggleTool={handleToggleConversationTool}
                    builtInTools={conversationBuiltInTools}
                    selectedBuiltInToolIds={selectedConversationBuiltInToolIdSet}
                    onToggleBuiltInTool={handleToggleConversationBuiltInTool}
                    selectedToolGroupIds={conversationToolGroupIdSet}
                    onToggleToolGroup={handleToggleConversationToolGroup}
                    toolGroups={effectiveToolGroups}
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
                    const selection = resolveConversationModelSelection(
                      activeConversation.model || '',
                      availableModels,
                    );
                    return selection.matchedModel
                      ? toProviderScopedModelKey(selection.matchedModel.provider, selection.matchedModel.id)
                      : selection.modelId;
                  })()}
                  onModelChange={changeModel}
                  getModelSelectionKey={(model) => toProviderScopedModelKey(model.provider, model.id)}
                  disabled={isStreaming || isModelsLoading}
                  loading={isModelsLoading}
                  triggerIcon={showWorkspaceConversationSelect ? <Bot size={14} /> : undefined}
                  triggerClassName={showWorkspaceConversationSelect ? 'chat-workspace-model-trigger' : undefined}
                />
                <button
                  className="btn btn-sm btn-secondary chat-new-chat-btn"
                  onClick={startFreshConversation}
                  title={isCreatingFreshConversation ? 'Creating a new conversation...' : 'Start a new conversation'}
                  disabled={isReadOnly || isCreatingFreshConversation}
                >
                  {isCreatingFreshConversation ? (
                    <>
                      <MiniLoadingSpinner variant="icon" size={14} className="chat-new-chat-spinner" ariaHidden />
                      <span className="chat-new-chat-label">Creating...</span>
                    </>
                  ) : (
                    <>
                      <MessageSquarePlus size={14} className="chat-new-chat-icon" aria-hidden="true" />
                      <span className="chat-new-chat-label">New Chat</span>
                    </>
                  )}
                </button>
                {!embedded && (
                  <button
                    className="btn btn-secondary btn-sm btn-icon"
                    onClick={toggleFullscreen}
                    title={isFullscreen ? 'Exit full screen' : 'Full screen'}
                  >
                    {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                  </button>
                )}
              </div>
            </div>

            {/* Messages */}
            {!isMessagesCollapsed && (
            <div className="chat-messages" ref={chatMessagesRef} onScroll={handleScroll}>
              {(isConversationSwitchLoading || isCreatingFreshConversation) ? (
                renderMessageBubbleSkeletons()
              ) : activeConversation.messages.length === 0 && !isStreaming ? (
                <div className="chat-welcome">
                  <h3>Start a conversation</h3>
                  <p>Ask questions about your indexed code, query databases, or get help with your systems.</p>
                </div>
              ) : (
                <>
                  {activeConversation.messages.map((msg, idx) => {
                    const branchGroups = branchGroupsByIndex.get(idx) ?? [];
                    const hasBranches = branchGroups.length > 0;
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
                                        parts={pendingReasoningParts.length > 0 ? pendingReasoningParts : undefined}
                                        durationSeconds={pendingReasoningDurationSeconds}
                                        showToolCalls={showToolCalls}
                                        workspaceId={workspaceId}
                                        conversationId={activeConversation.id}
                                        onOpenWorkspaceFile={onOpenWorkspaceFile}
                                      />
                                    );
                                    pendingReasoning = '';
                                    pendingReasoningParts = [];
                                    pendingReasoningDurationSeconds = undefined;
                                  };

                                  for (let evIdx = 0; evIdx < msg.events.length; evIdx++) {
                                    const ev = msg.events[evIdx];
                                    const channel = getChatEventChannel(ev);
                                    if (channel === 'analysis' && ev.type === 'reasoning') {
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
                                    } else if (channel === 'commentary' && ev.type === 'tool' && pendingReasoning && !isVisualizationToolName(ev.tool)) {
                                      pendingReasoningParts.push({
                                        type: 'tool',
                                        toolCall: {
                                          tool: ev.tool,
                                          input: ev.input,
                                          output: ev.output,
                                          presentation: ev.presentation,
                                          connection: ev.connection,
                                          status: 'complete' as const,
                                        },
                                      });
                                    } else {
                                      // Final content and visualization artifacts break reasoning adjacency.
                                      flushReasoning();
                                      if (channel === 'commentary' && ev.type === 'tool' && showToolCalls) {
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
                                              messageId={msg.message_id}
                                              messageIndex={idx}
                                              eventIndex={evIdx}
                                              onOpenWorkspaceFile={onOpenWorkspaceFile}
                                            />
                                          </div>
                                        );
                                      } else if (channel === 'final' && ev.type === 'content') {
                                        result.push(
                                          <div key={`event-${evIdx}`} className="chat-message-text markdown-content">
                                            <MemoizedMarkdown content={ev.content} />
                                          </div>
                                        );
                                      } else if (ev.type === 'error') {
                                        result.push(
                                          <div key={`event-${evIdx}`} className="chat-message-generation-error" role="status">
                                            <AlertCircle size={14} aria-hidden="true" />
                                            <span>Generation failed: {ev.content}</span>
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
                      const branchNav = hasBranches ? (
                        <span className="chat-branch-nav-stack">
                          {branchGroups.map((group) => {
                            const livePathOptionId = `__current__:${group.sourceBranchPointIndex}`;
                            const storedLivePathBranch = group.branches.find(b => !b.branch_kind) ?? null;
                            const hasLivePathOption = !storedLivePathBranch;
                            const allOptions = [
                              ...group.branches.map(b => ({ id: b.id, label: b.branch_kind ? (b.created_by_username || 'Branch') : 'Current' })),
                              ...(hasLivePathOption ? [{ id: livePathOptionId, label: 'Current' }] : []),
                            ];
                            const newestBranch = group.branches.length > 0 ? group.branches[group.branches.length - 1] : null;
                            const inferredCurrentBranchId = newestBranch?.parent_branch_id ?? null;
                            const branchIdsInGroup = new Set(group.branches.map(b => b.id));
                            let lineageBranchId: string | null = null;
                            if (activeBranchId) {
                              const visited = new Set<string>();
                              let curr: string | null = activeBranchId;
                              while (curr && !visited.has(curr)) {
                                visited.add(curr);
                                if (branchIdsInGroup.has(curr)) {
                                  lineageBranchId = curr;
                                  break;
                                }
                                const parent = branchesById.get(curr);
                                curr = parent?.parent_branch_id ?? null;
                              }
                            }

                            let matchIdx = lineageBranchId
                              ? allOptions.findIndex(o => o.id === lineageBranchId)
                              : -1;
                            if (matchIdx < 0 && !activeBranchId) {
                              if (hasLivePathOption) {
                                matchIdx = allOptions.findIndex(o => o.id === livePathOptionId);
                              } else if (storedLivePathBranch) {
                                // active_branch_id=null is the authoritative live path.
                                // A saved branch_kind=null row can be an older stashed
                                // Current path, so after edit/replay branch creation the
                                // live UI should snap to the newest option instead of
                                // reselecting that older stored row.
                                matchIdx = allOptions.length - 1;
                              }
                            }
                            if (matchIdx < 0 && branchSelections[group.selectionKey]) {
                              matchIdx = allOptions.findIndex(o => o.id === branchSelections[group.selectionKey]);
                            }
                            if (matchIdx < 0 && inferredCurrentBranchId) {
                              matchIdx = allOptions.findIndex(o => o.id === inferredCurrentBranchId);
                            }
                            const currentOptionIdx = matchIdx >= 0 ? matchIdx : allOptions.length - 1;

                            return (
                              <span key={group.selectionKey} className="chat-branch-nav">
                                <button className="chat-branch-nav-btn" onClick={() => { if (currentOptionIdx > 0 && !branchSwitching) switchBranch(allOptions[currentOptionIdx - 1].id); }} disabled={currentOptionIdx <= 0 || branchSwitching} aria-label="Previous branch">
                                  <ChevronLeft size={12} />
                                </button>
                                <span className="chat-branch-nav-label">{currentOptionIdx + 1}/{allOptions.length}</span>
                                <button className="chat-branch-nav-btn" onClick={() => { if (currentOptionIdx < allOptions.length - 1 && !branchSwitching) switchBranch(allOptions[currentOptionIdx + 1].id); }} disabled={currentOptionIdx >= allOptions.length - 1 || branchSwitching} aria-label="Next branch">
                                  <ChevronRight size={12} />
                                </button>
                              </span>
                            );
                          })}
                        </span>
                      ) : null;

                      const isCopied = copiedMessageIdx === idx;
                      // Only show the restore banner on the branch-point message for the active branch
                      const showBanner = inputBanner && activeBranchId && branchGroups.some(group => group.branches.some(b => b.id === activeBranchId));

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
                                    conversationId={activeConversation?.id}
                                    workspaceId={workspaceId}
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
                                {canShareConversation && !embedded && onShareConversationAtMessage && (
                                  <button
                                    className="chat-action-icon-btn"
                                    onClick={() => onShareConversationAtMessage(idx)}
                                    title="Share chat from this message"
                                    aria-label="Share chat from this message"
                                  >
                                    <Share2 size={12} />
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
                              {branchNav}
                              {branchNav && <span className="chat-message-actions-spacer" />}
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
                                {canShareConversation && !embedded && onShareConversationAtMessage && (
                                  <button
                                    className="chat-action-icon-btn"
                                    onClick={() => onShareConversationAtMessage(idx)}
                                    title="Share chat from this message"
                                    aria-label="Share chat from this message"
                                  >
                                    <Share2 size={12} />
                                  </button>
                                )}
                              </span>
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
                      title="Re-send"
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
                  conversationId={activeConversation?.id}
                  workspaceId={workspaceId}
                  disabled={isReadOnly || isStreaming}
                />
                <textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  placeholder={isReadOnly ? effectiveReadOnlyMessage : 'Ask a question or paste files/images (Ctrl+V)...'}
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
                        builtInTools={conversationBuiltInTools}
                        selectedBuiltInToolIds={selectedConversationBuiltInToolIdSet}
                        onToggleBuiltInTool={handleToggleConversationBuiltInTool}
                        selectedToolGroupIds={effectiveToolGroupIdSet}
                        onToggleToolGroup={handleToggleInlineToolGroup}
                        toolGroups={effectiveToolGroups}
                        openDirection="up"
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
                          builtInTools={conversationBuiltInTools}
                          selectedBuiltInToolIds={selectedConversationBuiltInToolIdSet}
                          onToggleBuiltInTool={handleToggleConversationBuiltInTool}
                          selectedToolGroupIds={effectiveToolGroupIdSet}
                          onToggleToolGroup={handleToggleInlineToolGroup}
                          toolGroups={effectiveToolGroups}
                          openDirection="up"
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
                          disabled={!activeConversation || !contextUsage.hasHeadroom}
                          title={contextUsage.hasHeadroom
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
        ) : isCreatingFreshConversation ? (
          renderFullChatSkeleton()
        ) : (
          <div className="chat-no-conversation">
            <h2>Welcome to Chat</h2>
            <p>Select a conversation or start a new one to begin.</p>
            <button
              className="btn"
              onClick={createNewConversation}
              disabled={isCreatingFreshConversation}
            >
              {isCreatingFreshConversation ? (
                <>
                  <MiniLoadingSpinner variant="icon" size={14} ariaHidden />
                  Creating...
                </>
              ) : (
                'New Conversation'
              )}
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

      {/* Archived Chats Modal */}
      {showArchiveModal && !embedded && !workspaceId && (
        <div
          className="modal-overlay chat-archive-modal-overlay"
          onClick={() => setShowArchiveModal(false)}
          onKeyDown={(e) => e.key === 'Escape' && setShowArchiveModal(false)}
        >
          <div className="modal chat-archive-modal" onClick={(e) => e.stopPropagation()}>
            <div className="chat-archive-modal-toolbar">
              <div className="chat-conversation-search chat-conversation-search-modal">
                <input
                  type="text"
                  className="chat-conversation-search-input"
                  placeholder="Search older chats..."
                  value={archiveSearchQuery}
                  onChange={(e) => setArchiveSearchQuery(e.target.value)}
                  autoFocus
                  aria-label="Search older conversations"
                />
                {archiveSearchQuery && (
                  <button
                    type="button"
                    className="chat-conversation-search-clear"
                    onClick={() => setArchiveSearchQuery('')}
                    title="Clear search"
                    aria-label="Clear search"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
              <div className="chat-archive-settings">
                <div className="chat-archive-settings-form">
                  <button
                    type="button"
                    className="btn btn-secondary btn-md"
                    onClick={cycleArchiveAgePreset}
                    title="Sidebar keeps recent chats visible and auto-hides older ones. Click to cycle archive window: 1d, 1wk, 1mo, 1yr."
                  >
                    {getArchiveAgeLabel(archiveAgeDays)}
                  </button>
                  <button className="modal-close" onClick={() => setShowArchiveModal(false)} title="Close" aria-label="Close">
                    &times;
                  </button>
                </div>
              </div>
            </div>
            <div className="modal-body chat-archive-modal-body" onScroll={handleArchiveModalScroll}>
              {archiveLoading && !archiveLoaded && archivedConversations.length === 0 ? (
                <div className="chat-empty-state" aria-live="polite">
                  <p>Loading older chats...</p>
                </div>
              ) : archiveError ? (
                <div className="chat-empty-state">
                  <p>{archiveError}</p>
                  <button className="btn btn-secondary btn-sm" onClick={() => void loadArchivedConversations()}>
                    Retry
                  </button>
                </div>
              ) : (() => {
                const trimmed = archiveSearchQuery.trim();
                const list = trimmed
                  ? archivedConversations.filter((c) => conversationMatchesQuery(c, trimmed))
                  : archivedConversations;
                if (list.length === 0) {
                  return (
                    <div className="chat-empty-state">
                      <p>{trimmed
                        ? `No older chats match "${trimmed}".`
                        : archiveAgeDays > 0
                          ? `No chats older than ${archiveAgeDays} day${archiveAgeDays === 1 ? '' : 's'}.`
                          : 'Auto-hide is off — all chats are already shown in the sidebar.'}</p>
                    </div>
                  );
                }
                return (
                  <div className="chat-conversation-list chat-conversation-list-non-admin chat-archive-conversation-list">
                    {list.map((conv) => renderConversationItem(conv, {
                      searchQuery: trimmed,
                      onClickOverride: () => {
                        setShowArchiveModal(false);
                        // Pull the older chat into the visible list so the
                        // chat panel can render it immediately without a
                        // refetch loop.
                        setConversations((prev) => prev.some((c) => c.id === conv.id) ? prev : [conv, ...prev]);
                        void selectConversation(conv);
                      },
                    }))}
                    {archiveLoading && archivedConversations.length > 0 && (
                      <div className="chat-empty-state" aria-live="polite">
                        <p>{trimmed ? 'Searching older chats...' : 'Loading more older chats...'}</p>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
