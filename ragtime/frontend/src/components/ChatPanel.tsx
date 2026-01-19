import { useState, useEffect, useRef, useCallback, useMemo, memo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Copy, Check, Loader2, Pencil, Trash2, Maximize2, X, AlertCircle, RefreshCw, FileText, Image as ImageIcon } from 'lucide-react';
import { api } from '@/api';
import type { Conversation, ChatMessage, AvailableModel, ChatTask, User, ContentPart } from '@/types';
import { FileAttachment, attachmentsToContentParts, type AttachmentFile } from './FileAttachment';

// Memoized markdown component to prevent re-parsing on every render
// Only re-renders when content actually changes
const MemoizedMarkdown = memo(function MemoizedMarkdown({ content }: { content: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>;
});

// Tool call info for display during streaming
interface ActiveToolCall {
  tool: string;
  input?: Record<string, unknown>;
  output?: string;
  status: 'running' | 'complete';
}

// Local render event to keep streaming items in arrival order
type StreamingRenderEvent =
  | { type: 'content'; content: string }
  | { type: 'tool'; toolCall: ActiveToolCall };

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

interface ChartData {
  __chart__: true;
  config: ChartConfig;
  description?: string;
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
                <td key={cellIdx}>{cell === null ? <span className="null-value">NULL</span> : String(cell)}</td>
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
        <p className="chart-description">{chartData.description}</p>
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

          // Initialize DataTable
          tableInstanceRef.current = $(tableEl).DataTable({
            ...tableData.config,
            destroy: true, // Allow re-initialization
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
                    <td key={cellIdx}>{cell === null ? 'NULL' : String(cell)}</td>
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

// Component to display a tool call with collapsible details
// Memoized to prevent re-renders when tool call data hasn't changed
interface ToolCallDisplayProps {
  toolCall: ActiveToolCall;
  defaultExpanded?: boolean;
  conversationId?: string;
  siblingEvents?: Array<{ type: string; tool?: string; output?: string }>;
  onRetrySuccess?: (newOutput: string) => void;
}

const ToolCallDisplay = memo(function ToolCallDisplay({
  toolCall,
  defaultExpanded = false,
  conversationId,
  siblingEvents,
  onRetrySuccess
}: ToolCallDisplayProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [copiedQuery, setCopiedQuery] = useState(false);
  const [copiedResult, setCopiedResult] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryOutput, setRetryOutput] = useState<string | null>(null);
  const [retryError, setRetryError] = useState<string | null>(null);

  // Check if this is a visualization tool that can be retried
  const isVisualizationTool = toolCall.tool === 'create_chart' || toolCall.tool === 'create_datatable';

  // Check if this tool call failed based on output content
  const hasErrorInOutput = useMemo(() => {
    const output = retryOutput || toolCall.output;
    if (!output) return false;
    const outputLower = output.toLowerCase();
    return outputLower.includes('error') ||
           outputLower.includes('validation error') ||
           outputLower.includes('failed') ||
           outputLower.includes('exception');
  }, [toolCall.output, retryOutput]);

  // Effective output (use retry output if available)
  const effectiveOutput = retryOutput || toolCall.output;

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

  // Determine the status icon
  const getStatusIcon = () => {
    if (toolCall.status === 'running') {
      return <Loader2 size={14} className="spinning" />;
    }
    if (isFailed && isVisualizationTool) {
      return <AlertCircle size={14} className="tool-call-error-icon" />;
    }
    return <Check size={14} />;
  };

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

  return (
    <div className={`tool-call tool-call-${toolCall.status}${isFailed ? ' tool-call-failed' : ''}`}>
      <div className="tool-call-header-row">
        <button
          className="tool-call-header"
          onClick={() => setExpanded(!expanded)}
        >
          <span className="tool-call-icon">
            {getStatusIcon()}
          </span>
          <span className="tool-call-name">{toolCall.tool}</span>
          <span className="tool-call-toggle">{expanded ? '▼' : '▶'}</span>
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
            <Loader2 size={12} className="spinning" />
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
          {inputDisplay && (
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
          {toolCall.output && (
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
          )}
        </div>
      )}
    </div>
  );
})

// Approximate tokens per character (rough estimate for display)
const CHARS_PER_TOKEN = 4;

// Consolidated streaming content - groups content events between tool calls
// This avoids re-rendering markdown for every token and dramatically improves performance
interface StreamingSegment {
  type: 'content' | 'tool';
  content?: string;  // For content segments - consolidated text
  toolCall?: ActiveToolCall;  // For tool segments
}

// Memoized component for rendering streaming segments efficiently
const StreamingSegmentDisplay = memo(function StreamingSegmentDisplay({
  segment,
  showToolCalls
}: {
  segment: StreamingSegment;
  showToolCalls: boolean;
}) {
  if (segment.type === 'tool' && segment.toolCall && showToolCalls) {
    return (
      <div className="chat-tool-calls">
        <ToolCallDisplay toolCall={segment.toolCall} defaultExpanded={false} />
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

// Estimate tokens from text
function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

// Estimate tokens from an object payload (e.g., tool inputs)
function estimateTokensFromObject(value: unknown): number {
  if (value === undefined || value === null) return 0;
  try {
    return estimateTokens(JSON.stringify(value));
  } catch {
    return estimateTokens(String(value));
  }
}

// Calculate tokens for a single message, including tool calls and chronological events
// If events exist, they contain the full picture (content + tools); otherwise fall back to content/tool_calls
function calculateMessageTokens(msg: ChatMessage): number {
  // If we have chronological events, use them (they contain content + tool calls)
  if (msg.events?.length) {
    let tokens = 0;
    for (const ev of msg.events) {
      if (ev.type === 'content') {
        tokens += estimateTokens(ev.content || '');
      } else if (ev.type === 'tool') {
        tokens += estimateTokensFromObject(ev.input);
        tokens += estimateTokens(ev.output || '');
      }
    }
    return tokens;
  }

  // Fallback: use content + legacy tool_calls
  let tokens = estimateTokens(msg.content || '');

  if (msg.tool_calls?.length) {
    for (const tc of msg.tool_calls) {
      tokens += estimateTokensFromObject(tc.input);
      tokens += estimateTokens(tc.output || '');
    }
  }

  return tokens;
}

// Calculate total tokens for a conversation
function calculateConversationTokens(messages: ChatMessage[]): number {
  return messages.reduce((total, msg) => total + calculateMessageTokens(msg), 0);
}

// Format relative time
function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

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

// Component to display message attachments
const MessageAttachments = memo(function MessageAttachments({ attachments }: { attachments: ContentPart[] }) {
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
                onClick={() => window.open(attachment.image_url.url, '_blank')}
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

interface ChatPanelProps {
  currentUser: User;
}

export function ChatPanel({ currentUser }: ChatPanelProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [inputValue, setInputValue] = useState('');
  const [attachments, setAttachments] = useState<AttachmentFile[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [streamingEvents, setStreamingEvents] = useState<StreamingRenderEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [editingTitle, setEditingTitle] = useState<string | null>(null);
  const [titleInput, setTitleInput] = useState('');
  const [editingMessageIdx, setEditingMessageIdx] = useState<number | null>(null);
  const [editMessageContent, setEditMessageContent] = useState('');
  const [hitMaxIterations, setHitMaxIterations] = useState(false);
  const [showToolCalls, setShowToolCalls] = useState(() => {
    const saved = localStorage.getItem('chat-show-tool-calls');
    return saved !== null ? saved === 'true' : true;
  });
  const [lastSentMessage, setLastSentMessage] = useState<string>('');
  const [isConnectionError, setIsConnectionError] = useState(false);
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});
  const isAdmin = currentUser.role === 'admin';

  // Inline confirmation for delete (conversation ID waiting for confirmation)
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Background task state
  const [activeTask, setActiveTask] = useState<ChatTask | null>(null);
  const [interruptedTask, setInterruptedTask] = useState<ChatTask | null>(null);  // Last interrupted task for continue
  const [_isPollingTask, setIsPollingTask] = useState(false);
  const taskPollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastSeenVersionRef = useRef<number>(0);  // Track last seen version for delta polling

  // Available models state
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = '0px';
      const scrollHeight = textarea.scrollHeight;
      textarea.style.height = scrollHeight + 'px';
    }
  }, [inputValue]);
  const abortControllerRef = useRef<AbortController | null>(null);

  const getOwnerKey = useCallback((conv: Conversation) => conv.username || conv.user_id || 'unknown', []);

  const getOwnerLabel = useCallback(
    (conv: Conversation) => conv.display_name || conv.username || 'Unknown user',
    [],
  );

  const groupedConversations = useMemo(() => {
    if (!isAdmin) return [] as Array<{ key: string; label: string; conversations: Conversation[] }>;

    const groups = conversations.reduce<Record<string, { label: string; conversations: Conversation[] }>>((acc, conv) => {
      const key = getOwnerKey(conv);
      const label = getOwnerLabel(conv);
      const existing = acc[key];
      acc[key] = existing
        ? { label: existing.label || label, conversations: [...existing.conversations, conv] }
        : { label, conversations: [conv] };
      return acc;
    }, {});

    return Object.entries(groups)
      .map(([key, value]) => ({ key, label: value.label, conversations: value.conversations }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [conversations, getOwnerKey, getOwnerLabel, isAdmin]);

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

  // Memoized consolidated segments for streaming - groups adjacent content events
  // This dramatically reduces re-renders by avoiding markdown parsing per-token
  const consolidatedSegments = useMemo((): StreamingSegment[] => {
    if (!streamingEvents.length) return [];

    const segments: StreamingSegment[] = [];
    let currentContent = '';

    for (const ev of streamingEvents) {
      if (ev.type === 'content') {
        // Accumulate content
        currentContent += ev.content;
      } else if (ev.type === 'tool') {
        // Flush any accumulated content first
        if (currentContent) {
          segments.push({ type: 'content', content: currentContent });
          currentContent = '';
        }
        // Add tool segment
        segments.push({ type: 'tool', toolCall: ev.toolCall });
      }
    }

    // Flush remaining content
    if (currentContent) {
      segments.push({ type: 'content', content: currentContent });
    }

    return segments;
  }, [streamingEvents]);

  // Save showToolCalls preference to localStorage
  useEffect(() => {
    localStorage.setItem('chat-show-tool-calls', showToolCalls.toString());
  }, [showToolCalls]);

  // Load conversations and available models on mount
  useEffect(() => {
    loadConversations();
    loadAvailableModels();
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeConversation?.messages, streamingContent]);

  // Focus input when conversation changes
  useEffect(() => {
    inputRef.current?.focus();
  }, [activeConversation?.id]);

  const loadAvailableModels = async () => {
    try {
      setModelsLoading(true);
      const data = await api.getAvailableModels();
      setAvailableModels(data.models);
    } catch (err) {
      console.error('Failed to load available models:', err);
    } finally {
      setModelsLoading(false);
    }
  };

  const loadConversations = async () => {
    try {
      const data = await api.listConversations();
      setConversations(data);
      // If no active conversation and we have conversations, select the most recent
      if (!activeConversation && data.length > 0) {
        setActiveConversation(data[0]);
      }
    } catch (err) {
      console.error('Failed to load conversations:', err);
    }
  };

  // Get context limit for a model from API-provided data (uses LiteLLM's dataset)
  const getContextLimit = useCallback((modelId: string): number => {
    const model = availableModels.find(m => m.id === modelId);
    return model?.context_limit ?? DEFAULT_CONTEXT_LIMIT;
  }, [availableModels]);

  const createNewConversation = async () => {
    try {
      const conversation = await api.createConversation();
      setConversations(prev => [conversation, ...prev]);
      setActiveConversation(conversation);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create conversation');
    }
  };

  // =========================================================================
  // Background Task Polling
  // =========================================================================

  const stopTaskPolling = useCallback(() => {
    if (taskPollIntervalRef.current) {
      clearInterval(taskPollIntervalRef.current);
      taskPollIntervalRef.current = null;
    }
    setIsPollingTask(false);
    lastSeenVersionRef.current = 0;  // Reset version tracking
  }, []);

  const pollTaskStatus = useCallback(async (taskId: string) => {
    try {
      // Use delta polling - pass last seen version to skip unchanged data
      const task = await api.getChatTask(taskId, lastSeenVersionRef.current);
      setActiveTask(task);

      // Update streaming display from task state
      // If streaming_state is null, server indicates no change since our version
      if (task.streaming_state) {
        // Update our tracked version
        lastSeenVersionRef.current = task.streaming_state.version;

        // Only update content if it actually changed (avoids React re-render)
        setStreamingContent(prev =>
          prev === task.streaming_state!.content ? prev : task.streaming_state!.content
        );
        // Convert task events to streaming render events
        // Use functional update to compare with previous state
        setStreamingEvents(prev => {
          const newEvents: StreamingRenderEvent[] = task.streaming_state!.events.map(ev => {
            if (ev.type === 'content') {
              return { type: 'content' as const, content: ev.content || '' };
            } else {
              return {
                type: 'tool' as const,
                toolCall: {
                  tool: ev.tool || '',
                  input: ev.input,
                  output: ev.output,
                  status: ev.output ? 'complete' as const : 'running' as const,
                }
              };
            }
          });
          // Skip update if events haven't changed (simple length + last event check)
          if (prev.length === newEvents.length && prev.length > 0) {
            const lastPrev = prev[prev.length - 1];
            const lastNew = newEvents[newEvents.length - 1];
            if (lastPrev.type === lastNew.type) {
              if (lastPrev.type === 'content' && lastNew.type === 'content' &&
                  lastPrev.content === lastNew.content) {
                return prev;
              }
              if (lastPrev.type === 'tool' && lastNew.type === 'tool' &&
                  lastPrev.toolCall.output === lastNew.toolCall.output) {
                return prev;
              }
            }
          }
          return newEvents;
        });

        // Check for max iterations flag
        if (task.streaming_state.hit_max_iterations) {
          setHitMaxIterations(true);
        }
      }

      // Check if task is done
      if (task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled') {
        stopTaskPolling();
        setIsStreaming(false);
        setActiveTask(null);

        // Refresh conversation to get final state (including any partial messages)
        // Do NOT clear streaming state until after refresh completes
        if (activeConversation) {
          const updated = await api.getConversation(activeConversation.id);
          setActiveConversation(updated);
          setConversations(prev => prev.map(c => c.id === updated.id ? updated : c));
          // Only clear streaming state after successful refresh
          setStreamingContent('');
          setStreamingEvents([]);
        } else {
          // No active conversation, safe to clear streaming state
          setStreamingContent('');
          setStreamingEvents([]);
        }

        if (task.status === 'failed' && task.error_message) {
          setError(task.error_message);
        }
      }
    } catch (err) {
      console.error('Failed to poll task status:', err);
      // Don't stop polling on transient errors
    }
  }, [activeConversation, stopTaskPolling]);

  const startTaskPolling = useCallback((taskId: string) => {
    stopTaskPolling();
    setIsPollingTask(true);
    setIsStreaming(true);
    lastSeenVersionRef.current = 0;  // Start fresh

    // Poll immediately, then every 300ms for responsive updates
    // Backend writes at ~400ms intervals, so 300ms gives us good balance
    pollTaskStatus(taskId);
    taskPollIntervalRef.current = setInterval(() => {
      pollTaskStatus(taskId);
    }, 300);
  }, [pollTaskStatus, stopTaskPolling]);

  // Check for active/interrupted background task when conversation changes
  useEffect(() => {
    const checkTasks = async () => {
      if (!activeConversation) {
        stopTaskPolling();
        setActiveTask(null);
        setInterruptedTask(null);
        return;
      }

      try {
        // Check for active (running/pending) task first
        const activeT = await api.getConversationActiveTask(activeConversation.id);
        if (activeT && (activeT.status === 'pending' || activeT.status === 'running')) {
          setActiveTask(activeT);
          setInterruptedTask(null);
          startTaskPolling(activeT.id);
        } else {
          setActiveTask(null);
          stopTaskPolling();

          // Check for interrupted task (from server restart)
          const interruptedT = await api.getConversationInterruptedTask(activeConversation.id);
          setInterruptedTask(interruptedT);
        }
      } catch (err) {
        console.error('Failed to check tasks:', err);
      }
    };

    checkTasks();

    // Cleanup on unmount or conversation change
    return () => {
      stopTaskPolling();
    };
  }, [activeConversation?.id, startTaskPolling, stopTaskPolling]);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      stopTaskPolling();
    };
  }, [stopTaskPolling]);

  const selectConversation = async (conversation: Conversation) => {
    try {
      // Stop any current polling when switching
      stopTaskPolling();
      setActiveTask(null);
      setIsStreaming(false);
      setStreamingContent('');
      setStreamingEvents([]);

      // Refresh the conversation to get latest messages
      const fresh = await api.getConversation(conversation.id);
      setActiveConversation(fresh);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load conversation');
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
      await api.deleteConversation(conversationId);
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
      const updated = await api.updateConversationTitle(conversationId, titleInput.trim());
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
      const updated = await api.updateConversationModel(activeConversation.id, newModel);
      setActiveConversation(updated);
      setConversations(prev => prev.map(c => c.id === updated.id ? updated : c));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to change model');
    }
  };

  const startFreshConversation = async () => {
    // Start a fresh conversation (same behavior as New button)
    // This replaces the old clearConversation which wiped messages
    if (isStreaming) return;
    try {
      const conversation = await api.createConversation();
      setConversations(prev => [conversation, ...prev]);
      setActiveConversation(conversation);
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
        await api.cancelChatTask(activeTask.id);
      } catch (err) {
        console.error('Failed to cancel task:', err);
      }
    }

    stopTaskPolling();
    setActiveTask(null);
    setInterruptedTask(null);
    setIsStreaming(false);

    // Refresh conversation to get current state (including partial messages)
    // Do NOT clear streaming state until after refresh completes
    if (activeConversation) {
      try {
        const updated = await api.getConversation(activeConversation.id);
        setActiveConversation(updated);
        setConversations(prev => prev.map(c => c.id === updated.id ? updated : c));
        // Only clear streaming state after successful refresh
        setStreamingContent('');
        setStreamingEvents([]);
      } catch {
        // Ignore refresh errors, but still clear streaming state
        setStreamingContent('');
        setStreamingEvents([]);
      }
    } else {
      // No active conversation, safe to clear streaming state
      setStreamingContent('');
      setStreamingEvents([]);
    }
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
    if (!message.trim() || !activeConversation || isStreaming) return;

    const userMessage = message.trim();
    setError(null);
    setHitMaxIterations(false);
    setIsConnectionError(false);
    setLastSentMessage(userMessage);

    // Check context limit before sending
    const currentTokens = calculateConversationTokens(activeConversation.messages);
    const newMessageTokens = estimateTokens(userMessage);
    const contextLimit = getContextLimit(activeConversation.model);

    if (currentTokens + newMessageTokens > contextLimit * 0.9) {
      setError(`Context limit nearly reached (${Math.round((currentTokens + newMessageTokens) / contextLimit * 100)}%). Consider starting a new conversation.`);
      return;
    }

    // Optimistically add user message
    const userMsg: ChatMessage = {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString(),
    };

    setActiveConversation(prev => prev ? {
      ...prev,
      messages: [...prev.messages, userMsg],
    } : null);

    setIsStreaming(true);
    setStreamingContent('');
    setStreamingEvents([]);
    setHitMaxIterations(false);

    try {
      // Use background task API
      const task = await api.sendMessageBackground(activeConversation.id, userMessage);
      setActiveTask(task);

      // Start polling for task updates
      startTaskPolling(task.id);

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

      // Remove the optimistic user message on error
      setActiveConversation(prev => prev ? {
        ...prev,
        messages: prev.messages.slice(0, -1),
      } : null);

      setIsStreaming(false);
      setStreamingContent('');
      setStreamingEvents([]);
    }
  };

  const sendMessage = async () => {
    if ((!inputValue.trim() && attachments.length === 0) || !activeConversation || isStreaming) return;

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
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const startEditMessage = (idx: number, content: string) => {
    setEditingMessageIdx(idx);
    setEditMessageContent(content);
  };

  const cancelEditMessage = () => {
    setEditingMessageIdx(null);
    setEditMessageContent('');
  };

  const submitEditMessage = async () => {
    if (!activeConversation || editingMessageIdx === null || !editMessageContent.trim()) return;

    const messageToSend = editMessageContent.trim();
    const truncateAt = editingMessageIdx;

    // Clear the edit state
    setEditingMessageIdx(null);
    setEditMessageContent('');
    setError(null);

    try {
      // Truncate messages on the backend to remove old message and all subsequent
      const truncated = await api.truncateConversation(activeConversation.id, truncateAt);
      setActiveConversation(truncated);
      setConversations(prev => prev.map(c => c.id === truncated.id ? truncated : c));

      // Check context limit before sending
      const currentTokens = calculateConversationTokens(truncated.messages);
      const newMessageTokens = estimateTokens(messageToSend);
      const contextLimit = getContextLimit(truncated.model);

      if (currentTokens + newMessageTokens > contextLimit * 0.9) {
        setError(`Context limit nearly reached. Consider starting a new conversation.`);
        return;
      }

      // Optimistically add user message
      const userMsg: ChatMessage = {
        role: 'user',
        content: messageToSend,
        timestamp: new Date().toISOString(),
      };

      setActiveConversation(prev => prev ? {
        ...prev,
        messages: [...prev.messages, userMsg],
      } : null);

      setIsStreaming(true);
      setStreamingContent('');
      setStreamingEvents([]);

      // Use background task API
      const task = await api.sendMessageBackground(activeConversation.id, messageToSend);
      setActiveTask(task);

      // Start polling for task updates
      startTaskPolling(task.id);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resend message');
      setIsStreaming(false);
      setStreamingContent('');
      setStreamingEvents([]);
    }
  };

  // Memoize context usage calculation to avoid recalculating on every render
  const contextUsagePercent = useMemo(() => {
    if (!activeConversation) return 0;
    const tokens = calculateConversationTokens(activeConversation.messages);
    const limit = getContextLimit(activeConversation.model);
    return Math.round(tokens / limit * 100);
  }, [activeConversation?.messages, activeConversation?.model, getContextLimit]);

  const renderConversationItem = (conv: Conversation) => {
    const metaText = `${conv.messages.length} messages | ${formatRelativeTime(conv.updated_at)}`;
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
                  <Loader2 size={12} className="spinning" />
                </span>
              )}
              {conv.title}
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

  return (
    <div className="chat-panel">
      {/* Conversations Sidebar */}
      <div className={`chat-sidebar ${showSidebar ? 'open' : ''}`}>
        <div className="chat-sidebar-header">
          <h3>Conversations</h3>
          <div className="chat-sidebar-header-actions">
            <button className="btn btn-sm" onClick={createNewConversation}>
              + New
            </button>
            <button
              className="chat-sidebar-collapse"
              onClick={() => setShowSidebar(false)}
              title="Hide sidebar"
            >
              «
            </button>
          </div>
        </div>

        <div className="chat-conversation-list">
          {conversations.length === 0 ? (
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
      </div>

      {/* Sidebar Expand Button (when collapsed) */}
      {!showSidebar && (
        <button
          className="chat-sidebar-expand"
          onClick={() => setShowSidebar(true)}
          title="Show sidebar"
        >
          »
        </button>
      )}

      {/* Main Chat Area */}
      <div className="chat-main">
        {activeConversation ? (
          <>
            {/* Chat Header */}
            <div className="chat-header">
              <div className="chat-header-info">
                <h2>{activeConversation.title}</h2>
                {availableModels.length > 0 ? (
                  <select
                    className="chat-model-select"
                    value={activeConversation.model}
                    onChange={(e) => changeModel(e.target.value)}
                    disabled={isStreaming || modelsLoading}
                    title="Select model for this conversation"
                  >
                    {availableModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} ({m.provider})
                      </option>
                    ))}
                    {/* If current model is not in available models, show it anyway */}
                    {!availableModels.some(m => m.id === activeConversation.model) && (
                      <option value={activeConversation.model}>
                        {activeConversation.model}
                      </option>
                    )}
                  </select>
                ) : (
                  <span className="chat-model-badge">{activeConversation.model}</span>
                )}
                {activeTask && (
                  <span className="chat-background-indicator" title="Processing in background - you can switch chats">
                    Background
                  </span>
                )}
              </div>
              <div className="chat-header-actions">
                <label className="chat-toggle-control" title="Show/hide tool calls in messages">
                  <span className="chat-toggle-label">Tools</span>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={showToolCalls}
                      onChange={(e) => setShowToolCalls(e.target.checked)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </label>
                <div className="chat-context-meter" title={`Context usage: ${contextUsagePercent}%`}>
                  <div
                    className="chat-context-fill"
                    style={{
                      width: `${Math.min(contextUsagePercent, 100)}%`,
                      backgroundColor: contextUsagePercent > 80 ? 'var(--color-error)' :
                                       contextUsagePercent > 60 ? '#fbbf24' :
                                       'var(--color-success)'
                    }}
                  />
                  <span className="chat-context-label">{contextUsagePercent}%</span>
                </div>
                <button className="btn btn-sm btn-secondary" onClick={startFreshConversation} title="Start a new conversation">
                  New Chat
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="chat-messages">
              {activeConversation.messages.length === 0 && !isStreaming ? (
                <div className="chat-welcome">
                  <h3>Start a conversation</h3>
                  <p>Ask questions about your indexed code, query databases, or get help with your systems.</p>
                </div>
              ) : (
                <>
                  {activeConversation.messages.map((msg, idx) => {
                    // Show separator for ANY "continue" message (continuation prompts)
                    const isContinuationPrompt = msg.role === 'user' && msg.content === 'continue';

                    if (isContinuationPrompt) {
                      return (
                        <div key={idx} className="chat-continuation-separator">
                          <span>···</span>
                        </div>
                      );
                    }

                    return (
                    <div key={idx} className={`chat-message chat-message-${msg.role}`}>
                      <div className="chat-message-content">
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
                            <div className="chat-edit-actions">
                              <button className="btn btn-sm" onClick={submitEditMessage}>Resend</button>
                              <button className="btn btn-sm btn-secondary" onClick={cancelEditMessage}>Cancel</button>
                            </div>
                          </>
                        ) : (
                          <>
                            {/* Render chronological events if available */}
                            {msg.role === 'assistant' && msg.events && msg.events.length > 0 ? (
                              <>
                                {msg.events.map((ev, evIdx) => (
                                  ev.type === 'tool' && showToolCalls ? (
                                    <div key={`event-${evIdx}`} className="chat-tool-calls">
                                      <ToolCallDisplay
                                        toolCall={{
                                          tool: ev.tool,
                                          input: ev.input,
                                          output: ev.output,
                                          status: 'complete'
                                        }}
                                        defaultExpanded={false}
                                        conversationId={activeConversation.id}
                                        siblingEvents={msg.events}
                                      />
                                    </div>
                                  ) : ev.type === 'content' ? (
                                    <div key={`event-${evIdx}`} className="chat-message-text markdown-content">
                                      <MemoizedMarkdown content={ev.content} />
                                    </div>
                                  ) : null
                                ))}
                              </>
                            ) : (
                              <>
                                {/* Fallback: Show stored tool calls for assistant messages (old format) */}
                                {showToolCalls && msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0 && (
                                  <div className="chat-tool-calls">
                                    {msg.tool_calls.map((tc, tcIdx) => (
                                      <ToolCallDisplay
                                        key={`${tc.tool}-${tcIdx}`}
                                        toolCall={{
                                          tool: tc.tool,
                                          input: tc.input,
                                          output: tc.output,
                                          status: 'complete'
                                        }}
                                        defaultExpanded={false}
                                      />
                                    ))}
                                  </div>
                                )}
                                {msg.role === 'user' ? (
                                  <>
                                    {(() => {
                                      const { text, attachments } = parseMessageContent(msg.content);
                                      return (
                                        <>
                                          {attachments.length > 0 && <MessageAttachments attachments={attachments} />}
                                          {text && (
                                            <div className="chat-message-text chat-message-user-text">
                                              {text}
                                            </div>
                                          )}
                                        </>
                                      );
                                    })()}
                                  </>
                                ) : (
                                  <div className="chat-message-text markdown-content">
                                    <MemoizedMarkdown content={typeof msg.content === 'string' ? msg.content : parseMessageContent(msg.content).text} />
                                  </div>
                                )}
                              </>
                            )}
                            <div className="chat-message-footer">
                              <span className="chat-message-time">
                                {formatRelativeTime(msg.timestamp)}
                              </span>
                              {msg.role === 'user' && !isStreaming && (
                                <button
                                  className="chat-message-edit-btn"
                                  onClick={() => startEditMessage(idx, msg.content)}
                                  title="Edit and resend"
                                >
                                  <Pencil size={12} />
                                </button>
                              )}
                            </div>
                          </>
                        )}
                      </div>
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
                          />
                        ))}
                        <div className="chat-message-streaming">
                          {consolidatedSegments.some(seg => seg.type === 'tool' && seg.toolCall?.status === 'running') ? 'Running tool...' : 'Generating...'}
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
                 interruptedTask) && (
                <div className="chat-continue-inline">
                  <span className="chat-continue-text">
                    Conversation interrupted, <button className="chat-continue-link" onClick={continueConversation}>continue?</button>
                  </span>
                </div>
              ))}

              <div ref={messagesEndRef} />
            </div>

            {/* Error Display */}
            {error && (
              <div className="chat-error">
                {error}
                <div className="chat-error-actions">
                  {isConnectionError && lastSentMessage && (
                    <button className="btn-resend" onClick={resendMessage}>Re-send</button>
                  )}
                  <button onClick={() => { setError(null); setIsConnectionError(false); }}>×</button>
                </div>
              </div>
            )}

            {/* Input Area */}
            <div className="chat-input-area">
              <div className="chat-input-wrapper">
                <FileAttachment
                  attachments={attachments}
                  onAttachmentsChange={setAttachments}
                  disabled={isStreaming}
                />
                <textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question or paste an image (Ctrl+V)..."
                  disabled={isStreaming}
                  rows={1}
                  className="chat-input"
                />
                {isStreaming ? (
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
                ) : (
                  (inputValue.trim() || attachments.length > 0) && (
                    <button
                      type="button"
                      className="btn chat-send-btn-inline"
                      onClick={sendMessage}
                      disabled={!activeConversation}
                      title="Send message"
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="19" x2="12" y2="5"></line>
                        <polyline points="5 12 12 5 19 12"></polyline>
                      </svg>
                    </button>
                  )
                )}
              </div>
            </div>
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
    </div>
  );
}
