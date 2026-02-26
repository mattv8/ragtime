import { useState, useEffect, useRef, useCallback, useMemo, memo, isValidElement, type ReactNode, type CSSProperties } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Copy, Check, Loader2, Pencil, Trash2, Maximize2, Minimize2, X, AlertCircle, RefreshCw, FileText, ChevronDown, ChevronRight, ChevronLeft, Users, Bot, MessageSquare, MessageSquarePlus } from 'lucide-react';
import { api } from '@/api';
import type { Conversation, ChatMessage, AvailableModel, ChatTask, User, ContentPart, ConversationMember, UserSpaceAvailableTool } from '@/types';
import { FileAttachment, attachmentsToContentParts, type AttachmentFile } from './FileAttachment';
import { ModelSelector } from './ModelSelector';
import { ResizeHandle } from './ResizeHandle';
import { calculateConversationTokens, calculateStreamingTokens, estimateTokens } from '@/utils/contextUsage';
import { ContextUsagePie } from './shared/ContextUsagePie';
import { MemberManagementModal } from './shared/MemberManagementModal';
import { ToolSelectorDropdown } from './shared/ToolSelectorDropdown';

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
  connection?: {
    tool_config_id: string;
    tool_config_name?: string;
    tool_type?: string;
  };
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

// Global URL regex for efficient linkification
const URL_PATTERN = /(https?:\/\/[^\s<]+[^<.,:;"')\]\s])/g;

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
    return null;
  };

  const statusIcon = getStatusIcon();

  return (
    <div className={`tool-call tool-call-${toolCall.status}${isFailed ? ' tool-call-failed' : ''}`}>
      <div className="tool-call-header-row">
        <button
          className="tool-call-header"
          onClick={() => setExpanded(!expanded)}
        >
          {statusIcon && <span className="tool-call-icon">{statusIcon}</span>}
          <span className="tool-call-name">{toolCall.tool}</span>
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

    if (previousTitleRef.current === 'New Chat' && title !== 'New Chat') {
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
  workspaceId?: string;
  onUserMessageSubmitted?: (message: string) => void | Promise<void>;
  onTaskComplete?: () => void;
  onFullscreenChange?: (fullscreen: boolean) => void;
  embedded?: boolean;
  readOnly?: boolean;
  readOnlyMessage?: string;
}

export function ChatPanel({
  currentUser,
  workspaceId,
  onUserMessageSubmitted,
  onTaskComplete,
  onFullscreenChange,
  embedded = false,
  readOnly = false,
  readOnlyMessage,
}: ChatPanelProps) {
  const MIN_INPUT_AREA_HEIGHT = 96;
  const INPUT_AREA_COLLAPSE_THRESHOLD = 80;

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
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
  const [editingTitle, setEditingTitle] = useState<string | null>(null);
  const [titleInput, setTitleInput] = useState('');
  const [editingMessageIdx, setEditingMessageIdx] = useState<number | null>(null);
  const [editMessageContent, setEditMessageContent] = useState('');
  const [editMessageAttachments, setEditMessageAttachments] = useState<AttachmentFile[]>([]);
  const [hitMaxIterations, setHitMaxIterations] = useState(false);
  const [showToolCalls, setShowToolCalls] = useState(() => {
    const saved = localStorage.getItem('chat-show-tool-calls');
    return saved !== null ? saved === 'true' : true;
  });
  const [lastSentMessage, setLastSentMessage] = useState<string>('');
  const [isConnectionError, setIsConnectionError] = useState(false);
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});
  const isAdmin = currentUser.role === 'admin';
  const effectiveReadOnlyMessage = readOnlyMessage || 'Workspace is read-only. Viewers can review messages but cannot send prompts.';

  // Inline confirmation for delete (conversation ID waiting for confirmation)
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Background task state
  const [activeTask, setActiveTask] = useState<ChatTask | null>(null);
  const [interruptedTask, setInterruptedTask] = useState<ChatTask | null>(null);  // Last interrupted task for continue
  const [_isPollingTask, setIsPollingTask] = useState(false);
  const lastSeenVersionRef = useRef<number>(0);  // Track last seen version for delta polling
  // Available models state
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [isWorkspaceConversationMenuOpen, setIsWorkspaceConversationMenuOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [conversationMembers, setConversationMembers] = useState<ConversationMember[]>([]);
  const [conversationToolIds, setConversationToolIds] = useState<string[]>([]);
  const [availableTools, setAvailableTools] = useState<UserSpaceAvailableTool[]>([]);
  const [showMembersModal, setShowMembersModal] = useState(false);
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [savingMembers, setSavingMembers] = useState(false);
  const [savingTools, setSavingTools] = useState(false);

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

  const isConversationOwner = myConversationRole === 'owner' || (activeConversation?.user_id === currentUser?.id && conversationMembers.length === 0);
  const isConversationViewer = myConversationRole === 'viewer';

  const toggleFullscreen = useCallback(() => {
    const next = !isFullscreen;
    setIsFullscreen(next);
    onFullscreenChange?.(next);
  }, [isFullscreen, onFullscreenChange]);

  const parseStoredConversationModel = useCallback((storedModel: string): { provider?: 'openai' | 'anthropic' | 'ollama'; modelId: string } => {
    if (!storedModel) {
      return { modelId: '' };
    }

    const delimiterIndex = storedModel.indexOf('::');
    if (delimiterIndex <= 0) {
      return { modelId: storedModel };
    }

    const provider = storedModel.slice(0, delimiterIndex) as 'openai' | 'anthropic' | 'ollama';
    const modelId = storedModel.slice(delimiterIndex + 2);

    if (provider !== 'openai' && provider !== 'anthropic' && provider !== 'ollama') {
      return { modelId: storedModel };
    }

    return { provider, modelId };
  }, []);

  // Image modal state
  const [modalImageUrl, setModalImageUrl] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      onFullscreenChange?.(false);
    };
  }, [onFullscreenChange]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatMessagesRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const processingTaskRef = useRef<string | null>(null);
  const titleSourceRef = useRef<Map<string, EventSource>>(new Map());
  const workspaceConversationDropdownRef = useRef<HTMLDivElement>(null);
  const chatMainRef = useRef<HTMLDivElement>(null);
  const prevSidebarWidth = useRef(280);
  const prevInputAreaHeight = useRef(MIN_INPUT_AREA_HEIGHT);

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

  const handleResizeInputArea = useCallback((delta: number) => {
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
        return prev;
      }

      // --- Compute max input height from container ---
      const container = chatMainRef.current;
      let maxInputHeight = 600; // fallback
      if (container) {
        const containerHeight = container.clientHeight;
        // Measure all non-input siblings (header, error, handle) to get exact available space
        let occupiedHeight = 0;
        for (const child of Array.from(container.children)) {
          const el = child as HTMLElement;
          if (el.classList.contains('chat-input-area')) continue;   // skip the input area itself
          if (el.classList.contains('chat-messages')) continue;     // skip messages (will be collapsed)
          occupiedHeight += el.getBoundingClientRect().height;
        }
        maxInputHeight = containerHeight - occupiedHeight;
      }

      // --- Collapse messages area (dragging up past max) ---
      if (draggingUp && proposed >= maxInputHeight) {
        if (!isMessagesCollapsed) {
          prevInputAreaHeight.current = prev;
          setIsMessagesCollapsed(true);
        }
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
  }, [INPUT_AREA_COLLAPSE_THRESHOLD, MIN_INPUT_AREA_HEIGHT, isInputAreaCollapsed, isMessagesCollapsed]);

  const expandInputArea = useCallback(() => {
    setIsInputAreaCollapsed(false);
    setInputAreaHeight(Math.max(MIN_INPUT_AREA_HEIGHT, prevInputAreaHeight.current || MIN_INPUT_AREA_HEIGHT));
    requestAnimationFrame(() => inputRef.current?.focus());
  }, [MIN_INPUT_AREA_HEIGHT]);

  const expandMessages = useCallback(() => {
    setIsMessagesCollapsed(false);
    setInputAreaHeight(Math.max(MIN_INPUT_AREA_HEIGHT, prevInputAreaHeight.current || MIN_INPUT_AREA_HEIGHT));
  }, [MIN_INPUT_AREA_HEIGHT]);

  // Auto-size textarea to fit content while filling resized input pane
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = '0px';
      const scrollHeight = textarea.scrollHeight;
      const wrapperHeight = textarea.parentElement?.clientHeight ?? 0;
      textarea.style.height = Math.max(scrollHeight, wrapperHeight) + 'px';
    }
  }, [inputValue, inputAreaHeight]);

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
  }, [workspaceId]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (!shouldAutoScrollRef.current || !chatMessagesRef.current) return;

    chatMessagesRef.current.scrollTo({
      top: chatMessagesRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [activeConversation?.messages, streamingContent]);

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
      const data = await api.listConversations(workspaceId);
      let userspaceConversationIds = new Set<string>();

      const getLinkedWorkspaceId = (conversation: Conversation): string | null => {
        const camelWorkspaceId = (conversation as Conversation & { workspaceId?: string | null }).workspaceId;
        return conversation.workspace_id ?? camelWorkspaceId ?? null;
      };

      if (!workspaceId) {
        try {
          const workspacePage = await api.listUserSpaceWorkspaces(0, 200);
          userspaceConversationIds = new Set(
            workspacePage.items.flatMap((workspace) => workspace.conversation_ids || [])
          );
        } catch (workspaceErr) {
          console.warn('Failed to load userspace workspaces for conversation filtering:', workspaceErr);
        }
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
    } catch (err) {
      console.error('Failed to load conversations:', err);
    }
  };

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
      const toolIds = await api.getConversationTools(conversationId);
      setConversationToolIds(toolIds);
    } catch (err) {
      console.error('Failed to fetch conversation tools:', err);
      setConversationToolIds([]);
    }
  }, []);

  const fetchAvailableTools = useCallback(async () => {
    try {
      const tools = await api.listUserSpaceAvailableTools();
      setAvailableTools(tools);
    } catch (err) {
      console.error('Failed to fetch available tools:', err);
      setAvailableTools([]);
    }
  }, []);

  // Load conversation members and tools when conversation changes
  useEffect(() => {
    if (activeConversation) {
      fetchConversationTools(activeConversation.id);
      if (!embedded) {
        fetchConversationMembers(activeConversation.id);
      }
    }
  }, [activeConversation, embedded, fetchConversationMembers, fetchConversationTools]);

  // Load available tools on mount
  useEffect(() => {
    if (!embedded || Boolean(workspaceId)) {
      fetchAvailableTools();
    }
  }, [embedded, fetchAvailableTools, workspaceId]);

  const handleOpenMembersModal = useCallback(async () => {
    if (!activeConversation || !isConversationOwner) return;
    try {
      const users = await api.listUsers();
      setAllUsers(users);
    } catch {
      setAllUsers([]);
    }
    setShowMembersModal(true);
  }, [activeConversation, isConversationOwner]);

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

  const handleToggleConversationTool = useCallback(async (toolId: string) => {
    if (!activeConversation || isConversationViewer) return;
    const nextSelected = new Set(conversationToolIds);
    if (nextSelected.has(toolId)) {
      nextSelected.delete(toolId);
    } else {
      nextSelected.add(toolId);
    }

    setSavingTools(true);
    try {
      await api.updateConversationTools(activeConversation.id, {
        tool_config_ids: Array.from(nextSelected),
      });
      setConversationToolIds(Array.from(nextSelected));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update tool selection');
    } finally {
      setSavingTools(false);
    }
  }, [activeConversation, conversationToolIds, isConversationViewer]);

  const formatUserLabel = useCallback((user?: Pick<User, 'username' | 'display_name'> | null, fallbackId?: string) => {
    const username = user?.username?.trim() || fallbackId?.trim() || 'unknown';
    const displayName = user?.display_name?.trim();
    if (displayName && displayName !== username) {
      return `${displayName} (@${username})`;
    }
    return `@${username}`;
  }, []);

  // Get context limit for a model from API-provided data (uses LiteLLM's dataset)
  const getContextLimit = useCallback((modelId: string): number => {
    const model = availableModels.find(m => m.id === modelId);
    return model?.context_limit ?? DEFAULT_CONTEXT_LIMIT;
  }, [availableModels]);

  const createNewConversation = async () => {
    if (readOnly) return;
    try {
      shouldAutoScrollRef.current = true;
      const conversation = await api.createConversation(undefined, workspaceId);
      setConversations(prev => [conversation, ...prev]);
      setActiveConversation(conversation);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create conversation');
    }
  };


  // Listen for auto-generated titles per conversation using SSE
  const stopTitleStreamFor = useCallback((conversationId: string) => {
    const es = titleSourceRef.current.get(conversationId);
    if (es) {
      es.close();
    }
    titleSourceRef.current.delete(conversationId);
  }, []);

  const startTitleStreamFor = useCallback((conversationId: string) => {
    if (titleSourceRef.current.has(conversationId)) return;

    const target = conversations.find(c => c.id === conversationId);
    if (!target || target.title !== 'New Chat') return;

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
        // Just close on error, don't retry locally as SSE retries automatically
        // But if connection keeps failing, we might want to clean up.
        // For now, let generic browser retry logic handle transient issues?
        // Actually, if we get error, it might be 404 or something, let's close to be safe from inf loops
        es.close();
        titleSourceRef.current.delete(conversationId);
      };

      titleSourceRef.current.set(conversationId, es);
    } catch (e) {
      console.error("Failed to start title stream", e);
    }
  }, [conversations]);

  useEffect(() => {
    const newChatIds = conversations.filter(c => c.title === 'New Chat').map(c => c.id);

    newChatIds.forEach(startTitleStreamFor);

    // Stop streams for conversations that resolved or were removed
    titleSourceRef.current.forEach((_, id) => {
      if (!newChatIds.includes(id)) {
        stopTitleStreamFor(id);
      }
    });

    return () => {
      titleSourceRef.current.forEach(es => es.close());
      titleSourceRef.current.clear();
    };
  }, [conversations, startTitleStreamFor, stopTitleStreamFor]);

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
                    if (ev.type === 'content') return { type: 'content', content: ev.content || '' };
                    return {
                        type: 'tool',
                        toolCall: {
                            tool: ev.tool || '',
                            input: ev.input,
                            output: ev.output,
                          connection: ev.connection,
                            status: ev.output ? 'complete' : 'running'
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
            setIsStreaming(false);
            setActiveTask(null);

            // Refresh conversation on completion
            if (activeConversation) {
                try {
                    const updated = await api.getConversation(activeConversation.id, workspaceId);
                    setActiveConversation(updated);
                    setConversations(prev => prev.map(c => c.id === updated.id ? updated : c));
                    setStreamingContent('');
                    setStreamingEvents([]);
                } catch (e) { console.error(e); }
            }

            // Notify parent that the task finished (e.g. refresh workspace preview)
            if (onTaskComplete) {
                try { onTaskComplete(); } catch (e) { console.error(e); }
            }
        }
    }
  }, [activeConversation, stopTaskStreaming, onTaskComplete, workspaceId]);

  const startTaskAndStream = useCallback(async (conversationId: string, message: string) => {
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

    try {
        // 2. Start background task
        const task = await api.sendMessageBackground(conversationId, message, workspaceId);
        setActiveTask(task);
        setInterruptedTask(null);

        // 3. Connect to stream
        await connectTaskStream(task.id);
    } catch (err: any) {
       console.error(err);
       setError(err.message || 'Failed to start task');
       setIsStreaming(false);
    }
  }, [activeConversation, connectTaskStream, workspaceId]);

  // Check for active/interrupted background task when conversation changes
  useEffect(() => {
    const checkTasks = async () => {
      // If we are switching conversations, ensure we stop any previous stream
      if (!activeConversation) {
        stopTaskStreaming();
        setActiveTask(null);
        setInterruptedTask(null);
        return;
      }

      // If we are already streaming a task for this conversation, don't interrupt it.
      // But how do we know if the running task belongs to THIS conversation?
      // processingTaskRef stores taskId. activeTask stores taskId.
      // We should check API to be sure.

      try {
        const activeT = await api.getConversationActiveTask(activeConversation.id, workspaceId);

        // Use functional state update to avoid dependency issues if needed, but here simple set is fine
        if (activeT && (activeT.status === 'pending' || activeT.status === 'running')) {
            setActiveTask(activeT);
            setInterruptedTask(null);

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
                 // Only verify interrupted if no active task
                 const interruptedT = await api.getConversationInterruptedTask(activeConversation.id, workspaceId);
                 setInterruptedTask(interruptedT);
            }
        }
      } catch (err) {
         console.error('Failed to check tasks:', err);
      }
    };

    checkTasks();

    return () => {
        // Stop streaming when conversation ID changes (unmounting this effect instance)
        stopTaskStreaming();
    };
  }, [activeConversation?.id, connectTaskStream, stopTaskStreaming, workspaceId]);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      stopTaskStreaming();
    };
  }, [stopTaskStreaming]);

  const selectConversation = async (conversation: Conversation) => {
    try {
      if (!embedded && typeof window !== 'undefined' && window.matchMedia('(max-width: 768px)').matches) {
        setShowSidebar(false);
      }
      shouldAutoScrollRef.current = true;
      // Stop any current streaming when switching
      stopTaskStreaming();
      setActiveTask(null);
      setIsStreaming(false);
      setStreamingContent('');
      setStreamingEvents([]);

      // Refresh the conversation to get latest messages
      const fresh = await api.getConversation(conversation.id, workspaceId);
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
      const selected = availableModels.find((model) => model.id === newModel);
      const updated = await api.updateConversationModel(
        activeConversation.id,
        newModel,
        workspaceId,
        selected?.provider,
      );
      setActiveConversation(updated);
      setConversations(prev => prev.map(c => c.id === updated.id ? updated : c));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to change model');
    }
  };

  const startFreshConversation = async () => {
    if (readOnly) return;
    // Start a fresh conversation (same behavior as New button)
    // This replaces the old clearConversation which wiped messages
    if (isStreaming) return;
    shouldAutoScrollRef.current = true;
    try {
      const conversation = await api.createConversation(undefined, workspaceId);
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
        await api.cancelChatTask(activeTask.id, workspaceId);
      } catch (err) {
        console.error('Failed to cancel task:', err);
      }
    }

    stopTaskStreaming();
    setActiveTask(null);
    setInterruptedTask(null);
    // Don't modify isStreaming yet to avoid UI flash/loss of content during refresh

    // Refresh conversation to get current state (including partial messages)
    if (activeConversation) {
      try {
        const updated = await api.getConversation(activeConversation.id, workspaceId);
        setActiveConversation(updated);
        setConversations(prev => prev.map(c => c.id === updated.id ? updated : c));
      } catch (err) {
        console.error('Failed to update conversation after stop:', err);
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
    if (!message.trim() || !activeConversation || isStreaming || readOnly) return;
    shouldAutoScrollRef.current = true;

    const userMessage = message.trim();
    setError(null);
    setHitMaxIterations(false);
    setIsConnectionError(false);
    setLastSentMessage(userMessage);

    // Check context limit before sending
    const estimatedConversationTokens = calculateConversationTokens(activeConversation.messages);
    const persistedConversationTokens = activeConversation.total_tokens || 0;
    const currentTokens = persistedConversationTokens > 0
      ? persistedConversationTokens
      : estimatedConversationTokens;
    const newMessageTokens = estimateTokens(userMessage);
    const contextLimit = getContextLimit(parseStoredConversationModel(activeConversation.model).modelId);

    if (currentTokens + newMessageTokens > contextLimit * 0.9) {
      setError(`Context limit nearly reached (${Math.round((currentTokens + newMessageTokens) / contextLimit * 100)}%). Consider starting a new conversation.`);
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
    if ((!inputValue.trim() && attachments.length === 0) || !activeConversation || isStreaming || readOnly) return;

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
    if (readOnly) return;
    if (e.key === 'Enter' && !e.shiftKey) {
      if (isStreaming) return; // Allow typing while a response streams

      e.preventDefault();
      sendMessage();
    }
  };

  const resizeImageDataUrl = useCallback(async (
    dataUrl: string,
    mimeType: string,
    maxDimension = 1024,
    quality = 0.8
  ): Promise<string> => {
    try {
      const img = new Image();
      img.src = dataUrl;
      await img.decode();

      const { naturalWidth, naturalHeight } = img;
      if (!naturalWidth || !naturalHeight) return dataUrl;

      const scale = Math.min(1, maxDimension / Math.max(naturalWidth, naturalHeight));
      if (scale === 1) return dataUrl;

      const width = Math.round(naturalWidth * scale);
      const height = Math.round(naturalHeight * scale);

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return dataUrl;
      ctx.drawImage(img, 0, 0, width, height);

      return canvas.toDataURL(mimeType || 'image/png', quality);
    } catch {
      return dataUrl;
    }
  }, []);

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
          const resized = await resizeImageDataUrl(url, mimeType);
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
  }, [resizeImageDataUrl]);

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

  const submitEditMessage = async () => {
    if (readOnly || !activeConversation || editingMessageIdx === null || (!editMessageContent.trim() && editMessageAttachments.length === 0)) return;
    shouldAutoScrollRef.current = true;

    let messageToSend: string = editMessageContent.trim();
    if (editMessageAttachments.length > 0) {
      const parts = attachmentsToContentParts(messageToSend, editMessageAttachments);
      messageToSend = JSON.stringify(parts);
    }

    const truncateAt = Math.max(0, editingMessageIdx);

    // Clear the edit state
    setEditingMessageIdx(null);
    setEditMessageContent('');
    setEditMessageAttachments([]);
    setError(null);

    try {
      // 1. Local Optimistic Truncation & Update
      // We do this immediately so the UI reflects the "revert" behavior users expect.
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

      // 2. Server Sync (Truncate)
      // Call endpoint to persist string truncation (removes old message & subsequent from DB)
      const truncated = await api.truncateConversation(activeConversation.id, truncateAt, workspaceId);

      // Note: We don't overwrite local state with 'truncated' here because 'truncated'
      // doesn't have our optimistic user message yet.

      // 3. Start background task
      // This sends the message and creates a background task
      const task = await api.sendMessageBackground(truncated.id, messageToSend, workspaceId);
      setActiveTask(task);
      setInterruptedTask(null);

      // 4. Connect to stream
      await connectTaskStream(task.id);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resend message');
      setIsStreaming(false);
      setStreamingContent('');
      setStreamingEvents([]);

      // Restore authoritative state from server on error
      try {
        const refreshed = await api.getConversation(activeConversation.id, workspaceId);
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
        contextLimit: DEFAULT_CONTEXT_LIMIT,
        contextUsagePercent: 0,
        projectedInputPercent: 0,
        hasHeadroom: true,
      };
    }

    const estimatedConversationTokens = calculateConversationTokens(activeConversation.messages);
    const persistedConversationTokens = activeConversation.total_tokens || 0;
    const currentTokens = persistedConversationTokens > 0
      ? persistedConversationTokens
      : estimatedConversationTokens;
    const streamingTokens = isStreaming ? calculateStreamingTokens(streamingEvents as any, streamingContent) : 0;
    const totalTokens = currentTokens + streamingTokens;
    const contextLimit = getContextLimit(parseStoredConversationModel(activeConversation.model).modelId);
    const contextUsagePercent = Math.round((totalTokens / contextLimit) * 100);
    const nextMessageTokens = estimateTokens(inputValue.trim());
    const projectedInputPercent = Math.round(((totalTokens + nextMessageTokens) / contextLimit) * 100);
    const hasHeadroom = totalTokens + nextMessageTokens <= contextLimit * 0.9;

    return {
      currentTokens,
      totalTokens,
      contextLimit,
      contextUsagePercent,
      projectedInputPercent,
      hasHeadroom,
    };
  }, [activeConversation, getContextLimit, inputValue, isStreaming, streamingContent, streamingEvents]);

  const showWorkspaceConversationSelect = embedded && Boolean(workspaceId);
  const showInlineToolSelector = Boolean(activeConversation)
    && !isConversationViewer
    && !readOnly;

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
                <Loader2 size={14} className="spinning" />
              </span>
            )}
          </div>
        </div>

        <div className={`chat-conversation-list ${!isAdmin ? 'chat-conversation-list-non-admin' : ''}`}>
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
        {activeConversation ? (
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
                    <button
                      type="button"
                      className="model-selector-trigger chat-workspace-conversation-trigger"
                      onClick={() => setIsWorkspaceConversationMenuOpen((open) => !open)}
                      title="Select a workspace chat"
                      aria-haspopup="listbox"
                      aria-expanded={isWorkspaceConversationMenuOpen}
                    >
                      <MessageSquare size={14} className="chat-workspace-conversation-icon" aria-hidden="true" />
                      <span className="model-selector-text chat-workspace-conversation-trigger-label">{activeConversation.title || 'New Chat'}</span>
                      <span className="model-selector-arrow chat-workspace-conversation-trigger-arrow">▾</span>
                    </button>

                    {isWorkspaceConversationMenuOpen && (
                      <div className="model-selector-dropdown chat-workspace-conversation-dropdown">
                        <div className="model-selector-dropdown-inner" role="listbox" aria-label="Workspace chats">
                          {conversations.map((conversation) => (
                            <button
                              key={conversation.id}
                              type="button"
                              role="option"
                              aria-selected={conversation.id === activeConversation.id}
                              className={`model-selector-item chat-workspace-conversation-item ${conversation.id === activeConversation.id ? 'is-selected' : ''}`}
                              onClick={() => {
                                setIsWorkspaceConversationMenuOpen(false);
                                void selectConversation(conversation);
                              }}
                            >
                              <span className="model-selector-item-name">{conversation.title || 'New Chat'}</span>
                            </button>
                          ))}
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
                {!modelsLoading && (
                  <ContextUsagePie
                    currentTokens={contextUsage.currentTokens}
                    totalTokens={contextUsage.totalTokens}
                    contextLimit={contextUsage.contextLimit}
                  />
                )}
                {!embedded && activeConversation && isConversationOwner && (
                  <button
                    className="btn btn-secondary btn-sm btn-icon"
                    onClick={handleOpenMembersModal}
                    title="Manage conversation members"
                  >
                    <Users size={14} />
                  </button>
                )}
                {!embedded && activeConversation && !isConversationViewer && (
                  <ToolSelectorDropdown
                    availableTools={availableTools}
                    selectedToolIds={new Set(conversationToolIds)}
                    onToggleTool={handleToggleConversationTool}
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
                  selectedModelId={parseStoredConversationModel(activeConversation.model).modelId}
                  onModelChange={changeModel}
                  disabled={isStreaming || modelsLoading}
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
                <button className="btn btn-sm btn-secondary chat-new-chat-btn" onClick={startFreshConversation} title="Start a new conversation" disabled={readOnly}>
                  <MessageSquarePlus size={14} className="chat-new-chat-icon" aria-hidden="true" />
                  <span className="chat-new-chat-label">New Chat</span>
                </button>
              </div>
            </div>

            {/* Messages */}
            {!isMessagesCollapsed && (
            <div className="chat-messages" ref={chatMessagesRef} onScroll={handleScroll}>
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
                      <div className="chat-message-content" key={editingMessageIdx === idx ? 'editing' : 'viewing'}>
                        {editingMessageIdx === idx ? (
                          <>
                            {/* Attachments for editing mode */}
                            <div className="chat-edit-attachments-wrapper">
                              <FileAttachment
                                attachments={editMessageAttachments}
                                onAttachmentsChange={setEditMessageAttachments}
                              />
                            </div>
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
                                          connection: ev.connection,
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
                                          connection: tc.connection,
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
                                {formatRelativeTime(msg.timestamp)}
                              </span>
                              {msg.role === 'user' && !isStreaming && !readOnly && (
                                <button
                                  className="chat-message-edit-btn"
                                  onClick={() => {
                                    const parsed = parseMessageContent(msg.content);
                                    startEditMessage(idx, parsed.text, parsed.attachments);
                                  }}
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
                 interruptedTask) && !readOnly && (
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
                    <button className="btn-resend" onClick={resendMessage}>Re-send</button>
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
            <div className="chat-input-area" style={{ height: `${inputAreaHeight}px`, minHeight: `${inputAreaHeight}px` }}>
              {readOnly && (
                <div className="chat-readonly-note" role="status">
                  {effectiveReadOnlyMessage}
                </div>
              )}
              <div className="chat-input-wrapper">
                <FileAttachment
                  attachments={attachments}
                  onAttachmentsChange={setAttachments}
                  disabled={readOnly || isStreaming}
                />
                <textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={readOnly ? effectiveReadOnlyMessage : 'Ask a question or paste an image (Ctrl+V)...'}
                  rows={1}
                  className="chat-input"
                  disabled={readOnly}
                />
                {isStreaming ? (
                  <div className="chat-input-inline-actions">
                    {showInlineToolSelector && (
                      <ToolSelectorDropdown
                        availableTools={availableTools}
                        selectedToolIds={new Set(conversationToolIds)}
                        onToggleTool={handleToggleConversationTool}
                        disabled={savingTools}
                        readOnly={false}
                        saving={savingTools}
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
                  !readOnly && (
                    <div className="chat-input-inline-actions">
                      {showInlineToolSelector && (
                        <ToolSelectorDropdown
                          availableTools={availableTools}
                          selectedToolIds={new Set(conversationToolIds)}
                          onToggleTool={handleToggleConversationTool}
                          disabled={savingTools}
                          readOnly={false}
                          saving={savingTools}
                          title="Workspace Tools"
                        />
                      )}
                      {(inputValue.trim() || attachments.length > 0) && (
                        <button
                          type="button"
                          className="btn chat-send-btn-inline"
                          onClick={sendMessage}
                          disabled={!activeConversation || !contextUsage.hasHeadroom}
                          title={contextUsage.hasHeadroom ? 'Send message' : `Context headroom too low (${contextUsage.projectedInputPercent}%)`}
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
          allUsers={allUsers}
          ownerId={conversationOwnerId}
          entityType="conversation"
          formatUserLabel={formatUserLabel}
          saving={savingMembers}
        />
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
