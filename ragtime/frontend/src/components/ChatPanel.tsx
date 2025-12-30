import { useState, useEffect, useRef, useCallback, useMemo, memo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api } from '@/api';
import type { Conversation, ChatMessage, AvailableModel, ChatTask, User } from '@/types';

// Memoized markdown component to prevent re-parsing on every render
// Only re-renders when content actually changes
const MemoizedMarkdown = memo(function MemoizedMarkdown({ content }: { content: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>;
});

// Confirmation modal state
interface ConfirmationState {
  message: string;
  onConfirm: () => void;
  onCancel?: () => void;
}

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

// Component to display a tool call with collapsible details
// Memoized to prevent re-renders when tool call data hasn't changed
const ToolCallDisplay = memo(function ToolCallDisplay({ toolCall, defaultExpanded = false }: { toolCall: ActiveToolCall; defaultExpanded?: boolean }) {
  const [expanded, setExpanded] = useState(defaultExpanded);

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

  return (
    <div className={`tool-call tool-call-${toolCall.status}`}>
      <button
        className="tool-call-header"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="tool-call-icon">
          {toolCall.status === 'running' ? '⏳' : '✓'}
        </span>
        <span className="tool-call-name">{toolCall.tool}</span>
        <span className="tool-call-toggle">{expanded ? '▼' : '▶'}</span>
      </button>
      {expanded && (
        <div className="tool-call-details">
          {inputDisplay && (
            <div className="tool-call-section">
              <div className="tool-call-section-label">Query:</div>
              <pre className="tool-call-code">{inputDisplay}</pre>
            </div>
          )}
          {toolCall.output && (
            <div className="tool-call-section">
              <div className="tool-call-section-label">Result:</div>
              <pre className="tool-call-code">{toolCall.output}</pre>
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

// Get context limit for a model
function getContextLimit(model: string): number {
  const limits: Record<string, number> = {
    'gpt-4-turbo': 128000,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-3.5-turbo': 16385,
    'gpt-3.5-turbo-16k': 16385,
    'claude-3-opus-20240229': 200000,
    'claude-3-sonnet-20240229': 200000,
    'claude-3-haiku-20240307': 200000,
    'claude-3-5-sonnet-20241022': 200000,
    'llama2': 4096,
    'llama3': 8192,
    'llama3.1': 128000,
    'mistral': 8192,
    'mixtral': 32768,
    'codellama': 16384,
    'qwen2.5': 32768,
  };

  // Try exact match first
  if (limits[model]) return limits[model];

  // Try partial match
  for (const [key, value] of Object.entries(limits)) {
    if (model.toLowerCase().includes(key.toLowerCase())) {
      return value;
    }
  }

  // Default fallback
  return 8192;
}

// Estimate tokens from text
function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

// Calculate total tokens for a conversation
function calculateConversationTokens(messages: ChatMessage[]): number {
  return messages.reduce((total, msg) => total + estimateTokens(msg.content), 0);
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

interface ChatPanelProps {
  currentUser: User;
}

export function ChatPanel({ currentUser }: ChatPanelProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [inputValue, setInputValue] = useState('');
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

  // Confirmation modal state
  const [confirmation, setConfirmation] = useState<ConfirmationState | null>(null);

  // Inline confirmation for delete (conversation ID waiting for confirmation)
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Background task state
  const [activeTask, setActiveTask] = useState<ChatTask | null>(null);
  const [isPollingTask, setIsPollingTask] = useState(false);
  const taskPollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastSeenVersionRef = useRef<number>(0);  // Track last seen version for delta polling

  // Available models state
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
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

  // Check for active background task when conversation changes
  useEffect(() => {
    const checkActiveTask = async () => {
      if (!activeConversation) {
        stopTaskPolling();
        setActiveTask(null);
        return;
      }

      try {
        const task = await api.getConversationActiveTask(activeConversation.id);
        if (task && (task.status === 'pending' || task.status === 'running')) {
          setActiveTask(task);
          startTaskPolling(task.id);
        } else {
          setActiveTask(null);
          stopTaskPolling();
        }
      } catch (err) {
        console.error('Failed to check active task:', err);
      }
    };

    checkActiveTask();

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

  const clearConversation = async () => {
    if (!activeConversation) return;

    // Show modal confirmation
    setConfirmation({
      message: 'Clear all messages in this conversation?',
      onConfirm: async () => {
        setConfirmation(null);
        try {
          const updated = await api.clearConversation(activeConversation.id);
          setActiveConversation(updated);
          setConversations(prev => prev.map(c => c.id === updated.id ? updated : c));
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to clear conversation');
        }
      },
      onCancel: () => setConfirmation(null)
    });
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
    // Directly send the continuation message
    sendMessageDirect('Please continue from where you left off.');
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
    if (!inputValue.trim() || !activeConversation || isStreaming) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    sendMessageDirect(userMessage);
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
  }, [activeConversation?.messages, activeConversation?.model]);

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
                  ⏳
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
                ✓
              </button>
              <button
                className="chat-action-btn cancel-delete"
                onClick={(e) => {
                  e.stopPropagation();
                  setDeleteConfirmId(null);
                }}
                title="Cancel"
              >
                ✗
              </button>
            </>
          ) : (
            <>
              <button
                className="chat-action-btn"
                onClick={(e) => startEditingTitle(conv, e)}
                title="Rename"
              >
                ✏️
              </button>
              <button
                className="chat-action-btn"
                onClick={(e) => deleteConversation(conv.id, e)}
                title="Delete"
              >
                🗑️
              </button>
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="chat-panel">
      {/* Confirmation Modal */}
      {confirmation && (
        <div className="modal-overlay" onClick={() => confirmation.onCancel?.()}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Confirm Action</h3>
              <button className="modal-close" onClick={() => confirmation.onCancel?.()}>×</button>
            </div>
            <div className="modal-body">
              <p>{confirmation.message}</p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => confirmation.onCancel?.()}>
                Cancel
              </button>
              <button className="btn btn-danger" onClick={confirmation.onConfirm}>
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

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
                <button className="btn btn-sm btn-secondary" onClick={clearConversation}>
                  Clear
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
                  {activeConversation.messages.map((msg, idx) => (
                    <div key={idx} className={`chat-message chat-message-${msg.role}`}>
                      <div className="chat-message-content">
                        {editingMessageIdx === idx ? (
                          <div className="chat-message-edit">
                            <textarea
                              value={editMessageContent}
                              onChange={(e) => setEditMessageContent(e.target.value)}
                              className="chat-edit-input"
                              autoFocus
                            />
                            <div className="chat-edit-actions">
                              <button className="btn btn-sm" onClick={submitEditMessage}>Resend</button>
                              <button className="btn btn-sm btn-secondary" onClick={cancelEditMessage}>Cancel</button>
                            </div>
                          </div>
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
                                <div className="chat-message-text markdown-content">
                                  <MemoizedMarkdown content={msg.content} />
                                </div>
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
                                  ✏️
                                </button>
                              )}
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  ))}

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

              {/* Continue Button - shows after completed assistant response */}
              {!isStreaming && activeConversation && activeConversation.messages.length > 0 &&
               activeConversation.messages[activeConversation.messages.length - 1].role === 'assistant' && (
                <div className="chat-continue-inline">
                  {hitMaxIterations && (
                    <span className="chat-continue-reason">Max iterations reached</span>
                  )}
                  <button
                    className="btn btn-continue-inline"
                    onClick={continueConversation}
                    title="Ask the assistant to continue its response"
                  >
                    Continue
                  </button>
                </div>
              )}

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
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a message... (Enter to send, Shift+Enter for new line)"
                disabled={isStreaming}
                rows={1}
                className="chat-input"
              />
              {isStreaming ? (
                <button
                  className="btn chat-stop-btn"
                  onClick={stopStreaming}
                >
                  Stop
                </button>
              ) : (
                <button
                  className="btn chat-send-btn"
                  onClick={sendMessage}
                  disabled={!inputValue.trim()}
                >
                  Send
                </button>
              )}
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
