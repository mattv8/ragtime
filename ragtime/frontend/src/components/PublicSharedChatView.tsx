import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { Info } from 'lucide-react';

import { api } from '@/api';
import type { AuthStatus, Conversation, MessageEvent, PublicShareTargetResponse, SharedConversationResponse, User } from '@/types';
import { formatChatTimestamp } from '@/utils';
import { calculateConversationContextUsage, parseStoredModelIdentifier } from '@/utils/contextUsage';

import { LinkifiedText, MemoizedMarkdown, MessageAttachments, ToolCallDisplay, parseMessageContent, type ActiveToolCall } from './ChatPanel';
import { FileAttachment, attachmentsToContentParts, type AttachmentFile } from './FileAttachment';
import { LoginCard } from './LoginPage';
import { Popover } from './Popover';
import { ResizeHandle } from './ResizeHandle';
import { ContextUsagePie } from './shared/ContextUsagePie';
import { UserMenu } from './UserMenu';
import { UserSpaceSharedView } from './UserSpaceSharedView';

const SHARED_CHAT_POLL_INTERVAL_MS = 30000;
const SHARED_CHAT_SSE_REFRESH_DEBOUNCE_MS = 250;
const SHARED_CHAT_MIN_INPUT_HEIGHT = 96;
const SHARED_CHAT_MAX_INPUT_HEIGHT = 400;
const SHARED_CHAT_DEFAULT_CONTEXT_LIMIT = 128_000;

interface PublicSharedChatViewProps {
  shareToken?: string;
  ownerUsername?: string;
  shareSlug?: string;
  currentUser: User | null;
  authStatus: AuthStatus | null;
  serverName?: string;
  onLoginSuccess: (user: User) => void;
  onLogout: () => void;
}

function SharedChatSurface({
  shareToken,
  ownerUsername,
  shareSlug,
  currentUser,
  authStatus,
  serverName = 'Ragtime',
  onLoginSuccess,
  onLogout,
}: PublicSharedChatViewProps) {
  const [sharedConversation, setSharedConversation] = useState<SharedConversationResponse | null>(null);
  const [redirectingToAuthenticatedChat, setRedirectingToAuthenticatedChat] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showLogin, setShowLogin] = useState(false);
  const [messageDraft, setMessageDraft] = useState('');
  const [attachments, setAttachments] = useState<AttachmentFile[]>([]);
  const [sending, setSending] = useState(false);
  const [sharePasswordDraft, setSharePasswordDraft] = useState('');
  const [submittedSharePassword, setSubmittedSharePassword] = useState<string | undefined>(undefined);
  const [passwordRequired, setPasswordRequired] = useState(false);
  const [inputAreaHeight, setInputAreaHeight] = useState(SHARED_CHAT_MIN_INPUT_HEIGHT);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const sendingRef = useRef(false);
  const showLoginRef = useRef(false);
  // Snapshot the currentUser at mount time. Auto-redirect only fires when the
  // user transitions from anonymous (null at mount) to authenticated during
  // this view's lifetime — i.e., after signing in via the inline modal.
  // Visiting the public link while already authenticated (e.g., the owner
  // clicking "Open Link" from the share modal) preserves the public preview.
  const initialCurrentUserIdRef = useRef<string | null>(currentUser?.id ?? null);

  useEffect(() => {
    sendingRef.current = sending;
  }, [sending]);

  useEffect(() => {
    showLoginRef.current = showLogin;
  }, [showLogin]);

  useEffect(() => {
    if (!currentUser) {
      return;
    }
    if (!sharedConversation?.can_edit || !sharedConversation.conversation.id) {
      return;
    }
    // If the user was already authenticated when this view first mounted,
    // they're previewing the public share — do not redirect.
    if (initialCurrentUserIdRef.current) {
      return;
    }

    setRedirectingToAuthenticatedChat(true);
    const params = new URLSearchParams();
    params.set('view', 'chat');
    params.set('conversation', sharedConversation.conversation.id);
    window.location.assign(`/?${params.toString()}`);
  }, [currentUser, sharedConversation?.can_edit, sharedConversation?.conversation.id]);

  const loadSharedConversation = useCallback(async (silent = false) => {
    if (!silent) {
      setLoading(true);
    }
    try {
      const response = shareToken
        ? await api.getSharedConversation(shareToken, submittedSharePassword)
        : await api.getSharedConversationBySlug(ownerUsername as string, shareSlug as string, submittedSharePassword);
      setSharedConversation(response);
      setError(null);
      setPasswordRequired(false);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : 'Failed to load shared conversation';
      const isPasswordIssue = message.toLowerCase().includes('password required') || message.toLowerCase().includes('invalid password');
      if (silent && !isPasswordIssue) {
        // Silent refresh — keep last known state, don't surface transient errors
        return;
      }
      setSharedConversation(null);
      setError(message);
      setPasswordRequired(isPasswordIssue);
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, [ownerUsername, shareSlug, shareToken, submittedSharePassword]);

  useEffect(() => {
    void loadSharedConversation(false);
  }, [loadSharedConversation, currentUser?.id]);

  // Live polling — refresh conversation periodically. Skip while user is sending,
  // a password is required, or the login overlay is open. Acts as a long
  // safety-net fallback while the SSE stream below drives realtime updates.
  useEffect(() => {
    if (passwordRequired) return;
    const interval = window.setInterval(() => {
      if (sendingRef.current || showLoginRef.current) return;
      void loadSharedConversation(true);
    }, SHARED_CHAT_POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [loadSharedConversation, passwordRequired]);

  // Realtime SSE — subscribe to per-conversation events so streaming chat
  // feels instant. The backend publishes `task_started`, throttled
  // `task_progress` and `task_completed` events on `conversation:{id}`; on
  // each event we trigger a debounced silent reload of the conversation.
  useEffect(() => {
    if (passwordRequired) return;
    if (!sharedConversation) return;

    const url = shareToken
      ? api.getSharedConversationEventsUrl(shareToken, submittedSharePassword)
      : api.getSharedConversationEventsUrlBySlug(
          ownerUsername as string,
          shareSlug as string,
          submittedSharePassword,
        );

    const es = new EventSource(url, { withCredentials: true });
    let pendingTimeout: number | null = null;
    let cancelled = false;

    const scheduleRefresh = () => {
      if (cancelled) return;
      if (pendingTimeout !== null) return;
      pendingTimeout = window.setTimeout(() => {
        pendingTimeout = null;
        if (cancelled) return;
        if (sendingRef.current || showLoginRef.current) return;
        void loadSharedConversation(true);
      }, SHARED_CHAT_SSE_REFRESH_DEBOUNCE_MS);
    };

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (
          data?.event === 'task_started' ||
          data?.event === 'task_progress' ||
          data?.event === 'task_completed' ||
          data?.type === 'title_update'
        ) {
          scheduleRefresh();
        }
      } catch {
        // Ignore malformed events
      }
    };

    es.onerror = () => {
      // EventSource will auto-reconnect; if the share/password becomes
      // invalid the next reconnect will surface via subsequent fetches.
    };

    return () => {
      cancelled = true;
      if (pendingTimeout !== null) {
        window.clearTimeout(pendingTimeout);
      }
      es.close();
    };
  }, [
    loadSharedConversation,
    ownerUsername,
    passwordRequired,
    shareSlug,
    shareToken,
    sharedConversation?.conversation.id,
    submittedSharePassword,
  ]);

  const handleSendMessage = useCallback(async () => {
    const trimmedMessage = messageDraft.trim();
    if (!trimmedMessage && attachments.length === 0) {
      return;
    }
    setSending(true);
    try {
      const payload = attachments.length > 0
        ? { message: JSON.stringify(attachmentsToContentParts(trimmedMessage, attachments)) }
        : { message: trimmedMessage };
      const response = shareToken
        ? await api.sendSharedConversationMessage(shareToken, payload, submittedSharePassword)
        : await api.sendSharedConversationMessageBySlug(ownerUsername as string, shareSlug as string, payload, submittedSharePassword);
      setSharedConversation((previous) => previous ? {
        ...previous,
        conversation: response.conversation,
      } : previous);
      setMessageDraft('');
      setAttachments([]);
      setError(null);
    } catch (sendError) {
      setError(sendError instanceof Error ? sendError.message : 'Failed to send message');
    } finally {
      setSending(false);
    }
  }, [attachments, messageDraft, ownerUsername, shareSlug, shareToken, submittedSharePassword]);

  const conversation: Conversation | null = sharedConversation?.conversation || null;
  const canEdit = Boolean(sharedConversation?.can_edit);
  const ownerLabel = sharedConversation?.owner_display_name || sharedConversation?.owner_username || 'unknown';

  const modelLabel = useMemo(() => {
    const raw = (conversation?.model || '').trim();
    if (!raw) return null;
    const parsed = parseStoredModelIdentifier(raw);
    return parsed.modelId || raw;
  }, [conversation?.model]);

  // Server already pre-slices messages based on the share record's
  // scope_anchor_message_idx + scope_direction (the scope is bound to the share
  // link, not to URL params, so it can't be bypassed by clients).
  const visibleMessages = useMemo(
    () => conversation?.messages || [],
    [conversation?.messages],
  );

  const contextLimitForPie = sharedConversation?.context_limit && sharedConversation.context_limit > 0
    ? sharedConversation.context_limit
    : SHARED_CHAT_DEFAULT_CONTEXT_LIMIT;

  // When the server pre-slices messages (scoped share), persisted total tokens
  // for the full conversation are no longer accurate.
  const isScopedShare = Boolean(
    sharedConversation?.scope_anchor_message_idx !== null
    && sharedConversation?.scope_anchor_message_idx !== undefined
    && sharedConversation?.scope_direction,
  );

  const contextUsage = useMemo(() => calculateConversationContextUsage({
    messages: visibleMessages,
    persistedConversationTokens: isScopedShare ? null : conversation?.total_tokens,
    contextLimit: contextLimitForPie,
  }), [contextLimitForPie, conversation?.total_tokens, isScopedShare, visibleMessages]);

  // Auto-scroll to latest message
  useEffect(() => {
    if (!loading && conversation?.messages.length) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [loading, conversation?.messages.length]);

  // Auto-resize textarea
  const handleTextareaInput = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, []);

  useEffect(() => {
    handleTextareaInput();
  }, [messageDraft, handleTextareaInput]);

  const handleKeyDown = useCallback((event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!sending && (messageDraft.trim() || attachments.length > 0)) {
        void handleSendMessage();
      }
    }
  }, [attachments.length, handleSendMessage, messageDraft, sending]);

  const handleResizeInputArea = useCallback((delta: number) => {
    setInputAreaHeight((prev) => {
      const proposed = prev - delta;
      return Math.min(SHARED_CHAT_MAX_INPUT_HEIGHT, Math.max(SHARED_CHAT_MIN_INPUT_HEIGHT, proposed));
    });
  }, []);

  const showLoginModal = showLogin && !currentUser && Boolean(authStatus);

  if (currentUser && redirectingToAuthenticatedChat) {
    return (
      <div className="userspace-shared-status">Opening chat in your workspace...</div>
    );
  }

  return (
    <div className="app-shell app-shell-locked">
      <div className={showLoginModal ? 'shared-blur-host shared-blur-host-active' : 'shared-blur-host'}>
        <nav className="topnav">
          <span className="topnav-brand">{serverName}</span>
          <div className="topnav-links">
            {currentUser ? (
              <UserMenu user={currentUser} onLogout={onLogout} />
            ) : (
              <button className="topnav-link" onClick={() => setShowLogin((previous) => !previous)}>
                Sign In
              </button>
            )}
          </div>
        </nav>

        <div className="container chat-page-container">
          <div className="chat-panel chat-panel-shared">
          <div className="chat-main">
            <div className="chat-header chat-header-shared">
              <div className="chat-header-info">
                <div className="chat-header-title-row">
                  <h2 className="chat-shared-title">
                    {conversation?.title || 'Shared Chat'}
                    {conversation && (
                      <Popover
                        trigger="click"
                        position="bottom"
                        content={(
                          <div className="chat-shared-context-popover">
                            <div className="chat-shared-context-popover-model" title={modelLabel || 'Unknown model'}>
                              {modelLabel || 'Unknown model'}
                            </div>
                            <div className="chat-shared-context-popover-divider" />
                            <div className="chat-shared-context-popover-row">
                              <ContextUsagePie
                                currentTokens={contextUsage.currentTokens}
                                totalTokens={contextUsage.totalTokens}
                                contextLimit={contextUsage.contextLimit}
                              />
                              <div className="chat-shared-context-popover-tokens">
                                <span className="chat-shared-context-popover-tokens-value">
                                  {contextUsage.currentTokens.toLocaleString()}
                                </span>
                                <span className="chat-shared-context-popover-tokens-label">
                                  / {contextUsage.contextLimit.toLocaleString()} tokens
                                </span>
                              </div>
                            </div>
                          </div>
                        )}
                      >
                        <button
                          type="button"
                          className="chat-shared-context-info-btn"
                          aria-label="Show model and context usage"
                          title="Model and context usage"
                        >
                          <Info size={11} />
                        </button>
                      </Popover>
                    )}
                  </h2>
                </div>
              </div>
              <div className="chat-shared-header-right">
                <div className="chat-shared-owner-col" aria-label="Shared by">
                  <span className="chat-shared-owner-label">Shared by</span>
                  <span className="chat-shared-owner-value" title={ownerLabel}>{ownerLabel}</span>
                </div>
              </div>
            </div>

            {error && !passwordRequired && !loading && (
              <div className="chat-error" role="alert">
                {error}
              </div>
            )}

            {loading ? (
              <div className="chat-messages chat-messages-skeleton">
                <div className="chat-loading">Loading shared conversation...</div>
              </div>
            ) : passwordRequired ? (
              <div className="chat-messages">
                <div className="chat-welcome" style={{ maxWidth: 420, margin: '40px auto', textAlign: 'left' }}>
                  <h3>Password required</h3>
                  <p className="chat-welcome-subtitle" style={{ marginTop: 8 }}>
                    The owner protected this shared chat with a password.
                  </p>
                  <input
                    type="password"
                    className="chat-input"
                    value={sharePasswordDraft}
                    onChange={(event) => setSharePasswordDraft(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') {
                        event.preventDefault();
                        setSubmittedSharePassword(sharePasswordDraft || undefined);
                      }
                    }}
                    placeholder="Enter share password"
                    style={{ marginTop: 16 }}
                  />
                  <div style={{ marginTop: 12, display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={() => setSubmittedSharePassword(sharePasswordDraft || undefined)}
                      disabled={!sharePasswordDraft}
                    >
                      Unlock
                    </button>
                  </div>
                </div>
              </div>
            ) : conversation ? (
              <div className="chat-messages">
                {visibleMessages.length === 0 ? (
                  <div className="chat-welcome">
                    <h3>No messages yet</h3>
                    <p>This shared chat doesn&apos;t have any messages yet.</p>
                  </div>
                ) : (
                  visibleMessages.map((msg, idx) => {
                    const { text, attachments } = parseMessageContent(msg.content);
                    const isUser = msg.role === 'user';
                    const siblingEvents = msg.events?.map((event) => (
                      event.type === 'tool'
                        ? { type: 'tool', tool: event.tool, output: event.output }
                        : { type: event.type }
                    ));
                    const fallbackToolCalls = (!msg.events || msg.events.length === 0) ? (msg.tool_calls ?? []) : [];

                    return (
                      <div
                        key={`shared-msg-${idx}-${msg.timestamp || idx}`}
                        className={`chat-branch-wrapper chat-branch-wrapper-${msg.role}`}
                      >
                        <div className={`chat-message chat-message-${msg.role}`}>
                          <div className="chat-message-content">
                            {isUser ? (
                              <>
                                {attachments.length > 0 && <MessageAttachments attachments={attachments} />}
                                {text && (
                                  <div className="chat-message-text chat-message-user-text">
                                    <LinkifiedText text={text} />
                                  </div>
                                )}
                              </>
                            ) : (
                              <>
                                {msg.events && msg.events.length > 0 ? (
                                  msg.events.map((event: MessageEvent, eventIdx: number) => {
                                    if (event.type === 'tool') {
                                      const toolCall: ActiveToolCall = {
                                        tool: event.tool,
                                        input: event.input,
                                        output: event.output,
                                        presentation: event.presentation,
                                        connection: event.connection,
                                        status: 'complete',
                                      };
                                      return (
                                        <div key={`shared-tool-${idx}-${eventIdx}`} className="chat-tool-calls">
                                          <ToolCallDisplay
                                            toolCall={toolCall}
                                            defaultExpanded={false}
                                            siblingEvents={siblingEvents}
                                            allowRerun={false}
                                          />
                                        </div>
                                      );
                                    }

                                    if (event.type === 'content' || event.type === 'reasoning') {
                                      return (
                                        <div key={`shared-content-${idx}-${eventIdx}`} className="chat-message-text markdown-content">
                                          <MemoizedMarkdown content={event.content} />
                                        </div>
                                      );
                                    }

                                    return null;
                                  })
                                ) : (
                                  <div className="chat-message-text markdown-content">
                                    <MemoizedMarkdown content={text} />
                                  </div>
                                )}
                                {fallbackToolCalls.length > 0 && (
                                  <div className="chat-tool-calls">
                                    {fallbackToolCalls.map((toolCall, toolIdx) => (
                                      <ToolCallDisplay
                                        key={`shared-legacy-tool-${idx}-${toolIdx}`}
                                        toolCall={{ ...toolCall, status: 'complete' }}
                                        defaultExpanded={false}
                                        allowRerun={false}
                                      />
                                    ))}
                                  </div>
                                )}
                              </>
                            )}
                            <div className="chat-message-footer">
                              <span className="chat-message-time">
                                {formatChatTimestamp(msg.timestamp)}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
                <div ref={messagesEndRef} />
              </div>
            ) : null}

            {!loading && !passwordRequired && conversation && canEdit && (
              <>
                <ResizeHandle
                  direction="vertical"
                  className="resize-handle resize-handle-vertical chat-resize-handle"
                  onResize={handleResizeInputArea}
                />
                <div
                  className="chat-input-area manual-resize"
                  style={{ height: `${inputAreaHeight}px`, minHeight: `${inputAreaHeight}px` }}
                >
                  <div className="chat-input-wrapper">
                    <FileAttachment
                      attachments={attachments}
                      onAttachmentsChange={setAttachments}
                      conversationId={conversation.id}
                      disabled={sending}
                    />
                    <textarea
                      ref={textareaRef}
                      value={messageDraft}
                      onChange={(event) => setMessageDraft(event.target.value)}
                      onInput={handleTextareaInput}
                      onKeyDown={handleKeyDown}
                      placeholder="Ask a question or paste files/images (Ctrl+V)..."
                      rows={1}
                      className="chat-input"
                      disabled={sending}
                    />
                    {(messageDraft.trim() || attachments.length > 0) && (
                      <div className="chat-input-inline-actions">
                        <button
                          type="button"
                          className="btn chat-send-btn-inline"
                          onClick={() => void handleSendMessage()}
                          disabled={sending}
                          title={sending ? 'Sending...' : 'Send message'}
                        >
                          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="12" y1="19" x2="12" y2="5"></line>
                            <polyline points="5 12 12 5 19 12"></polyline>
                          </svg>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>

          </div>
        </div>
      </div>

      {showLoginModal && (
        <div
          className="shared-login-modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(event) => {
            if (event.target === event.currentTarget) setShowLogin(false);
          }}
        >
          <div className="shared-login-modal-card">
            <LoginCard authStatus={authStatus!} onLoginSuccess={onLoginSuccess} serverName={serverName} />
            <div className="shared-login-modal-footer">
              <button type="button" className="btn btn-link" onClick={() => setShowLogin(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function PublicSharedChatView(props: PublicSharedChatViewProps) {
  const { shareToken, ownerUsername, shareSlug } = props;
  const [target, setTarget] = useState<PublicShareTargetResponse['target_type'] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      setLoading(true);
      try {
        const response = shareToken
          ? await api.resolvePublicShareTarget(shareToken)
          : await api.resolvePublicShareTargetBySlug(ownerUsername as string, shareSlug as string);
        if (!cancelled) {
          setTarget(response.target_type);
          setError(null);
        }
      } catch (resolveError) {
        if (!cancelled) {
          setError(resolveError instanceof Error ? resolveError.message : 'Failed to resolve shared link');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ownerUsername, shareSlug, shareToken]);

  if (loading) {
    return <div className="userspace-shared-status">Loading shared link...</div>;
  }

  if (error || !target || target === 'unknown') {
    return <div className="userspace-shared-status userspace-error">{error || 'Shared link not found'}</div>;
  }

  if (target === 'workspace') {
    return shareToken
      ? <UserSpaceSharedView shareToken={shareToken} />
      : <UserSpaceSharedView ownerUsername={ownerUsername} shareSlug={shareSlug} />;
  }

  return <SharedChatSurface {...props} />;
}