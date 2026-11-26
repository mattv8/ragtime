import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { ChevronDown, ChevronUp, GripVertical, Plus, Trash2, Upload, X } from 'lucide-react';
import { api } from '@/api';
import { INSTRUCTION_SKILL_LIMITS as LIMITS, INSTRUCTION_SKILL_NAME_PATTERN } from '@/types';
import type { InstructionSkill, InstructionSkillOwner, InstructionSkillMode } from '@/types';
import { estimateTokens, estimateTokensForLength } from '@/utils/contextUsage';
import {
  DEFAULT_MCP_ROUTE_OWNER_ID,
  builtinSkillState,
  type BuiltinInstructionSkill,
} from '@/utils/instructionSkills';
import { DeleteConfirmButton } from './DeleteConfirmButton';
import { ToastContainer, useToast } from './shared/Toast';

interface Props {
  open: boolean;
  owner: InstructionSkillOwner;
  ownerName: string;
  onClose(): void;
  onSkillsChanged?(skills: InstructionSkill[]): void;
}

type Fields = Pick<InstructionSkill, 'name' | 'description' | 'content' | 'mode'>;
const emptyFields: Fields = { name: '', description: '', content: '', mode: 'on_demand' };
const fieldNames = ['name', 'description', 'content', 'mode'] as const;
const NEAR_BUDGET_RATIO = 0.9;
const UPLOAD_MAX_BYTES = 200 * 1024;
const number = (value: number) => value.toLocaleString('en-US');
const sanitize = (value: string) => value.toLowerCase().replace(/[^a-z0-9-]/g, '-');
const normalize = (value: string) => sanitize(value).replace(/-+/g, '-').replace(/^-|-$/g, '');
const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : 'Request failed';
const errorStatus = (error: unknown) =>
  typeof error === 'object' && error !== null && 'status' in error ? error.status : undefined;

function parseUpload(text: string, filename: string): { fields: Fields; hasHeader: boolean } {
  const match = text.match(/^---\s*\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|$)/);
  let name = filename.replace(/\.(md|txt)$/i, '');
  let description = '';
  if (match) {
    const lines = match[1].split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const item = lines[i].match(/^(name|description):\s*(.*)$/);
      if (!item) continue;
      let value = item[2].trim();
      if (value.startsWith('"')) {
        try {
          value = JSON.parse(value) as string;
        } catch {
          /* Keep the original scalar. */
        }
      } else if (value.startsWith("'") && value.endsWith("'")) {
        value = value.slice(1, -1).replace(/''/g, "'");
      } else if (value === '>' || value === '|') {
        const parts: string[] = [];
        while (i + 1 < lines.length && /^\s+/.test(lines[i + 1])) parts.push(lines[++i].trim());
        value = parts.join(value === '>' ? ' ' : '\n');
      }
      if (item[1] === 'name') name = value;
      else description = value;
    }
  }
  return {
    fields: {
      ...emptyFields,
      name: normalize(name).slice(0, LIMITS.nameMaxLength),
      description,
      content: match ? text.slice(match[0].length) : text,
    },
    hasHeader: Boolean(match),
  };
}

export function InstructionSkillsDialog({
  open,
  owner,
  ownerName,
  onClose,
  onSkillsChanged,
}: Props) {
  const [skills, setSkills] = useState<InstructionSkill[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState(false);
  const [draftEnabled, setDraftEnabled] = useState(true);
  const [fields, setFields] = useState<Fields>(emptyFields);
  const [snapshot, setSnapshot] = useState<Fields>(emptyFields);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [error, setError] = useState('');
  const [nameError, setNameError] = useState('');
  const [descriptionTouched, setDescriptionTouched] = useState(false);
  const [budgetError, setBudgetError] = useState('');
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<{
    action: () => void;
    close: boolean;
    origin: HTMLElement | null;
  } | null>(null);
  const [announcement, setAnnouncement] = useState('');
  const [dragId, setDragId] = useState<string | null>(null);
  const [drop, setDrop] = useState<{ id: string; position: 'before' | 'after' } | null>(null);
  const [retry, setRetry] = useState(0);
  const [builtins, setBuiltins] = useState<BuiltinInstructionSkill[]>([]);
  const [builtinsLoading, setBuiltinsLoading] = useState(false);
  const [builtinsError, setBuiltinsError] = useState(false);
  const [builtinsExpanded, setBuiltinsExpanded] = useState<boolean | null>(null);
  const [builtinsRetry, setBuiltinsRetry] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const openerRef = useRef<HTMLElement | null>(null);
  const generation = useRef(0);
  const reorderTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingReorder = useRef<(() => void) | null>(null);
  const flushReorder = useCallback(() => {
    if (reorderTimer.current) clearTimeout(reorderTimer.current);
    reorderTimer.current = null;
    const run = pendingReorder.current;
    pendingReorder.current = null;
    run?.();
  }, []);
  const navigateRef = useRef<(action: () => void, close?: boolean) => void>(() => {});
  const [toasts, toast] = useToast();
  const focusSkill = (id: string) => {
    const rows = rootRef.current?.querySelectorAll<HTMLElement>('[data-instruction-skill-id]');
    const row = Array.from(rows ?? []).find((element) => element.dataset.instructionSkillId === id);
    row?.querySelector<HTMLElement>('[data-instruction-skill-select]')?.focus();
  };
  const selected = skills.find((item) => item.id === selectedId);
  const isDefaultRoute = owner.kind === 'route' && owner.id === DEFAULT_MCP_ROUTE_OWNER_ID;
  const emptyHelp = isDefaultRoute
    ? "Skills are instructions added to the default /mcp route, after Ragtime's built-in guide. Only clients that can use the general tool catalog receive them."
    : owner.kind === 'tool'
      ? 'Skills are instructions the model follows when it uses this tool.'
      : 'Skills are instructions sent to every agent that connects to this route.';
  const alwaysHelp = isDefaultRoute
    ? "Sent to every eligible client on the default /mcp route. Workspace-only development credentials don't receive it."
    : owner.kind === 'route'
      ? 'Sent to every agent that connects to this route.'
      : 'The full text is sent on every model call that can use this tool.';
  const dirty = draft || fieldNames.some((key) => fields[key] !== snapshot[key]);
  const duplicate = skills.some((item) => item.name === fields.name && item.id !== selectedId);
  const invalidName = !INSTRUCTION_SKILL_NAME_PATTERN.test(fields.name);
  const reason = invalidName
    ? 'Enter a valid name'
    : duplicate
      ? 'A skill with this name already exists'
      : !fields.description.trim()
        ? 'Description is required'
        : fields.description.length > LIMITS.descriptionMaxLength
          ? 'Description is too long'
          : fields.content.length > LIMITS.contentMaxLength
            ? 'Instructions are too long'
            : '';
  const liveSkills = skills.map((item) => (item.id === selectedId ? { ...item, ...fields } : item));
  const total =
    liveSkills.reduce(
      (sum, item) => sum + (item.enabled && item.mode === 'always' ? item.content.length : 0),
      0,
    ) + (draft && draftEnabled && fields.mode === 'always' ? fields.content.length : 0);
  const overBudget = total > LIMITS.alwaysIncludeBudgetChars;
  const alwaysTokens = estimateTokensForLength(total);
  const budgetMessage = (value: number) =>
    `Always-include skills for this ${owner.kind} total ${number(value)} characters; the limit is ${number(LIMITS.alwaysIncludeBudgetChars)}.`;
  const uploadTooLargeMessage = `That file is too large. A skill can hold up to ${number(LIMITS.contentMaxLength)} characters.`;
  const skillLimitTitle = `Limit of ${LIMITS.maxSkillsPerOwner} skills reached`;

  useEffect(() => {
    if (!open) return;
    generation.current += 1;
    setBusy(false);
    setBuiltinsExpanded(null);
    openerRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    rootRef.current?.focus();
    return () => {
      generation.current += 1;
      openerRef.current?.focus();
      flushReorder();
    };
  }, [open, owner.id, owner.kind, flushReorder]);

  useEffect(() => {
    if (!open) return;
    let active = true;
    setLoading(true);
    setLoadError('');
    setDraft(false);
    setSelectedId(null);
    setFields(emptyFields);
    setSnapshot(emptyFields);
    setPending(null);
    setError('');
    setNameError('');
    api
      .listInstructionSkills({ kind: owner.kind, id: owner.id })
      .then((items) => {
        if (!active) return;
        setSkills(items);
        const first = items[0];
        if (first) {
          setSelectedId(first.id);
          setFields(first);
          setSnapshot(first);
        }
        setLoading(false);
        requestAnimationFrame(() => {
          if (!active) return;
          rootRef.current
            ?.querySelector<HTMLElement>(
              first ? '[data-instruction-skill-select]' : '#instruction-skills-new',
            )
            ?.focus();
        });
      })
      .catch((cause: unknown) => {
        if (active) {
          setLoading(false);
          setLoadError(errorMessage(cause));
        }
      });
    return () => {
      active = false;
    };
  }, [open, owner.id, owner.kind, retry]);

  useEffect(() => {
    if (!open || owner.kind !== 'tool') return;
    let active = true;
    setBuiltins([]);
    setBuiltinsError(false);
    setBuiltinsLoading(true);
    api
      .listBuiltinInstructionSkills(owner.id)
      .then((items) => {
        if (active) setBuiltins(items);
      })
      .catch(() => {
        if (active) setBuiltinsError(true);
      })
      .finally(() => {
        if (active) setBuiltinsLoading(false);
      });
    return () => {
      active = false;
    };
  }, [open, owner.id, owner.kind, builtinsRetry]);

  useEffect(() => {
    if (pending) document.getElementById('instruction-skills-unsaved-confirm')?.focus();
  }, [pending]);

  if (!open) return null;
  const publish = (items: InstructionSkill[], requestGeneration: number) => {
    if (generation.current === requestGeneration) setSkills(items);
    onSkillsChanged?.(items);
  };
  const navigate = (action: () => void, close = false) => {
    if (busy) return;
    if (reorderTimer.current) {
      if (!close) return;
      flushReorder();
    }
    if (dirty)
      setPending({
        action,
        close,
        origin: document.activeElement instanceof HTMLElement ? document.activeElement : null,
      });
    else action();
  };
  navigateRef.current = navigate;
  const select = (item: InstructionSkill) => {
    if (!draft && selectedId === item.id) return;
    navigate(() => {
      setSelectedId(item.id);
      setDraft(false);
      setFields(item);
      setSnapshot(item);
      setError('');
      setNameError('');
      setDescriptionTouched(false);
    });
  };
  const newDraft = (next: Fields = emptyFields) => {
    const requestGeneration = generation.current;
    setSelectedId(null);
    setDraft(true);
    setDraftEnabled(true);
    setFields(next);
    setSnapshot(emptyFields);
    setError('');
    setNameError('');
    setDescriptionTouched(false);
    requestAnimationFrame(() => {
      if (generation.current !== requestGeneration) return;
      nameRef.current?.focus();
      nameRef.current?.select();
    });
  };
  const discardDraft = () => {
    if (busy || reorderTimer.current) return;
    const discard = () => {
      const replacement = skills[0];
      setDraft(false);
      setSelectedId(replacement?.id ?? null);
      setFields(replacement ?? emptyFields);
      setSnapshot(replacement ?? emptyFields);
    };
    if (fields.name || fields.description || fields.content) navigate(discard);
    else discard();
  };
  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const requestGeneration = generation.current;
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (!/\.(md|txt)$/i.test(file.name)) {
      toast.error('Only .md or .txt files can be uploaded.');
      return;
    }
    if (file.size > UPLOAD_MAX_BYTES) {
      toast.error(uploadTooLargeMessage);
      return;
    }
    try {
      const bytes = await file.arrayBuffer();
      if (generation.current !== requestGeneration) return;
      const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
      if (text.includes('\0')) throw new Error('text');
      if (!text.trim()) {
        toast.error('That file is empty.');
        return;
      }
      if (text.length > LIMITS.contentMaxLength) {
        toast.error(uploadTooLargeMessage);
        return;
      }
      const parsed = parseUpload(text, file.name);
      navigateRef.current(() => {
        newDraft(parsed.fields);
        if (!parsed.hasHeader)
          toast.info('No SKILL.md header found. Name taken from the file name; add a description.');
      });
    } catch {
      if (generation.current === requestGeneration)
        toast.error('That file does not look like text.');
    }
  };
  const save = async () => {
    if (busy || reorderTimer.current || !dirty || reason || overBudget) return;
    const requestGeneration = generation.current;
    setBusy(true);
    setError('');
    setNameError('');
    try {
      let saved: InstructionSkill;
      if (draft)
        saved = await api.createInstructionSkill(owner, { ...fields, enabled: draftEnabled });
      else {
        if (!selected) return;
        const changes: Partial<Fields> = {};
        for (const key of fieldNames)
          if (fields[key] !== snapshot[key]) Object.assign(changes, { [key]: fields[key] });
        saved = await api.updateInstructionSkill(owner, selected.id, changes);
      }
      const updated = draft
        ? [...skills, saved]
        : skills.map((item) => (item.id === saved.id ? saved : item));
      publish(updated, requestGeneration);
      if (generation.current !== requestGeneration) return;
      setSelectedId(saved.id);
      setDraft(false);
      setSnapshot(fields);
      toast.success('Skill saved');
      requestAnimationFrame(() => {
        if (generation.current === requestGeneration) focusSkill(saved.id);
      });
    } catch (cause) {
      if (generation.current !== requestGeneration) return;
      if (errorStatus(cause) === 409) {
        setNameError(`A skill named \`${fields.name}\` already exists on this ${owner.kind}.`);
        nameRef.current?.focus();
      } else setError(errorMessage(cause));
    } finally {
      if (generation.current === requestGeneration) setBusy(false);
    }
  };
  const toggle = async (item: InstructionSkill, enabled: boolean) => {
    if (busy || reorderTimer.current) return;
    const requestGeneration = generation.current;
    const current = item.id === selectedId ? { ...item, ...fields } : item;
    const nextTotal =
      total + (current.mode === 'always' ? (enabled ? 1 : -1) * current.content.length : 0);
    if (enabled && nextTotal > LIMITS.alwaysIncludeBudgetChars) {
      const message = budgetMessage(nextTotal);
      setBudgetError(message);
      toast.error(message);
      return;
    }
    setBusy(true);
    setBudgetError('');
    try {
      const saved = await api.updateInstructionSkill(owner, item.id, { enabled });
      publish(
        skills.map((entry) => (entry.id === item.id ? saved : entry)),
        requestGeneration,
      );
    } catch (cause) {
      if (generation.current === requestGeneration) toast.error(errorMessage(cause));
    } finally {
      if (generation.current === requestGeneration) setBusy(false);
    }
  };
  const move = (id: string, target: number) => {
    if (busy) return;
    const requestGeneration = generation.current;
    const index = skills.findIndex((item) => item.id === id);
    if (index < 0 || target < 0 || target >= skills.length || target === index) return;
    const next = [...skills];
    const [item] = next.splice(index, 1);
    next.splice(target, 0, item);
    setSkills(next);
    setAnnouncement(`Moved ${item.name} to position ${target + 1} of ${next.length}.`);
    requestAnimationFrame(() => {
      if (generation.current === requestGeneration) focusSkill(id);
    });
    if (reorderTimer.current) clearTimeout(reorderTimer.current);
    pendingReorder.current = () => void persistReorder();
    const persistReorder = async () => {
      if (generation.current === requestGeneration) setBusy(true);
      try {
        publish(
          await api.reorderInstructionSkills(
            owner,
            next.map((entry) => entry.id),
          ),
          requestGeneration,
        );
      } catch (cause) {
        if (generation.current !== requestGeneration) return;
        if (errorStatus(cause) === 400 || errorStatus(cause) === 409) {
          try {
            const refreshed = await api.listInstructionSkills(owner);
            if (generation.current !== requestGeneration) return;
            setSkills(refreshed);
          } catch {
            /* Preserve the local list on reload failure. */
          }
          if (generation.current !== requestGeneration) return;
          toast.error('The skill list changed elsewhere and was reloaded.');
        } else {
          setSkills(skills);
          toast.error(errorMessage(cause));
        }
      } finally {
        if (generation.current === requestGeneration) setBusy(false);
      }
    };
    reorderTimer.current = setTimeout(flushReorder, 400);
  };
  const remove = async () => {
    if (!selected || busy || reorderTimer.current) return;
    const requestGeneration = generation.current;
    setBusy(true);
    try {
      await api.deleteInstructionSkill(owner, selected.id);
      const index = skills.indexOf(selected);
      const next = skills.filter((item) => item.id !== selected.id);
      publish(next, requestGeneration);
      if (generation.current !== requestGeneration) return;
      setAnnouncement(`Deleted ${selected.name}.`);
      const replacement = next[Math.min(index, next.length - 1)];
      setSelectedId(replacement?.id ?? null);
      if (replacement) {
        setFields(replacement);
        setSnapshot(replacement);
      } else {
        setFields(emptyFields);
        setSnapshot(emptyFields);
      }
      requestAnimationFrame(() => {
        if (generation.current !== requestGeneration) return;
        if (replacement) focusSkill(replacement.id);
        else rootRef.current?.querySelector<HTMLElement>('#instruction-skills-new')?.focus();
      });
    } catch (cause) {
      if (generation.current === requestGeneration) toast.error(errorMessage(cause));
    } finally {
      if (generation.current === requestGeneration) setBusy(false);
    }
  };
  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.defaultPrevented) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      if (busy) return;
      if (pending) {
        pending.origin?.focus();
        setPending(null);
      } else navigate(onClose, true);
    } else if (event.key === 'Tab') {
      const controls = Array.from(
        rootRef.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled):not([type="file"]), textarea:not(:disabled), [tabindex="0"]',
        ) ?? [],
      ).filter((element) => element.getClientRects().length);
      if (!controls.length) return;
      if (event.shiftKey && document.activeElement === controls[0]) {
        event.preventDefault();
        controls[controls.length - 1]?.focus();
      } else if (!event.shiftKey && document.activeElement === controls[controls.length - 1]) {
        event.preventDefault();
        controls[0].focus();
      }
    }
  };
  const limited = skills.length >= LIMITS.maxSkillsPerOwner;
  const uploadButton = (label = 'Upload') => (
    <button
      id="instruction-skills-upload"
      type="button"
      className="btn btn-sm btn-secondary"
      title={limited ? skillLimitTitle : 'Upload .md or .txt'}
      disabled={busy || loading || Boolean(loadError) || limited}
      onClick={() => inputRef.current?.click()}
    >
      <Upload size={14} aria-hidden="true" />
      {label}
    </button>
  );
  const newButton = (label = 'New') => (
    <button
      id="instruction-skills-new"
      type="button"
      className="btn btn-sm"
      title={limited ? skillLimitTitle : undefined}
      disabled={busy || loading || Boolean(loadError) || limited}
      onClick={() => navigate(() => newDraft())}
    >
      <Plus size={14} aria-hidden="true" />
      {label}
    </button>
  );
  const noItems = !loading && !loadError && !skills.length && !draft;
  const adminSkillNames = loading || loadError ? null : new Set(skills.map((item) => item.name));
  const builtinRows = builtins.map((item) => ({
    item,
    state: builtinSkillState(item, adminSkillNames),
  }));
  const builtinActiveCount = builtinRows.filter((row) => row.state === 'active').length;
  const builtinsOpen = builtinsExpanded ?? noItems;
  const builtinListOpen = builtinsOpen && builtins.length > 0;
  const builtinStatus = builtinsLoading
    ? 'Loading built-in skills…'
    : builtinsError
      ? 'Could not load built-in skills.'
      : '';
  const builtinSection = () => {
    if (owner.kind !== 'tool') return null;
    if (!builtinsLoading && !builtinsError && !builtins.length) return null;
    return (
      <section
        id="instruction-skills-builtin"
        className="instruction-skills-builtin"
        data-instruction-skills-builtin
      >
        <button
          id="instruction-skills-builtin-toggle"
          type="button"
          className="instruction-builtin-toggle"
          aria-expanded={builtinListOpen}
          aria-controls={builtinListOpen ? 'instruction-builtin-list' : undefined}
          disabled={!builtins.length}
          onClick={() => setBuiltinsExpanded(!builtinListOpen)}
        >
          <span>{builtins.length ? `Built in (${number(builtins.length)})` : 'Built in'}</span>
          {builtins.length > 0 && (
            <span className="instruction-builtin-summary">{number(builtinActiveCount)} active</span>
          )}
          {builtins.length > 0 &&
            (builtinListOpen ? (
              <ChevronUp size={14} aria-hidden="true" />
            ) : (
              <ChevronDown size={14} aria-hidden="true" />
            ))}
        </button>
        {builtinsLoading && (
          <p id="instruction-skills-builtin-loading" className="field-help">
            Loading built-in skills…
          </p>
        )}
        {builtinsError && (
          <p id="instruction-skills-builtin-error" className="field-help">
            Could not load built-in skills.{' '}
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={() => {
                setBuiltinsError(false);
                setBuiltinsRetry((count) => count + 1);
              }}
            >
              Retry
            </button>
          </p>
        )}
        {builtinListOpen && (
          <>
            <p id="instruction-skills-builtin-help" className="field-help">
              Ship with Ragtime. You can&apos;t edit or turn them off. To change one, create a skill
              with the same name; yours replaces it.
            </p>
            <ul
              id="instruction-builtin-list"
              className="instruction-builtin-list"
              aria-label="Built-in skills"
            >
              {builtinRows.map(({ item, state }) => {
                const adminSkill = skills.find((entry) => entry.name === item.name);
                return (
                  <li
                    key={item.name}
                    className="instruction-builtin-row"
                    data-builtin-skill={item.name}
                    data-builtin-state={state}
                  >
                    <div className="instruction-builtin-head">
                      <span className="instruction-skill-row-name">{item.name}</span>
                      <span
                        className={`tool-badge ${item.mode === 'always' ? 'instruction-skill-badge-always' : ''}`}
                      >
                        {item.mode === 'always' ? 'Always' : 'On demand'}
                      </span>
                    </div>
                    <span className="instruction-builtin-desc" title={item.description}>
                      {item.description}
                    </span>
                    <div className="instruction-builtin-status">
                      {state === 'replaced' && adminSkill ? (
                        <button
                          type="button"
                          className="instruction-builtin-link"
                          data-builtin-replaced-by={item.name}
                          onClick={() => select(adminSkill)}
                        >
                          Replaced by your skill
                        </button>
                      ) : state === 'replaced' ? (
                        <span>Replaced by your skill</span>
                      ) : state === 'active' ? (
                        <span className="tool-badge">Active</span>
                      ) : (
                        <>
                          <span className="tool-badge">Not active</span>
                          {item.reason && <span>{item.reason}</span>}
                        </>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </section>
    );
  };

  return (
    <div
      id="instruction-skills-dialog-overlay"
      className="modal-overlay"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) navigate(onClose, true);
      }}
    >
      <div
        id="instruction-skills-dialog"
        ref={rootRef}
        className="modal-content modal-large instruction-skills-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="instruction-skills-dialog-title"
        tabIndex={-1}
        onKeyDown={handleKeyDown}
      >
        <div className="modal-header">
          <h3 id="instruction-skills-dialog-title">
            Skills{' '}
            <span className="instruction-skills-owner" title={ownerName}>
              · {ownerName}
            </span>
          </h3>
          <button
            type="button"
            className="modal-close"
            aria-label="Close skills dialog"
            onClick={() => navigate(onClose, true)}
          >
            <X size={18} />
          </button>
        </div>
        <input
          id="instruction-skills-file-input"
          ref={inputRef}
          type="file"
          accept=".md,.txt,text/markdown,text/plain"
          aria-hidden="true"
          tabIndex={-1}
          hidden
          onChange={(event) => void upload(event)}
        />
        <div
          id="instruction-skills-body"
          className="instruction-skills-body"
          data-empty={noItems || undefined}
        >
          {noItems ? (
            <div id="instruction-skills-empty" className="empty-state instruction-skills-empty">
              <h4>No skills yet</h4>
              <p>{emptyHelp}</p>
              <div>
                {newButton('New skill')} {uploadButton('Upload .md or .txt')}
              </div>
              {builtinSection()}
            </div>
          ) : (
            <>
              <div id="instruction-skills-rail" className="instruction-skills-rail">
                <div id="instruction-skills-rail-header" className="instruction-skills-rail-header">
                  <h4 className="instruction-skills-rail-heading">Skills</h4>
                  {uploadButton()}
                  {newButton()}
                </div>
                <ol
                  id="instruction-skills-list"
                  className="instruction-skills-list"
                  aria-label="Skills"
                >
                  {loading && (
                    <li id="instruction-skills-loading" role="status" className="field-help">
                      Loading skills…
                    </li>
                  )}
                  {skills.map((item, index) => (
                    <li
                      key={item.id}
                      className="instruction-skill-row"
                      data-instruction-skill-id={item.id}
                      data-enabled={item.enabled}
                      aria-busy={busy || undefined}
                      data-selected={item.id === selectedId || undefined}
                      data-drop-position={drop?.id === item.id ? drop.position : undefined}
                      draggable={dragId === item.id}
                      onDragStart={(event) => {
                        event.dataTransfer.setData('application/x-ragtime-skill', item.id);
                      }}
                      onDragEnd={() => {
                        setDragId(null);
                        setDrop(null);
                      }}
                      onDragOver={(event) => {
                        if (!event.dataTransfer.types.includes('application/x-ragtime-skill'))
                          return;
                        event.preventDefault();
                        setDrop({
                          id: item.id,
                          position:
                            event.clientY <
                            event.currentTarget.getBoundingClientRect().top +
                              event.currentTarget.offsetHeight / 2
                              ? 'before'
                              : 'after',
                        });
                      }}
                      onDrop={(event) => {
                        event.preventDefault();
                        const source = event.dataTransfer.getData('application/x-ragtime-skill');
                        const from = skills.findIndex((entry) => entry.id === source);
                        const insertion = index + (drop?.position === 'after' ? 1 : 0);
                        move(source, insertion - (from < insertion ? 1 : 0));
                        setDrop(null);
                        setDragId(null);
                      }}
                    >
                      <span
                        className="instruction-skill-row-grip"
                        data-instruction-skill-grip
                        aria-hidden="true"
                        title="Drag to reorder"
                        onMouseDown={() => setDragId(item.id)}
                        onMouseUp={() => setDragId(null)}
                      >
                        <GripVertical size={14} />
                      </span>
                      <button
                        type="button"
                        className="instruction-skill-row-main"
                        data-instruction-skill-select
                        aria-current={item.id === selectedId ? 'true' : undefined}
                        aria-keyshortcuts="Alt+ArrowUp Alt+ArrowDown"
                        aria-describedby="instruction-skills-reorder-hint"
                        title={item.name}
                        disabled={busy}
                        onClick={() => select(item)}
                        onKeyDown={(event) => {
                          if (
                            event.altKey &&
                            (event.key === 'ArrowUp' || event.key === 'ArrowDown')
                          ) {
                            event.preventDefault();
                            move(item.id, index + (event.key === 'ArrowUp' ? -1 : 1));
                          }
                        }}
                      >
                        <span className="instruction-skill-row-name">{item.name}</span>
                        <span className="instruction-skill-row-meta">
                          <span
                            className={`tool-badge ${item.mode === 'always' ? 'instruction-skill-badge-always' : ''}`}
                          >
                            {item.mode === 'always' ? 'Always' : 'On demand'}
                          </span>
                          <span>
                            {item.enabled
                              ? `≈ ${number(estimateTokens(item.content))} tokens`
                              : 'Off'}
                          </span>
                        </span>
                      </button>
                      {item.id === selectedId && (
                        <span className="instruction-skill-move">
                          <button
                            type="button"
                            className="btn btn-secondary"
                            data-instruction-skill-move="up"
                            aria-label="Move up"
                            disabled={busy || index === 0}
                            onClick={() => move(item.id, index - 1)}
                          >
                            <ChevronUp size={16} />
                          </button>
                          <button
                            type="button"
                            className="btn btn-secondary"
                            data-instruction-skill-move="down"
                            aria-label="Move down"
                            disabled={busy || index === skills.length - 1}
                            onClick={() => move(item.id, index + 1)}
                          >
                            <ChevronDown size={16} />
                          </button>
                        </span>
                      )}
                      <label className="toggle-switch instruction-skill-row-toggle">
                        <input
                          data-instruction-skill-toggle
                          type="checkbox"
                          aria-label={`Enable skill ${item.name}`}
                          checked={item.enabled}
                          disabled={busy}
                          onChange={(event) => void toggle(item, event.target.checked)}
                        />
                        <span className="toggle-slider" />
                      </label>
                    </li>
                  ))}
                  {draft && (
                    <li
                      className="instruction-skill-row"
                      data-instruction-skill-draft
                      data-selected="true"
                    >
                      <button
                        type="button"
                        className="instruction-skill-row-main"
                        data-instruction-skill-select
                        aria-current="true"
                      >
                        <span className="instruction-skill-row-name">
                          {fields.name || 'untitled-skill'}
                        </span>
                        <span className="instruction-skill-row-meta">
                          <span
                            className={`tool-badge ${fields.mode === 'always' ? 'instruction-skill-badge-always' : ''}`}
                          >
                            {fields.mode === 'always' ? 'Always' : 'On demand'}
                          </span>
                          <span>
                            {draftEnabled
                              ? `≈ ${number(estimateTokens(fields.content))} tokens`
                              : 'Off'}
                          </span>
                        </span>
                      </button>
                      <label className="toggle-switch instruction-skill-row-toggle">
                        <input
                          data-instruction-skill-toggle
                          type="checkbox"
                          aria-label={`Enable skill ${fields.name || 'draft'}`}
                          checked={draftEnabled}
                          onChange={(event) => setDraftEnabled(event.target.checked)}
                        />
                        <span className="toggle-slider" />
                      </label>
                    </li>
                  )}
                </ol>
                {builtinSection()}
                <div id="instruction-skills-budget" className="instruction-skills-budget">
                  Sent on every call: ≈ {number(alwaysTokens)} tokens ({number(total)} of{' '}
                  {number(LIMITS.alwaysIncludeBudgetChars)} characters)
                  {builtins.length > 0 && (
                    <p id="instruction-skills-budget-builtin-note" className="field-help">
                      Built-in skills aren&apos;t counted here.
                    </p>
                  )}
                  <div
                    id="instruction-skills-budget-meter"
                    className="instruction-skills-meter"
                    data-level={
                      overBudget
                        ? 'over'
                        : total >= LIMITS.alwaysIncludeBudgetChars * NEAR_BUDGET_RATIO
                          ? 'near'
                          : 'ok'
                    }
                    aria-hidden="true"
                  >
                    <span
                      style={{
                        width: `${Math.min(100, (total / LIMITS.alwaysIncludeBudgetChars) * 100)}%`,
                      }}
                    />
                  </div>
                  {(budgetError || overBudget) && (
                    <p role="alert">{budgetError || budgetMessage(total)}</p>
                  )}
                </div>
              </div>
              <div
                id="instruction-skill-editor"
                className="instruction-skills-detail"
                aria-busy={busy || loading}
              >
                {loadError && (
                  <div id="instruction-skills-load-error" className="error-banner" role="alert">
                    Could not load skills: {loadError}{' '}
                    <button
                      type="button"
                      className="btn btn-sm btn-secondary"
                      onClick={() => setRetry(retry + 1)}
                    >
                      Retry
                    </button>
                  </div>
                )}
                {(selected || draft) && (
                  <>
                    <div className="form-group">
                      <div className="instruction-skill-label-row">
                        <label htmlFor="instruction-skill-name">Name</label>
                        <span>lowercase letters, digits, hyphens</span>
                      </div>
                      <input
                        ref={nameRef}
                        id="instruction-skill-name"
                        className="instruction-skills-mono"
                        type="text"
                        value={fields.name}
                        maxLength={LIMITS.nameMaxLength}
                        spellCheck={false}
                        autoCapitalize="off"
                        autoComplete="off"
                        aria-invalid={Boolean(
                          nameError || duplicate || (fields.name && invalidName),
                        )}
                        aria-describedby="instruction-skill-name-error"
                        onChange={(event) => {
                          const start = event.target.selectionStart;
                          const end = event.target.selectionEnd;
                          setFields({ ...fields, name: sanitize(event.target.value) });
                          setNameError('');
                          requestAnimationFrame(() => {
                            if (start !== null && end !== null)
                              nameRef.current?.setSelectionRange(start, end);
                          });
                        }}
                        onBlur={() =>
                          setFields((current) => ({ ...current, name: normalize(current.name) }))
                        }
                      />
                      <p id="instruction-skill-name-error" className="field-help">
                        {nameError ||
                          (duplicate
                            ? `A skill named \`${fields.name}\` already exists on this ${owner.kind}.`
                            : fields.name && invalidName
                              ? 'Use lowercase letters, digits and single hyphens.'
                              : '')}
                      </p>
                      {builtins.some((item) => item.name === fields.name) && (
                        <p id="instruction-skill-name-builtin-note" className="field-help">
                          Replaces the built-in skill of the same name.
                        </p>
                      )}
                    </div>
                    <div className="form-group">
                      <div className="instruction-skill-label-row">
                        <label htmlFor="instruction-skill-description">Description</label>
                        <span
                          id="instruction-skill-description-count"
                          data-over={
                            fields.description.length > LIMITS.descriptionMaxLength || undefined
                          }
                        >
                          {number(fields.description.length)} /{' '}
                          {number(LIMITS.descriptionMaxLength)}
                          {fields.description.length > LIMITS.descriptionMaxLength &&
                            ` (over by ${number(fields.description.length - LIMITS.descriptionMaxLength)})`}
                        </span>
                      </div>
                      <textarea
                        id="instruction-skill-description"
                        rows={2}
                        value={fields.description}
                        aria-describedby="instruction-skill-description-count instruction-skill-description-error"
                        aria-invalid={
                          fields.description.length > LIMITS.descriptionMaxLength ||
                          (descriptionTouched && !fields.description.trim())
                        }
                        onBlur={() => setDescriptionTouched(true)}
                        onChange={(event) =>
                          setFields({ ...fields, description: event.target.value })
                        }
                      />
                      <p className="field-help">
                        What the skill covers and when to use it. The model always sees this line.
                      </p>
                      <p
                        id="instruction-skill-description-error"
                        className="instruction-skill-field-error"
                      >
                        {descriptionTouched && !fields.description.trim()
                          ? 'Description is required.'
                          : ''}
                      </p>
                    </div>
                    <div className="form-group">
                      <div id="instruction-skill-mode-label">Delivery</div>
                      <div
                        id="instruction-skill-mode"
                        className="appearance-mode-toggle"
                        role="group"
                        aria-labelledby="instruction-skill-mode-label"
                        aria-describedby="instruction-skill-mode-help"
                      >
                        {(['always', 'on_demand'] as InstructionSkillMode[]).map((mode) => (
                          <button
                            key={mode}
                            type="button"
                            className="appearance-mode-option"
                            aria-pressed={fields.mode === mode}
                            onClick={() => setFields({ ...fields, mode })}
                          >
                            {mode === 'always' ? 'Always include' : 'Load on demand'}
                          </button>
                        ))}
                      </div>
                      <p id="instruction-skill-mode-help" className="field-help">
                        {fields.mode === 'always'
                          ? alwaysHelp
                          : 'Only the name and description are sent. The model loads the full text when the task calls for it.'}
                      </p>
                    </div>
                    <div className="form-group">
                      <div className="instruction-skill-label-row">
                        <label htmlFor="instruction-skill-content">Instructions (Markdown)</label>
                        <span
                          id="instruction-skill-content-count"
                          data-over={fields.content.length > LIMITS.contentMaxLength || undefined}
                        >
                          ≈ {number(estimateTokens(fields.content))} tokens{' '}
                          <span>
                            {number(fields.content.length)} / {number(LIMITS.contentMaxLength)}
                            {fields.content.length > LIMITS.contentMaxLength &&
                              ` (over by ${number(fields.content.length - LIMITS.contentMaxLength)})`}
                          </span>
                        </span>
                      </div>
                      <textarea
                        id="instruction-skill-content"
                        className="instruction-skills-mono"
                        rows={9}
                        value={fields.content}
                        aria-describedby="instruction-skill-content-count"
                        aria-invalid={fields.content.length > LIMITS.contentMaxLength}
                        onChange={(event) => setFields({ ...fields, content: event.target.value })}
                      />
                    </div>
                    {error && (
                      <div id="instruction-skill-error" className="error-banner" role="alert">
                        {error}
                      </div>
                    )}
                  </>
                )}
              </div>
            </>
          )}
        </div>
        {pending && (
          <div
            id="instruction-skills-unsaved-confirm"
            data-instruction-skills-dirty-confirm
            className="tool-access-callout tool-access-callout-warning"
            role="alertdialog"
            tabIndex={-1}
          >
            <p>You have unsaved changes. Discard them?</p>
            <button
              type="button"
              className="btn btn-sm btn-danger"
              onClick={() => {
                const action = pending.action;
                setPending(null);
                action();
              }}
            >
              {pending.close ? 'Discard and close' : 'Discard and continue'}
            </button>
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={() => {
                pending.origin?.focus();
                setPending(null);
              }}
            >
              Keep editing
            </button>
          </div>
        )}
        <div id="instruction-skills-footer" className="modal-footer instruction-skills-footer">
          <div>
            {selected && (
              <DeleteConfirmButton
                id="instruction-skill-delete"
                className="btn btn-sm btn-secondary"
                buttonText="Delete skill"
                title="Delete skill"
                icon={<Trash2 size={14} aria-hidden="true" />}
                disabled={busy}
                deleting={busy}
                onDelete={() => void remove()}
              />
            )}
            {draft && (
              <button type="button" className="btn btn-sm btn-secondary" onClick={discardDraft}>
                Discard draft
              </button>
            )}
          </div>
          <div className="instruction-skills-footer-actions">
            {dirty && (
              <span id="instruction-skill-dirty" role="status">
                Unsaved changes
              </span>
            )}
            {dirty && reason && <span className="field-help">{reason}</span>}
            <button
              id="instruction-skills-close"
              type="button"
              className="btn btn-secondary"
              onClick={() => navigate(onClose, true)}
            >
              Close
            </button>
            {(selected || draft) && (
              <button
                id="instruction-skill-save"
                type="button"
                className="btn"
                disabled={!dirty || Boolean(reason) || overBudget || busy}
                onClick={() => void save()}
              >
                Save skill
              </button>
            )}
          </div>
        </div>
        <span id="instruction-skills-reorder-hint" className="instruction-skills-visually-hidden">
          Press Alt and an arrow key to reorder.
        </span>
        <span
          id="instruction-skills-announcer"
          className="instruction-skills-visually-hidden"
          role="status"
        >
          {announcement}
        </span>
        {/* Mounted empty with the dialog; the built-in status fills in after the fetch starts. */}
        <span
          id="instruction-skills-builtin-status"
          className="instruction-skills-visually-hidden"
          role="status"
        >
          {owner.kind === 'tool' ? builtinStatus : ''}
        </span>
        <ToastContainer toasts={toasts} onDismiss={toast.dismiss} />
      </div>
    </div>
  );
}
