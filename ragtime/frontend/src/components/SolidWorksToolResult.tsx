import { Fragment, useState, type ReactNode } from 'react';
import { AlertCircle, Box, Check, Copy, Info, LoaderCircle } from 'lucide-react';
import { formatArtifactSize } from '@/utils/modelArtifact';
import type { ParsedSolidWorksResult } from '@/utils/solidworksToolResult';

type RecordValue = Record<string, unknown>;

const record = (value: unknown): RecordValue =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as RecordValue) : {};
const label = (key: string) =>
  key.replace(/_/g, ' ').replace(/^\w/, (letter) => letter.toUpperCase());
const text = (value: unknown) =>
  value == null ? '—' : typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value);
const records = (value: unknown): RecordValue[] =>
  Array.isArray(value) ? value.map(record).filter((item) => Object.keys(item).length > 0) : [];
const pill = (value: ReactNode, tone: 'muted' | 'error' | 'warning' | 'success' = 'muted') => (
  <span
    className={`badge ${{ muted: 'badge-muted', error: 'failed', warning: 'solidworks-result-badge-warning', success: 'completed' }[tone]}`}
  >
    {value}
  </span>
);

function CopyButton({ value, name }: { value: string; name: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };
  return (
    <button
      className="tool-call-copy-btn"
      onClick={copy}
      title={`Copy ${name}`}
      aria-label={copied ? `Copied ${name}` : `Copy ${name}`}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />}
    </button>
  );
}

function OmissionText({ value }: { value: unknown }) {
  return (
    <>
      {text(value)
        .split(/\.\.\.\s*\[([\d,]+) characters omitted\]\s*\.\.\./g)
        .map((part, index) =>
          index % 2 ? (
            <span className="solidworks-result-omitted" role="note" key={index}>
              {part} characters omitted
            </span>
          ) : (
            part
          ),
        )}
    </>
  );
}

function uptime(seconds: number): string {
  const minutes = Math.floor(Math.max(0, seconds) / 60);
  const hours = Math.floor(minutes / 60);
  return hours ? `${hours} h ${minutes % 60} min` : `${minutes} min`;
}

function Status({ result }: { result: RecordValue }) {
  const account = record(result.account);
  const session = record(result.session);
  const solidworks = record(result.solidworks);
  const instances = records(solidworks.instances);
  const pdm = record(result.pdm);
  const views = records(pdm.views ?? result.pdm_views ?? result.views);
  return (
    <div data-solidworks-status>
      <dl className="solidworks-result-facts">
        {result.host_version != null && (
          <>
            <dt>Agent version</dt>
            <dd>{text(result.host_version)}</dd>
          </>
        )}
        {result.protocol != null && (
          <>
            <dt>Protocol</dt>
            <dd>{text(result.protocol)}</dd>
          </>
        )}
        {account.name != null && (
          <>
            <dt>Account</dt>
            <dd className="solidworks-result-meta">
              {text(account.name)}
              {account.is_administrator === true && pill('Administrator')}
            </dd>
          </>
        )}
        {session.interactive != null && (
          <>
            <dt>Session</dt>
            <dd>{pill(session.interactive === true ? 'Interactive' : 'Not interactive')}</dd>
          </>
        )}
        {typeof result.uptime_seconds === 'number' && (
          <>
            <dt>Uptime</dt>
            <dd>{uptime(result.uptime_seconds)}</dd>
          </>
        )}
      </dl>
      {Array.isArray(solidworks.installed_versions) && (
        <div className="solidworks-result-meta">
          {solidworks.installed_versions.map((version) => (
            <Fragment key={text(version)}>{pill(text(version))}</Fragment>
          ))}
        </div>
      )}
      {instances.length > 0 && (
        <div className="solidworks-result-table-wrap" data-solidworks-instances>
          <table className="tool-result-table">
            <thead>
              <tr>
                <th>Id</th>
                <th>Mode</th>
                <th>Version</th>
                <th>Revision</th>
                <th>Documents</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {instances.map((instance, index) => (
                <tr key={`${text(instance.id ?? instance.instance_id)}-${index}`}>
                  <td>{text(instance.id ?? instance.instance_id)}</td>
                  <td>{text(instance.mode)}</td>
                  <td>{text(instance.version)}</td>
                  <td>{text(instance.revision)}</td>
                  <td>{text(instance.documents ?? instance.document_count)}</td>
                  <td>{instance.busy === true && pill('Busy')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {views.length > 0 && (
        <div className="solidworks-result-table-wrap" data-solidworks-pdm-views>
          <table className="tool-result-table">
            <thead>
              <tr>
                <th>Vault</th>
                <th>Root</th>
                <th>Login</th>
                <th>User</th>
              </tr>
            </thead>
            <tbody>
              {views.map((view, index) => (
                <tr key={`${text(view.vault ?? view.name)}-${index}`}>
                  <td>{text(view.vault ?? view.name)}</td>
                  <td>
                    <code>{text(view.root ?? view.root_path)}</code>
                  </td>
                  <td>{view.logged_in === true && pill('Logged in')}</td>
                  <td>{text(view.user ?? view.username)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Facts({ value, depth = 0 }: { value: RecordValue; depth?: number }) {
  return (
    <dl className="solidworks-result-facts">
      {Object.entries(value).map(([key, item]) => (
        <Fragment key={key}>
          <dt>{label(key)}</dt>
          <dd>
            {Array.isArray(item) ? (
              item.every((entry) => entry == null || typeof entry !== 'object') ? (
                <span className="solidworks-result-meta">
                  {item.map((entry, index) => (
                    <Fragment key={`${text(entry)}-${index}`}>{pill(text(entry))}</Fragment>
                  ))}
                </span>
              ) : (
                `${item.length} items`
              )
            ) : item && typeof item === 'object' ? (
              depth < 2 ? (
                <Facts value={record(item)} depth={depth + 1} />
              ) : (
                `${Object.keys(record(item)).length} fields`
              )
            ) : key.includes('path') || key === 'id' ? (
              <code title={text(item)}>{text(item)}</code>
            ) : (
              text(item)
            )}
          </dd>
        </Fragment>
      ))}
    </dl>
  );
}

const columns: Record<string, string[]> = {
  'documents.list': ['title', 'type', 'configuration', 'state'],
  'components.list': ['name', 'configuration', 'state', 'children'],
  'features.list': ['name', 'type', 'suppressed'],
  'configurations.list': ['name', 'description', 'active'],
  'dimensions.list': ['name', 'value', 'unit'],
  'mates.list': ['name', 'type', 'suppressed'],
  'equations.list': ['name', 'equation', 'value'],
};

function StatePills({ item }: { item: RecordValue }) {
  const states: [string, string][] = [
    ['read_only', 'Read-only'],
    ['modified', 'Modified'],
    ['opened_by_agent', 'Opened by agent'],
    ['opened_by_host', 'Opened by host'],
    ['suppressed', 'Suppressed'],
    ['hidden', 'Hidden'],
    ['fixed', 'Fixed'],
    ['virtual', 'Virtual'],
    ['is_virtual', 'Virtual'],
  ];
  return (
    <span className="solidworks-result-meta">
      {states
        .filter(([key]) => item[key] === true)
        .map(([key, name]) => (
          <Fragment key={key}>{pill(name)}</Fragment>
        ))}
    </span>
  );
}

function ListResult({ operation, result }: { operation: string; result: RecordValue }) {
  const listKey = Object.keys(result).find((key) => Array.isArray(result[key]));
  const items = records(listKey ? result[listKey] : []);
  const keys =
    columns[operation] ??
    Object.keys(items[0] ?? {})
      .filter((key) => items[0][key] == null || typeof items[0][key] !== 'object')
      .slice(0, 4);
  const offset = typeof result.offset === 'number' ? result.offset : 0;
  const total = typeof result.total === 'number' ? result.total : items.length;
  return (
    <div data-solidworks-list={operation}>
      {items.length ? (
        <div className="solidworks-result-table-wrap">
          <table
            className={`tool-result-table${operation === 'documents.list' ? ' solidworks-result-documents' : ''}`}
          >
            <thead>
              <tr>
                {keys.map((key) => (
                  <th key={key}>{label(key)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {items.map((item, index) => (
                <tr key={`${String(item.id ?? item.name ?? item.title ?? '')}-${index}`}>
                  {keys.map((key) => (
                    <td key={key}>
                      {key === 'state' ? (
                        <StatePills item={item} />
                      ) : key === 'type' ? (
                        pill(label(text(item[key])))
                      ) : (
                        <span
                          style={
                            key === 'name' && typeof item.depth === 'number'
                              ? { paddingInlineStart: `${Math.max(0, item.depth) * 12}px` }
                              : undefined
                          }
                        >
                          {text(
                            key === 'configuration'
                              ? (item.configuration ?? item.active_configuration)
                              : key === 'children'
                                ? (item.children ?? item.children_count)
                                : item[key],
                          )}
                        </span>
                      )}
                      {key === 'title' && typeof item.path === 'string' && (
                        <span className="solidworks-result-path" title={item.path}>
                          {item.path}
                        </span>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p>No items.</p>
      )}
      <p className="solidworks-result-paging">
        <span>
          Showing {items.length ? offset + 1 : 0}–{offset + items.length} of {total}
        </span>
        {result.next_offset != null && pill(`More: next offset ${text(result.next_offset)}`)}
        {result.units != null && (
          <span>
            Units:{' '}
            {typeof result.units === 'object'
              ? Object.values(record(result.units)).join(', ')
              : text(result.units)}
          </span>
        )}
      </p>
    </div>
  );
}

function Guide({
  result,
  requestedTopic,
  renderMarkdown,
}: {
  result: RecordValue;
  requestedTopic: string | undefined;
  renderMarkdown: (markdown: string) => ReactNode;
}) {
  const topics = records(result.topics);
  const parts =
    typeof result.markdown === 'string'
      ? result.markdown.split(/\.\.\.\s*\[([\d,]+) characters omitted\]\s*\.\.\./g)
      : [];
  const markdown = parts.length > 0 && (
    <div className="solidworks-result-markdown">
      {parts.map((part, index) =>
        index % 2 ? (
          <p className="solidworks-result-omitted" role="note" key={index}>
            {part} characters omitted
          </p>
        ) : (
          <Fragment key={index}>{renderMarkdown(part)}</Fragment>
        ),
      )}
    </div>
  );
  return (
    <div data-solidworks-guide>
      {requestedTopic ? (
        <>
          {markdown}
          {topics.length > 0 && (
            <div className="solidworks-result-meta" data-solidworks-other-topics>
              <span>Other topics</span>
              {topics
                .filter((topic) => topic.id !== requestedTopic)
                .map((topic) => (
                  <code
                    className="badge badge-muted"
                    key={text(topic.id)}
                    title={text(topic.title)}
                  >
                    <OmissionText value={topic.id} />
                  </code>
                ))}
            </div>
          )}
        </>
      ) : (
        <>
          <ul className="solidworks-result-topics">
            {topics.map((topic) => (
              <li key={text(topic.id)}>
                <div>
                  <code>
                    <OmissionText value={topic.id} />
                  </code>{' '}
                  <strong>
                    <OmissionText value={topic.title} />
                  </strong>
                </div>
                {topic.when_to_read != null && (
                  <span className="solidworks-result-topic-description">
                    <OmissionText value={topic.when_to_read} />
                  </span>
                )}
              </li>
            ))}
          </ul>
          {markdown && (
            <>
              <p className="solidworks-result-topic-description">
                {label(text(result.topic ?? 'overview'))}
              </p>
              {markdown}
            </>
          )}
        </>
      )}
    </div>
  );
}

function Describe({ result, truncated }: { result: RecordValue; truncated: boolean }) {
  const items = records(result.operations);
  const groups = new Map<string, RecordValue[]>([['', []]]);
  items.forEach((item) => {
    const name = text(item.name);
    const group = name.includes('.') ? name.split('.')[0] : '';
    groups.set(group, [...(groups.get(group) ?? []), item]);
  });
  return (
    <div data-solidworks-describe>
      <p>{items.length} operations</p>
      <div className="solidworks-result-describe">
        {Array.from(groups, ([group, rows]) => (
          <section key={group} data-operation-group={group}>
            {group && <strong>{group}</strong>}
            {rows.map((item) => (
              <div className="solidworks-result-operation" key={text(item.name)}>
                <div className="solidworks-result-meta">
                  <code>{text(item.name)}</code>
                  {item.access != null &&
                    pill(
                      text(item.access),
                      ['write', 'script'].includes(String(item.access)) ? 'warning' : 'muted',
                    )}
                  {item.paged === true && pill('Paged')}
                </div>
                <span className="solidworks-result-operation-summary">
                  {text(item.summary ?? item.description)}
                </span>
              </div>
            ))}
          </section>
        ))}
      </div>
      {truncated && <p>Showing the first {items.length}; output was truncated.</p>}
    </div>
  );
}

function ResultBody({
  parsed,
  artifactShown,
  renderMarkdown,
}: {
  parsed: ParsedSolidWorksResult;
  artifactShown: boolean;
  renderMarkdown: (markdown: string) => ReactNode;
}) {
  const result = parsed.result;
  if (!result) return null;
  if (parsed.operation === 'guide')
    return (
      <Guide
        result={result}
        requestedTopic={parsed.args.find((arg) => arg.key === 'topic')?.value}
        renderMarkdown={renderMarkdown}
      />
    );
  if (parsed.operation === 'describe')
    return <Describe result={result} truncated={Boolean(parsed.truncated)} />;
  if (parsed.operation.endsWith('.list'))
    return <ListResult operation={parsed.operation} result={result} />;
  if (parsed.operation === 'status') return <Status result={result} />;
  if (parsed.operation === 'document.open' || parsed.operation === 'document.close') {
    return (
      <div data-solidworks-document>
        <div className="solidworks-result-meta">
          {result.title != null && <strong>{text(result.title)}</strong>}
          {result.type != null && pill(text(result.type))}
          {result.active_configuration != null && <span>{text(result.active_configuration)}</span>}
          <StatePills item={result} />
          {result.modified === false && pill('Unmodified')}
        </div>
        <dl className="solidworks-result-facts">
          {result.document_id != null && (
            <>
              <dt>Id</dt>
              <dd>
                <code>{text(result.document_id)}</code>
              </dd>
            </>
          )}
          {typeof result.path === 'string' && (
            <>
              <dt>Path</dt>
              <dd>
                <code title={result.path}>{result.path}</code>{' '}
                <CopyButton name="path" value={result.path} />
              </dd>
            </>
          )}
        </dl>
      </div>
    );
  }
  if (parsed.operation === 'model.export' || parsed.operation === 'model.preview') {
    const artifact = record(result.artifact);
    const data = Object.keys(artifact).length ? artifact : result;
    return (
      <div data-solidworks-artifact-summary>
        <div className="solidworks-result-meta">
          <strong>{text(data.file_name ?? data.filename)}</strong>
          {data.format != null && pill(text(data.format).toUpperCase())}
          {typeof (data.size_bytes ?? data.size) === 'number' && (
            <span>{formatArtifactSize((data.size_bytes ?? data.size) as number)}</span>
          )}
          {artifactShown && <span>Shown in the card below</span>}
        </div>
        {Array.isArray(result.warnings) &&
          result.warnings.map((warning) => (
            <p className="solidworks-result-note" key={text(warning)}>
              <Info size={12} />
              {text(warning)}
            </p>
          ))}
      </div>
    );
  }
  return (
    <div data-solidworks-facts>
      <Facts value={result} />
    </div>
  );
}

export function SolidWorksToolResult({
  parsed,
  rawOutput,
  resultId,
  maskText,
  artifactShown,
  renderMarkdown,
}: {
  parsed: ParsedSolidWorksResult;
  rawOutput: string;
  resultId: string;
  maskText?: (text: string) => string;
  artifactShown: boolean;
  renderMarkdown: (markdown: string) => ReactNode;
}) {
  return (
    <div
      className="tool-call-section solidworks-result"
      id={resultId}
      data-solidworks-result
      data-operation={parsed.operation}
      data-state={parsed.state}
    >
      <div className="tool-call-section-header">
        <span className="tool-call-section-label">
          <Box size={12} />
          SolidWorks
        </span>
        <CopyButton value={rawOutput} name="JSON" />
      </div>
      {parsed.state === 'unparsed' ? (
        <p>
          {rawOutput.trim() ? 'Unable to parse the SolidWorks result.' : 'No output was recorded.'}
        </p>
      ) : (
        <>
          <div className="solidworks-result-meta">
            {parsed.state !== 'error' &&
              pill(
                <>
                  {parsed.state === 'running' ? <LoaderCircle size={12} /> : <Check size={12} />}
                  {parsed.state === 'running' ? 'Running' : 'Completed'}
                </>,
                parsed.state === 'ok' ? 'success' : 'muted',
              )}
            {parsed.solidworksVersion && pill(`SolidWorks ${parsed.solidworksVersion}`)}
            {parsed.elapsedMs != null && parsed.elapsedMs > 0 && (
              <span>
                {parsed.elapsedMs >= 1000
                  ? `${(parsed.elapsedMs / 1000).toFixed(1)} s`
                  : `${parsed.elapsedMs} ms`}
              </span>
            )}
            {parsed.truncated && pill('Output truncated')}
          </div>
          {parsed.args.length ? (
            <dl className="solidworks-result-args">
              {parsed.args.map((arg) => (
                <Fragment key={arg.key}>
                  <dt>{arg.label}</dt>
                  <dd>
                    {arg.kind === 'code' ? <code title={arg.value}>{arg.value}</code> : arg.value}
                  </dd>
                </Fragment>
              ))}
            </dl>
          ) : (
            <p>No arguments</p>
          )}
          {parsed.state === 'running' && (
            <p className="solidworks-result-note" role="status">
              <LoaderCircle size={12} />
              Waiting for SolidWorks…
            </p>
          )}
          {parsed.state === 'error' ? (
            <div className="solidworks-result-error" role="group" aria-label="Error">
              <AlertCircle size={14} />
              <div>
                {parsed.error?.code && pill(parsed.error.code, 'error')}{' '}
                <span>
                  {maskText
                    ? maskText(parsed.error?.message || 'SolidWorks reported a failure.')
                    : parsed.error?.message || 'SolidWorks reported a failure.'}
                </span>
                {parsed.error?.details.length ? (
                  <dl className="solidworks-result-facts">
                    {parsed.error.details.map((detail) => (
                      <Fragment key={detail.label}>
                        <dt>{detail.label}</dt>
                        <dd>{detail.value}</dd>
                      </Fragment>
                    ))}
                  </dl>
                ) : null}
              </div>
            </div>
          ) : (
            parsed.state === 'ok' && (
              <ResultBody
                parsed={parsed}
                artifactShown={artifactShown}
                renderMarkdown={renderMarkdown}
              />
            )
          )}
          {parsed.truncated && (
            <p className="solidworks-result-truncated">
              <Info size={12} />
              Output was truncated in chat history; showing what was recorded.
            </p>
          )}
        </>
      )}
    </div>
  );
}
