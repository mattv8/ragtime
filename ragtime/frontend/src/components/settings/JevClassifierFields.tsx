import { Loader2 } from 'lucide-react';
import { useState } from 'react';
import {
  detectJevKeyTransport,
  type ContentProtectionReadinessResult,
  type JevModel,
  type JevTransport,
} from '@/api/contentProtection';
import { FetchableModelField } from '../shared/FetchableModelField';

type KeySetting = 'typesafe_api_key' | 'openrouter_api_key';
interface JevClassifierFieldsProps {
  transport: JevTransport;
  model: string;
  onTransportChange: (transport: JevTransport) => void;
  onModelChange: (model: string) => void;
  draftKey: string;
  onDraftKeyChange: (key: string) => void;
  keyBusy: boolean;
  onSaveKey: (setting: KeySetting, key: string) => void;
  typesafeConfigured: boolean;
  openrouterConfigured: boolean;
  jevModels: JevModel[];
  jevModelsLoaded: boolean;
  jevModelsFetching: boolean;
  jevModelsError: string | null;
  onFetchModels: () => void;
  readiness: ContentProtectionReadinessResult | null;
  readinessBusy: boolean;
  onCheckReadiness: () => void;
  onOpenRouterLinkClick: () => void;
  disabled?: boolean;
}

function keyHelpText({
  hasDraftKey,
  configuredForTransport,
  unrecognizedAuto,
  unrecognizedOpenRouter,
  contradictory,
  detectedTransport,
}: {
  hasDraftKey: boolean;
  configuredForTransport: boolean;
  unrecognizedAuto: boolean;
  unrecognizedOpenRouter: boolean;
  contradictory: boolean;
  detectedTransport: Exclude<JevTransport, 'auto'> | null;
}): string {
  if (!hasDraftKey) {
    return configuredForTransport
      ? 'A key is configured. Enter a new value to replace it.'
      : 'No key configured. Enter a TypeSafe (ts_…) or OpenRouter (sk-or-…) key.';
  }
  if (unrecognizedAuto)
    return '⚠ Key prefix not recognized. Choose TypeSafe or OpenRouter to save.';
  if (unrecognizedOpenRouter) return '⚠ OpenRouter keys start with sk-or-.';
  if (contradictory) return '⚠ This key does not match the selected provider.';
  if (detectedTransport)
    return `Detected: ${detectedTransport === 'typesafe' ? 'TypeSafe' : 'OpenRouter'} key.`;
  return 'Will save as TypeSafe.';
}

export function JevClassifierFields(props: JevClassifierFieldsProps): JSX.Element {
  const [keyTouched, setKeyTouched] = useState(false);
  const detectedTransport = detectJevKeyTransport(props.draftKey);
  const hasDraftKey = Boolean(props.draftKey.trim());
  const contradictory = Boolean(
    detectedTransport && props.transport !== 'auto' && props.transport !== detectedTransport,
  );
  const unrecognizedAuto = hasDraftKey && props.transport === 'auto' && !detectedTransport;
  const unrecognizedOpenRouter =
    hasDraftKey && props.transport === 'openrouter' && !detectedTransport;
  const saveDisabled =
    props.keyBusy || !hasDraftKey || contradictory || unrecognizedAuto || unrecognizedOpenRouter;
  const configuredForTransport =
    (props.transport !== 'openrouter' && props.typesafeConfigured) ||
    (props.transport !== 'typesafe' && props.openrouterConfigured);
  const fetchDisabled =
    contradictory ||
    unrecognizedAuto ||
    unrecognizedOpenRouter ||
    (!hasDraftKey && !configuredForTransport);
  const effectiveTransport =
    detectedTransport || (props.transport === 'auto' ? null : props.transport);
  const keyError = unrecognizedAuto || unrecognizedOpenRouter || contradictory;
  const keyHelp = keyHelpText({
    hasDraftKey,
    configuredForTransport,
    unrecognizedAuto,
    unrecognizedOpenRouter,
    contradictory,
    detectedTransport,
  });
  return (
    <>
      <div className="form-group" id="jev-provider-row">
        <label htmlFor="jev-provider-select">Provider</label>
        <div className="input-with-button input-with-actions">
          <select
            id="jev-provider-select"
            value={props.transport}
            disabled={props.disabled}
            onChange={(event) => props.onTransportChange(event.target.value as JevTransport)}
          >
            <option value="auto">Auto (TypeSafe preferred)</option>
            <option value="typesafe">TypeSafe</option>
            <option value="openrouter">OpenRouter</option>
          </select>
        </div>
      </div>
      <div className="form-group" id="jev-api-key-row" data-jev-api-key-row>
        <label htmlFor="jev-api-key-input">API key</label>
        <div className="input-with-button">
          <input
            id="jev-api-key-input"
            type="password"
            autoComplete="new-password"
            maxLength={512}
            value={props.draftKey}
            disabled={props.disabled}
            placeholder={configuredForTransport ? '••••••••' : 'Enter key'}
            aria-describedby="jev-api-key-help"
            onChange={(event) => {
              const key = event.target.value;
              props.onDraftKeyChange(key);
              const detected = detectJevKeyTransport(key);
              if (detected && detected !== props.transport) props.onTransportChange(detected);
            }}
            onBlur={() => setKeyTouched(true)}
          />
          <button
            type="button"
            className="btn btn-secondary"
            disabled={props.disabled || saveDisabled}
            onClick={() =>
              effectiveTransport &&
              props.onSaveKey(
                effectiveTransport === 'typesafe' ? 'typesafe_api_key' : 'openrouter_api_key',
                props.draftKey,
              )
            }
          >
            {props.keyBusy ? 'Saving…' : 'Save key'}
          </button>
        </div>
        <p
          id="jev-api-key-help"
          className="field-help"
          role={keyError && keyTouched ? 'alert' : undefined}
        >
          {keyHelp}
        </p>
      </div>
      {hasDraftKey && effectiveTransport === 'openrouter' && (
        <p
          id="jev-openrouter-shared-warning"
          className="field-help"
          role="alert"
          style={{ color: 'var(--color-warning)' }}
        >
          ⚠ Saving this key will replace the shared OpenRouter key used by LLM Providers and
          Embedding Configuration.
        </p>
      )}
      <FetchableModelField
        idPrefix="jev-model"
        label="Model"
        value={props.model}
        onChange={props.onModelChange}
        onBlur={() => !props.model.trim() && props.onModelChange('jev-latest')}
        models={props.jevModels.map((model) => ({
          id: model.id,
          label: `${model.name || model.id}${model.release_date ? ` (${model.release_date})` : ''}`,
        }))}
        loaded={props.jevModelsLoaded}
        fetching={props.jevModelsFetching}
        error={props.jevModelsError}
        onFetch={props.onFetchModels}
        fetchDisabled={fetchDisabled}
        disabled={props.disabled}
        placeholder="jev-latest"
        help={
          props.jevModelsLoaded
            ? 'Select a Jev model. jev-latest tracks the current recommended release.'
            : 'jev-latest tracks the current recommended release. Pin a specific version to freeze behavior. Click "Fetch Models" to see available options.'
        }
      />
      {(props.transport === 'auto' || props.transport === 'openrouter') && (
        <div id="jev-openrouter-notice" data-jev-openrouter-notice className="form-group">
          <p className="field-help">
            {props.openrouterConfigured
              ? 'An OpenRouter key is configured (shared with LLM Providers).'
              : 'No OpenRouter key is configured.'}{' '}
            <a href="#setting-openrouter-api-key" onClick={props.onOpenRouterLinkClick}>
              Manage the OpenRouter key in provider settings.
            </a>
          </p>
        </div>
      )}
      <button
        type="button"
        className="btn btn-secondary"
        disabled={props.readinessBusy}
        onClick={props.onCheckReadiness}
      >
        {props.readinessBusy && <Loader2 size={16} className="content-protection-spinner" />} Check
        readiness
      </button>
      {props.readiness && (
        <section
          id="readiness-result"
          className="content-protection-result"
          role="status"
          data-content-protection-readiness-result
        >
          <strong>{props.readiness.code === 'ready' ? 'Ready' : 'Not ready'}</strong>
          {props.readiness.model && (
            <span className="tool-badge">
              Resolved: {props.readiness.model} via {props.readiness.transport}
            </span>
          )}
          {props.readiness.cases?.map((item) => (
            <span key={item.name}>
              {item.name}: {item.verdict} ·{' '}
              {Object.entries(item.probabilities)
                .map(([id, probability]) => `${id} ${(probability * 100).toFixed(0)}%`)
                .join(', ')}
            </span>
          ))}
        </section>
      )}
    </>
  );
}
