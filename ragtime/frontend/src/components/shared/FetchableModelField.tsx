export interface FetchableModelOption {
  id: string;
  label: string;
}

interface FetchableModelFieldProps {
  idPrefix: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  models: FetchableModelOption[];
  loaded: boolean;
  fetching: boolean;
  error: string | null;
  onFetch: () => void;
  fetchDisabled: boolean;
  placeholder: string;
  help: string;
  selectPlaceholder?: string;
  onBlur?: () => void;
  disabled?: boolean;
  helpId?: string;
}

export function FetchableModelField({
  idPrefix,
  label,
  value,
  onChange,
  models,
  loaded,
  fetching,
  error,
  onFetch,
  fetchDisabled,
  placeholder,
  help,
  selectPlaceholder,
  onBlur,
  disabled = false,
  helpId: providedHelpId,
}: FetchableModelFieldProps): JSX.Element {
  const hasModels = loaded && models.length > 0;
  const inputId = `${idPrefix}-input`;
  const selectId = `${idPrefix}-select`;
  const helpId = providedHelpId || `${idPrefix}-help`;
  const errorId = `${idPrefix}-error`;
  const describedBy = error ? `${helpId} ${errorId}` : helpId;
  const hasUnlistedValue = value && !models.some((model) => model.id === value);
  return (
    <div className="form-group" id={`${idPrefix}-row`}>
      <label htmlFor={hasModels ? selectId : inputId}>{label}</label>
      <div className="input-with-button">
        {hasModels ? (
          <select
            id={selectId}
            value={value}
            disabled={disabled}
            aria-describedby={describedBy}
            onChange={(event) => onChange(event.target.value)}
          >
            {selectPlaceholder && <option value="">{selectPlaceholder}</option>}
            {hasUnlistedValue && <option value={value}>{value}</option>}
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label}
              </option>
            ))}
          </select>
        ) : (
          <input
            id={inputId}
            type="text"
            value={value}
            disabled={disabled}
            placeholder={placeholder}
            aria-describedby={describedBy}
            onChange={(event) => onChange(event.target.value)}
            onBlur={onBlur}
          />
        )}
        <button
          type="button"
          className={`btn btn-test${loaded ? ' btn-connected' : ''}`}
          onClick={onFetch}
          disabled={disabled || fetchDisabled || fetching}
          aria-describedby={describedBy}
        >
          {fetching ? 'Fetching...' : loaded ? 'Loaded' : 'Fetch Models'}
        </button>
      </div>
      {error && (
        <p id={errorId} className="field-error" role="alert">
          {error}
        </p>
      )}
      <p id={helpId} className="field-help">
        {help}
      </p>
    </div>
  );
}
