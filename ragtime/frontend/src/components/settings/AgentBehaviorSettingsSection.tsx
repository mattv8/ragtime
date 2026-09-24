import type { Dispatch, SetStateAction } from 'react';

import type { UpdateSettingsRequest } from '@/types';

import { SettingsAccordionSection } from './SettingsAccordionSection';
import { MasterToggle } from './MasterToggle';
import type { SettingsAccordionSectionId } from './settingsAccordionState';

export interface AgentBehaviorSettingsSectionProps {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  formData: UpdateSettingsRequest;
  setFormData: Dispatch<SetStateAction<UpdateSettingsRequest>>;
  handleSaveAgentBehavior: () => void | Promise<void>;
  agentBehaviorSaving: boolean;
  isAdmin: boolean;
  userspaceExecTimeoutDefaultDraft: string;
  userspaceExecTimeoutMaxDraft: string;
  onUserspaceExecTimeoutDefaultDraftChange: (value: string) => void;
  onUserspaceExecTimeoutMaxDraftChange: (value: string) => void;
  userspaceExecTimeoutError: string | null;
}

export function AgentBehaviorSettingsSection(
  props: AgentBehaviorSettingsSectionProps,
): JSX.Element {
  const {
    open,
    onToggle,
    formData,
    setFormData,
    handleSaveAgentBehavior,
    agentBehaviorSaving,
    isAdmin,
    userspaceExecTimeoutDefaultDraft,
    userspaceExecTimeoutMaxDraft,
    onUserspaceExecTimeoutDefaultDraftChange,
    onUserspaceExecTimeoutMaxDraftChange,
    userspaceExecTimeoutError,
  } = props;

  const toolSkillsEnabled = formData.tool_skills_enabled === true;
  const maxIterations = formData.max_iterations ?? 30;
  const maxToolOutputChars = formData.max_tool_output_chars ?? 5000;
  const scratchpadWindowSize = formData.scratchpad_window_size ?? 6;

  return (
    <SettingsAccordionSection
      id="agent-behavior"
      title="Agent Behavior"
      open={open}
      onToggle={onToggle}
    >
      <fieldset id="setting-agent_behavior">
        <legend>Agent Behavior</legend>
        <p className="fieldset-help">Configure global agent execution and tool behavior.</p>

        <MasterToggle
          settingId="setting-tool_skills_enabled"
          inputId="agent-behavior-tool-skills-enabled"
          label="Load tools on demand"
          checked={toolSkillsEnabled}
          onChange={(checked) => setFormData({ ...formData, tool_skills_enabled: checked })}
          help={
            toolSkillsEnabled
              ? 'Only essential tools and the tool-skill controls are sent initially. The agent can load other tools during the same request. Loaded tools remain available for the conversation while current access and health checks continue to apply.'
              : 'All eligible tools and schemas are sent with every request, matching legacy behavior.'
          }
        />

        <div className="agent-behavior-settings-grid">
          <div className="form-group">
            <label htmlFor="agent-behavior-max-iterations">Max Tool Iterations</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                id="agent-behavior-max-iterations"
                type="range"
                min="1"
                max="100"
                step="1"
                style={{ flex: 1 }}
                value={maxIterations}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    max_iterations: parseInt(e.target.value, 10),
                  })
                }
              />
              <span
                style={{
                  minWidth: '30px',
                  textAlign: 'right',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {maxIterations}
              </span>
            </div>
            <p className="field-help">Maximum number of agent tool-calling steps.</p>
          </div>

          <div className="form-group">
            <label htmlFor="agent-behavior-max-tool-output">Max Tool Output (chars)</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                id="agent-behavior-max-tool-output"
                type="range"
                min="0"
                max="50000"
                step="1000"
                style={{ flex: 1 }}
                value={maxToolOutputChars}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    max_tool_output_chars: parseInt(e.target.value, 10),
                  })
                }
              />
              <span
                style={{
                  minWidth: '60px',
                  textAlign: 'right',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {maxToolOutputChars === 0 ? 'Off' : `${(maxToolOutputChars / 1000).toFixed(0)}K`}
              </span>
            </div>
            <p className="field-help">
              Cap on each tool response before truncation (0 = no limit). Lower values curb token
              growth during multi-step tool loops.
            </p>
          </div>

          <div className="form-group">
            <label htmlFor="agent-behavior-context-window">Context Window (steps)</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                id="agent-behavior-context-window"
                type="range"
                min="0"
                max="30"
                step="1"
                style={{ flex: 1 }}
                value={scratchpadWindowSize}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    scratchpad_window_size: parseInt(e.target.value, 10),
                  })
                }
              />
              <span
                style={{
                  minWidth: '40px',
                  textAlign: 'right',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {scratchpadWindowSize === 0 ? 'All' : scratchpadWindowSize}
              </span>
            </div>
            <p className="field-help">
              Number of recent tool steps kept in full detail; older steps are compressed (0 = keep
              all). Smaller windows reduce input tokens in long conversations.
            </p>
          </div>
        </div>

        {isAdmin && (
          <div
            className="agent-behavior-settings-grid"
            id="setting-userspace-exec-timeouts"
            aria-describedby={
              userspaceExecTimeoutError ? 'agent-behavior-userspace-exec-timeout-error' : undefined
            }
          >
            <div className="form-group">
              <label htmlFor="agent-behavior-userspace-exec-timeout-default">
                Default command timeout (seconds)
              </label>
              <input
                id="agent-behavior-userspace-exec-timeout-default"
                type="number"
                min="1"
                max={userspaceExecTimeoutMaxDraft || undefined}
                step="1"
                inputMode="numeric"
                value={userspaceExecTimeoutDefaultDraft}
                onChange={(event) => onUserspaceExecTimeoutDefaultDraftChange(event.target.value)}
              />
              <p className="field-help">Used when a workspace terminal command has no timeout.</p>
            </div>

            <div className="form-group">
              <label htmlFor="agent-behavior-userspace-exec-timeout-max">
                Maximum command timeout (seconds)
              </label>
              <input
                id="agent-behavior-userspace-exec-timeout-max"
                type="number"
                min="30"
                max="3600"
                step="1"
                inputMode="numeric"
                value={userspaceExecTimeoutMaxDraft}
                onChange={(event) => onUserspaceExecTimeoutMaxDraftChange(event.target.value)}
              />
              <p className="field-help">Hard platform ceiling: 3600 seconds.</p>
            </div>

            <p className="field-help" id="agent-behavior-userspace-exec-timeout-help">
              These instance-wide limits apply only to workspace terminal commands. Long active
              commands can keep a sanctioned app restart busy until they finish.
            </p>
            {userspaceExecTimeoutError && (
              <p
                className="field-error"
                id="agent-behavior-userspace-exec-timeout-error"
                role="alert"
              >
                {userspaceExecTimeoutError}
              </p>
            )}
          </div>
        )}

        <div className="form-actions">
          <button
            type="button"
            className="btn"
            onClick={handleSaveAgentBehavior}
            disabled={agentBehaviorSaving || (isAdmin && userspaceExecTimeoutError !== null)}
          >
            {agentBehaviorSaving ? 'Saving...' : 'Save Agent Behavior'}
          </button>
        </div>
      </fieldset>
    </SettingsAccordionSection>
  );
}
