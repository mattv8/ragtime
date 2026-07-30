import type { Dispatch, SetStateAction } from 'react';

import type { UpdateSettingsRequest } from '@/types';

import { SettingsAccordionSection } from './SettingsAccordionSection';
import type { SettingsAccordionSectionId } from './settingsAccordionState';

export interface AgentBehaviorSettingsSectionProps {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  formData: UpdateSettingsRequest;
  setFormData: Dispatch<SetStateAction<UpdateSettingsRequest>>;
  handleSaveAgentBehavior: () => void | Promise<void>;
  agentBehaviorSaving: boolean;
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
  } = props;

  const toolSkillsEnabled = formData.tool_skills_enabled !== false;
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

        <div
          className="form-group agent-behavior-settings-switch-card"
          id="setting-tool_skills_enabled"
        >
          <div className="agent-behavior-settings-switch-copy">
            <label
              htmlFor="agent-behavior-tool-skills-enabled"
              className="agent-behavior-settings-switch-title"
            >
              Load tools on demand
            </label>
            <p className="field-help">
              {toolSkillsEnabled
                ? 'Only essential tools and the tool-skill controls are sent initially. The agent can load other tools during the same request. Loaded tools remain available for the conversation while current access and health checks continue to apply.'
                : 'All eligible tools and schemas are sent with every request, matching legacy behavior.'}
            </p>
          </div>

          <label className="toggle-switch agent-behavior-settings-switch-toggle">
            <input
              id="agent-behavior-tool-skills-enabled"
              type="checkbox"
              aria-label="Load tools on demand"
              checked={toolSkillsEnabled}
              onChange={(e) => setFormData({ ...formData, tool_skills_enabled: e.target.checked })}
            />
            <span className="toggle-slider"></span>
          </label>
        </div>

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

        <div className="form-actions">
          <button
            type="button"
            className="btn"
            onClick={handleSaveAgentBehavior}
            disabled={agentBehaviorSaving}
          >
            {agentBehaviorSaving ? 'Saving...' : 'Save Agent Behavior'}
          </button>
        </div>
      </fieldset>
    </SettingsAccordionSection>
  );
}
