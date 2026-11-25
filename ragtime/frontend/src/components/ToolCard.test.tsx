import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ToolCard, type ToolCardProps } from './ToolCard';
import { pdmTool, ungroupedTool } from './ToolsPanel.testSupport';

vi.mock('@/utils/useFitLineClamp', () => ({
  useFitLineClamp: () => ({ lines: 2, isClamped: true }),
}));

const props = (overrides: Partial<ToolCardProps> = {}): ToolCardProps => ({
  tool: ungroupedTool,
  tools: [ungroupedTool],
  heartbeat: null,
  showFooterActions: false,
  onEdit: vi.fn(),
  onDelete: vi.fn(),
  onTest: vi.fn(),
  onOpenSkills: vi.fn(),
  testing: false,
  ...overrides,
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('ToolCard', () => {
  it('orders Disabled and Key Mismatch before Write', () => {
    const tool = {
      ...ungroupedTool,
      enabled: false,
      allow_write: true,
      undecryptable_fields: ['password'],
    };
    const { container } = render(<ToolCard {...props({ tool })} />);
    const labels = Array.from(container.querySelectorAll('.tool-card-badges .tool-badge')).map(
      (badge) => badge.textContent,
    );
    expect(labels.slice(0, 3)).toEqual(['Disabled', 'Key Mismatch', 'Write']);
  });

  it('formats PDM document counts in its facts', () => {
    const stats = { document_count: 44595 } as NonNullable<ToolCardProps['pdmStats']>;
    const { container } = render(<ToolCard {...props({ tool: pdmTool, pdmStats: stats })} />);
    expect(screen.getByText(`${new Intl.NumberFormat().format(44595)} documents`)).toBeTruthy();
    const rows = container.querySelectorAll('.tool-card-facts > div');
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      expect(row.matches('.tool-card-schema-stats')).toBe(true);
      expect(row.querySelector('dt.schema-stats-label')).toBeTruthy();
      expect(row.querySelector('dd.schema-stats-value')).toBeTruthy();
    }
  });

  it('omits heartbeat errors while disabled or testing', () => {
    const heartbeat = { alive: false, error: 'Offline' } as NonNullable<ToolCardProps['heartbeat']>;
    const { container, rerender } = render(
      <ToolCard {...props({ tool: { ...ungroupedTool, enabled: false }, heartbeat })} />,
    );
    expect(container.querySelector('.tool-card-heartbeat-error')).toBeNull();
    rerender(<ToolCard {...props({ heartbeat, testing: true })} />);
    expect(container.querySelector('.tool-card-heartbeat-error')).toBeNull();
    rerender(<ToolCard {...props({ heartbeat })} />);
    expect(container.querySelector('.tool-card-heartbeat-error')?.textContent).toContain('Offline');
  });

  it('opens actions at its button and reflects expanded state', () => {
    const onOpenActions = vi.fn();
    const { rerender } = render(<ToolCard {...props({ onOpenActions })} />);
    const button = screen.getByRole('button', { name: `Tool actions for ${ungroupedTool.name}` });
    expect(button.getAttribute('aria-expanded')).toBe('false');
    fireEvent.click(button);
    expect(onOpenActions).toHaveBeenCalledWith(ungroupedTool, button);
    rerender(<ToolCard {...props({ onOpenActions, actionsMenuOpen: true })} />);
    expect(button.getAttribute('aria-expanded')).toBe('true');
  });

  it('opens the complete description from More when clamped', () => {
    const tool = { ...ungroupedTool, description: 'A long description in its entirety' };
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
      top: 300,
      bottom: 320,
      left: 100,
      right: 200,
      width: 100,
      height: 20,
    } as DOMRect);
    render(<ToolCard {...props({ tool })} />);
    const button = screen.getByRole('button', { name: 'Show full description' });
    expect(button.getAttribute('aria-expanded')).toBe('false');
    fireEvent.click(button);
    expect(button.getAttribute('aria-expanded')).toBe('true');
    const tooltip = screen.getByRole('tooltip');
    expect(tooltip.textContent).toContain(tool.description);
    expect(tooltip.classList.contains('popover-bottom')).toBe(true);
    fireEvent.click(screen.getAllByText(tool.description)[1]);
    expect(screen.queryByRole('textbox')).toBeNull();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('tooltip')).toBeNull();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(button);
  });

  it('exposes the full name as a title only when the heading is truncated', () => {
    vi.spyOn(HTMLElement.prototype, 'scrollWidth', 'get').mockReturnValue(200);
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(100);
    render(<ToolCard {...props()} />);
    expect(screen.getByRole('heading', { name: ungroupedTool.name }).getAttribute('title')).toBe(
      ungroupedTool.name,
    );
  });

  it('closes More on outside mouse down but not on the popover', () => {
    render(<ToolCard {...props()} />);
    const button = screen.getByRole('button', { name: 'Show full description' });
    fireEvent.click(button);
    fireEvent.mouseDown(screen.getByRole('tooltip'));
    expect(button.getAttribute('aria-expanded')).toBe('true');
    fireEvent.mouseDown(document.body);
    expect(button.getAttribute('aria-expanded')).toBe('false');
  });
});
