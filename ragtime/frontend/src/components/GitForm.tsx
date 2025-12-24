import { useState } from 'react';
import { api } from '@/api';
import type { IndexJob } from '@/types';

interface GitFormProps {
  onJobCreated: () => void;
}

type StatusType = 'info' | 'success' | 'error' | null;

export function GitForm({ onJobCreated }: GitFormProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<{ type: StatusType; message: string }>({
    type: null,
    message: '',
  });

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;

    const name = (form.elements.namedItem('name') as HTMLInputElement).value;
    const description = (form.elements.namedItem('description') as HTMLTextAreaElement).value;
    const gitUrl = (form.elements.namedItem('git_url') as HTMLInputElement).value;
    const gitBranch = (form.elements.namedItem('git_branch') as HTMLInputElement).value;
    const filePatterns = (form.elements.namedItem('file_patterns') as HTMLInputElement).value;
    const excludePatterns = (form.elements.namedItem('exclude_patterns') as HTMLInputElement).value;

    setIsLoading(true);
    setStatus({ type: 'info', message: 'Starting git clone...' });

    try {
      const job: IndexJob = await api.indexFromGit({
        name,
        git_url: gitUrl,
        git_branch: gitBranch,
        config: {
          name,
          description,
          file_patterns: filePatterns.split(',').map((s) => s.trim()),
          exclude_patterns: excludePatterns.split(',').map((s) => s.trim()),
        },
      });
      setStatus({ type: 'success', message: `Job started - ID: ${job.id} - Status: ${job.status}` });
      form.reset();
      onJobCreated();
    } catch (err) {
      setStatus({ type: 'error', message: `Error: ${err instanceof Error ? err.message : 'Request failed'}` });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-group">
        <label>Index Name *</label>
        <input
          type="text"
          name="name"
          placeholder="e.g., odoo-17, my-codebase"
          required
        />
      </div>

      <div className="form-group">
        <label>Description (for AI context)</label>
        <textarea
          name="description"
          placeholder="Describe what this index contains so the AI knows when to use it, e.g., 'Odoo 17 modules source code including accounting, inventory, and sales apps'"
          rows={2}
          style={{ resize: 'vertical', minHeight: '60px' }}
        />
      </div>

      <div className="row">
        <div className="form-group">
          <label>Git URL *</label>
          <input
            type="text"
            name="git_url"
            placeholder="https://github.com/user/repo.git"
            required
          />
        </div>
        <div className="form-group">
          <label>Branch</label>
          <input type="text" name="git_branch" defaultValue="main" />
        </div>
      </div>

      <div className="row">
        <div className="form-group">
          <label>File Patterns (comma-separated)</label>
          <input
            type="text"
            name="file_patterns"
            defaultValue="**/*.py,**/*.md,**/*.xml"
          />
        </div>
        <div className="form-group">
          <label>Exclude Patterns</label>
          <input
            type="text"
            name="exclude_patterns"
            defaultValue="**/test/**,**/tests/**,**/__pycache__/**"
          />
        </div>
      </div>

      <button type="submit" className="btn" disabled={isLoading}>
        Clone & Index
      </button>

      {status.type && (
        <div className={`status-message ${status.type}`}>{status.message}</div>
      )}
    </form>
  );
}
