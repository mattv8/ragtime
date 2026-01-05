import { GitIndexWizard } from './GitIndexWizard';

interface GitFormProps {
  onJobCreated: () => void;
}

export function GitForm({ onJobCreated }: GitFormProps) {
  return <GitIndexWizard onJobCreated={onJobCreated} />;
}
