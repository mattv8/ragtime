import { createPortal } from 'react-dom';
import { ModelArtifactModal } from './ModelArtifactModal';
import { useModelArtifactModal } from './modelArtifactModalStore';

export function ModelArtifactModalHost() {
  const state = useModelArtifactModal();
  return state
    ? createPortal(<ModelArtifactModal key={state.artifact.id} {...state} />, document.body)
    : null;
}
