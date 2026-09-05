import { useCallback, useState, type ChangeEvent, type DragEvent, type ReactNode } from 'react';
import { FileTypeIcon } from './FileTypeIcon';

interface FileDropZoneProps {
  accept: string;
  disabled?: boolean;
  file: File | null;
  icon: ReactNode;
  helpText: ReactNode;
  prompt: ReactNode;
  onFileSelected: (file: File) => void;
}

export function FileDropZone({
  accept,
  disabled = false,
  file,
  icon,
  helpText,
  prompt,
  onFileSelected,
}: FileDropZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) onFileSelected(dropped);
    },
    [onFileSelected],
  );

  const handleFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const selected = e.target.files?.[0];
      if (selected) onFileSelected(selected);
    },
    [onFileSelected],
  );

  return (
    <div
      className={`file-input-wrapper ${isDragOver ? 'dragover' : ''} ${file ? 'has-file' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="icon">{icon}</div>
      <div>{prompt}</div>
      <div style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', marginTop: 8 }}>
        {helpText}
      </div>
      {file && (
        <div className="file-name">
          <FileTypeIcon path={file.name} size={16} /> {file.name}
        </div>
      )}
      <input
        type="file"
        name="file"
        accept={accept}
        onChange={handleFileChange}
        disabled={disabled}
      />
    </div>
  );
}
