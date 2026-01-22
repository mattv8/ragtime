import { useState, useRef, useEffect } from 'react';
import { X, FileText, Upload, Link } from 'lucide-react';
import type { ContentPart } from '@/types';

export interface AttachmentFile {
  id: string;
  type: 'image' | 'file';
  name: string;
  size: number;
  mimeType: string;
  preview?: string;  // data URL for images
  dataUrl?: string;  // base64 data URL
  filePath?: string; // For file path input
}

interface FileAttachmentProps {
  attachments: AttachmentFile[];
  onAttachmentsChange: (attachments: AttachmentFile[]) => void;
  disabled?: boolean;
}

const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
const SUPPORTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
const SUPPORTED_FILE_TYPES = [
  ...SUPPORTED_IMAGE_TYPES,
  'text/plain',
  'application/pdf',
  'text/csv',
  'application/json',
  'text/markdown'
];

export function FileAttachment({ attachments, onAttachmentsChange, disabled }: FileAttachmentProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [filePathInput, setFilePathInput] = useState('');
  const [showFilePathInput, setShowFilePathInput] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };

    if (showDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showDropdown]);

  // Handle paste from clipboard
  useEffect(() => {
    const handlePaste = async (e: ClipboardEvent) => {
      if (disabled) return;

      const items = e.clipboardData?.items;
      if (!items) return;

      const files: File[] = [];

      // Check for files in clipboard
      for (let i = 0; i < items.length; i++) {
        const item = items[i];

        // Only process file items
        if (item.kind === 'file') {
          const file = item.getAsFile();
          if (file && SUPPORTED_FILE_TYPES.includes(file.type)) {
            files.push(file);
          }
        }
      }

      if (files.length > 0) {
        e.preventDefault();
        await handleFiles(files);
      }
    };

    document.addEventListener('paste', handlePaste);
    return () => document.removeEventListener('paste', handlePaste);
  }, [disabled, attachments]);

  const readFileAsDataURL = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const resizeImageDataUrl = async (
    dataUrl: string,
    mimeType: string,
    maxDimension = 1024,
    quality = 0.8
  ): Promise<string> => {
    try {
      const img = new Image();
      img.src = dataUrl;
      await img.decode();

      const { naturalWidth, naturalHeight } = img;
      if (!naturalWidth || !naturalHeight) return dataUrl;

      const scale = Math.min(1, maxDimension / Math.max(naturalWidth, naturalHeight));
      if (scale === 1) return dataUrl; // Already within bounds

      const width = Math.round(naturalWidth * scale);
      const height = Math.round(naturalHeight * scale);

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return dataUrl;
      ctx.drawImage(img, 0, 0, width, height);

      return canvas.toDataURL(mimeType || 'image/png', quality);
    } catch {
      return dataUrl; // Fallback to original if resize fails
    }
  };

  const handleFiles = async (files: File[]) => {
    const newAttachments: AttachmentFile[] = [];

    for (const file of files) {
      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        alert(`File "${file.name}" is too large. Maximum size is 20MB.`);
        continue;
      }

      // Validate file type
      if (!SUPPORTED_FILE_TYPES.includes(file.type)) {
        alert(`File type "${file.type}" is not supported for "${file.name}".`);
        continue;
      }

      const isImage = SUPPORTED_IMAGE_TYPES.includes(file.type);
      let dataUrl = await readFileAsDataURL(file);

      // Downsize images to reduce token usage while preserving visual fidelity
      if (isImage) {
        dataUrl = await resizeImageDataUrl(dataUrl, file.type);
      }

      newAttachments.push({
        id: `${Date.now()}-${Math.random()}`,
        type: isImage ? 'image' : 'file',
        name: file.name,
        size: file.size,
        mimeType: file.type,
        preview: isImage ? dataUrl : undefined,
        dataUrl
      });
    }

    onAttachmentsChange([...attachments, ...newAttachments]);
  };

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      await handleFiles(files);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Global drag & drop support
  useEffect(() => {
    if (disabled) return;

    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      setDragActive(true);
    };

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
    };

    const handleDragLeave = (e: DragEvent) => {
      if (e.relatedTarget === null) {
        setDragActive(false);
      }
    };

    const handleDrop = async (e: DragEvent) => {
      e.preventDefault();
      setDragActive(false);

      const files = Array.from(e.dataTransfer?.files || []);
      if (files.length > 0) {
        await handleFiles(files);
      }
    };

    document.addEventListener('dragenter', handleDragEnter);
    document.addEventListener('dragover', handleDragOver);
    document.addEventListener('dragleave', handleDragLeave);
    document.addEventListener('drop', handleDrop);

    return () => {
      document.removeEventListener('dragenter', handleDragEnter);
      document.removeEventListener('dragover', handleDragOver);
      document.removeEventListener('dragleave', handleDragLeave);
      document.removeEventListener('drop', handleDrop);
    };
  }, [disabled, attachments]);

  const handleFilePathSubmit = () => {
    if (!filePathInput.trim()) return;

    const filename = filePathInput.split('/').pop() || filePathInput.split('\\').pop() || 'file';

    onAttachmentsChange([...attachments, {
      id: `${Date.now()}-${Math.random()}`,
      type: 'file',
      name: filename,
      size: 0,
      mimeType: 'application/octet-stream',
      filePath: filePathInput.trim()
    }]);

    setFilePathInput('');
    setShowFilePathInput(false);
  };

  const removeAttachment = (id: string) => {
    onAttachmentsChange(attachments.filter(a => a.id !== id));
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return 'Path';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <>
      {attachments.length > 0 && (
        <div className="attachment-preview-list">
          {attachments.map(attachment => (
            <div key={attachment.id} className="attachment-item">
              {attachment.type === 'image' && attachment.preview ? (
                <div className="attachment-image-preview">
                  <img src={attachment.preview} alt={attachment.name} />
                </div>
              ) : (
                <div className="attachment-file-preview">
                  {attachment.filePath ? (
                    <Link size={20} />
                  ) : (
                    <FileText size={20} />
                  )}
                </div>
              )}
              <div className="attachment-info">
                <span className="attachment-name" title={attachment.name}>
                  {attachment.name}
                </span>
                <span className="attachment-size">{formatFileSize(attachment.size)}</span>
              </div>
              <button
                type="button"
                className="attachment-remove"
                onClick={() => removeAttachment(attachment.id)}
                disabled={disabled}
                aria-label="Remove attachment"
              >
                <X size={16} />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="file-attachment">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={SUPPORTED_FILE_TYPES.join(',')}
          onChange={handleFileInput}
          disabled={disabled}
          style={{ display: 'none' }}
        />

        <div className="attachment-controls" ref={dropdownRef}>
          <button
            type="button"
            className="btn-attach-menu"
            onClick={() => setShowDropdown(!showDropdown)}
            disabled={disabled}
            title="Attach files or images"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
          </button>

          {showDropdown && (
            <div className="attachment-dropdown">
              <button
                type="button"
                className="attachment-dropdown-item"
                onClick={() => {
                  fileInputRef.current?.click();
                  setShowDropdown(false);
                }}
                disabled={disabled}
              >
                <Upload size={18} />
                <span>Upload Files</span>
              </button>
              <button
                type="button"
                className="attachment-dropdown-item"
                onClick={() => {
                  setShowFilePathInput(!showFilePathInput);
                  setShowDropdown(false);
                }}
                disabled={disabled}
              >
                <Link size={18} />
                <span>File Path</span>
              </button>
            </div>
          )}
        </div>

        {showFilePathInput && (
          <div className="file-path-input-group">
            <input
              type="text"
              className="file-path-input"
              placeholder="/path/to/file.txt"
              value={filePathInput}
              onChange={(e) => setFilePathInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  handleFilePathSubmit();
                } else if (e.key === 'Escape') {
                  setShowFilePathInput(false);
                  setFilePathInput('');
                }
              }}
              disabled={disabled}
              autoFocus
            />
            <button
              type="button"
              className="btn-path-submit"
              onClick={handleFilePathSubmit}
              disabled={disabled || !filePathInput.trim()}
            >
              Add
            </button>
            <button
              type="button"
              className="btn-path-close"
              onClick={() => {
                setShowFilePathInput(false);
                setFilePathInput('');
              }}
              title="Close"
            >
              <X size={14} />
            </button>
          </div>
        )}
      </div>

      {dragActive && (
        <div className="drag-overlay-global">
          <div className="drag-overlay-content">
            <Upload size={48} />
            <p>Drop files here</p>
          </div>
        </div>
      )}
    </>
  );
}

// Helper to convert attachments to API format
export function attachmentsToContentParts(text: string, attachments: AttachmentFile[]): ContentPart[] {
  const parts: ContentPart[] = [];

  // Add text content
  if (text.trim()) {
    parts.push({
      type: 'text',
      text: text.trim()
    });
  }

  // Add attachments
  for (const attachment of attachments) {
    if (attachment.type === 'image' && attachment.dataUrl) {
      parts.push({
        type: 'image_url',
        image_url: {
          url: attachment.dataUrl,
          detail: 'auto'
        }
      });
    } else if (attachment.filePath) {
      parts.push({
        type: 'file',
        file_path: attachment.filePath,
        filename: attachment.name,
        mime_type: attachment.mimeType
      });
    }
  }

  return parts;
}
