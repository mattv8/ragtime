import { Database, Terminal, Monitor, Folder, File, Plug, Loader2, Check, Pencil, Trash2, X, HardDrive } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export type IconType = 'database' | 'terminal' | 'server' | 'folder' | 'file' | 'plug' | 'loader' | 'check' | 'pencil' | 'trash' | 'close' | 'harddrive';

const iconMap: Record<IconType, LucideIcon> = {
  database: Database,
  terminal: Terminal,
  server: Monitor,
  folder: Folder,
  file: File,
  plug: Plug,
  loader: Loader2,
  check: Check,
  pencil: Pencil,
  trash: Trash2,
  close: X,
  harddrive: HardDrive,
};

interface IconProps {
  name: IconType;
  size?: number;
  className?: string;
}

export function Icon({ name, size = 20, className = '' }: IconProps) {
  const IconComponent = iconMap[name];
  if (!IconComponent) return null;
  return <IconComponent size={size} className={className} />;
}

/**
 * Get the appropriate icon type for a tool type's icon field
 */
export function getToolIconType(iconField: string | undefined): IconType {
  switch (iconField) {
    case 'database':
      return 'database';
    case 'terminal':
      return 'terminal';
    case 'folder':
      return 'folder';
    case 'harddrive':
      return 'harddrive';
    default:
      return 'server';
  }
}
