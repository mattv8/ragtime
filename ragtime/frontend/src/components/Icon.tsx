import { Database, Terminal, Monitor, Folder, File, Plug, Loader2, Check, Pencil, Trash2, X, HardDrive, Users, ChevronUp, ChevronDown, AlertCircle, AlertTriangle, Circle, RefreshCw } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export type IconType = 'database' | 'terminal' | 'server' | 'folder' | 'file' | 'plug' | 'loader' | 'check' | 'pencil' | 'trash' | 'close' | 'harddrive' | 'users' | 'chevron-up' | 'chevron-down' | 'alert-circle' | 'alert-triangle' | 'circle' | 'refresh';

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
  users: Users,
  'chevron-up': ChevronUp,
  'chevron-down': ChevronDown,
  'alert-circle': AlertCircle,
  'alert-triangle': AlertTriangle,
  circle: Circle,
  refresh: RefreshCw,
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
