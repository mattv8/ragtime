import type { UserSpaceFileInfo } from '@/types';

export interface UserSpaceTreeNode {
  name: string;
  path: string;
  type: 'folder' | 'file';
  children: UserSpaceTreeNode[];
}

interface MutableTreeNode extends UserSpaceTreeNode {
  childrenMap: Map<string, MutableTreeNode>;
}

function normalizePath(path: string): string {
  return path.trim().replace(/^\/+|\/+$/g, '').replace(/\/+/g, '/');
}

function toNode(node: MutableTreeNode): UserSpaceTreeNode {
  const children = Array.from(node.childrenMap.values()).map(toNode);
  children.sort((left, right) => {
    if (left.type !== right.type) {
      return left.type === 'folder' ? -1 : 1;
    }
    return left.name.localeCompare(right.name, undefined, { sensitivity: 'base' });
  });

  return {
    name: node.name,
    path: node.path,
    type: node.type,
    children,
  };
}

export function getAncestorFolderPaths(filePath: string): string[] {
  const normalizedPath = normalizePath(filePath);
  if (!normalizedPath) {
    return [];
  }

  const segments = normalizedPath.split('/').filter(Boolean);
  if (segments.length <= 1) {
    return [];
  }

  const ancestors: string[] = [];
  for (let index = 0; index < segments.length - 1; index += 1) {
    ancestors.push(segments.slice(0, index + 1).join('/'));
  }
  return ancestors;
}

export function listFolderPaths(files: UserSpaceFileInfo[]): Set<string> {
  const folderPaths = new Set<string>();
  for (const file of files) {
    for (const folderPath of getAncestorFolderPaths(file.path)) {
      folderPaths.add(folderPath);
    }
  }
  return folderPaths;
}

export function buildUserSpaceTree(files: UserSpaceFileInfo[]): UserSpaceTreeNode[] {
  const rootMap = new Map<string, MutableTreeNode>();

  for (const file of files) {
    const normalizedPath = normalizePath(file.path);
    if (!normalizedPath) {
      continue;
    }

    const segments = normalizedPath.split('/').filter(Boolean);
    let currentMap = rootMap;

    for (let index = 0; index < segments.length; index += 1) {
      const name = segments[index];
      const path = segments.slice(0, index + 1).join('/');
      const isFile = index === segments.length - 1;
      const existingNode = currentMap.get(name);

      if (existingNode) {
        currentMap = existingNode.childrenMap;
        continue;
      }

      const nextNode: MutableTreeNode = {
        name,
        path,
        type: isFile ? 'file' : 'folder',
        children: [],
        childrenMap: new Map<string, MutableTreeNode>(),
      };

      currentMap.set(name, nextNode);
      currentMap = nextNode.childrenMap;
    }
  }

  const tree = Array.from(rootMap.values()).map(toNode);
  tree.sort((left, right) => {
    if (left.type !== right.type) {
      return left.type === 'folder' ? -1 : 1;
    }
    return left.name.localeCompare(right.name, undefined, { sensitivity: 'base' });
  });

  return tree;
}
