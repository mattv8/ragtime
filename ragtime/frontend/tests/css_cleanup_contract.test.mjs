import { expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const stylesDir = path.resolve(import.meta.dirname, '../src/styles');

function readStyleFile(name) {
  return readFileSync(path.join(stylesDir, name), 'utf8');
}

const attachmentsCss = readStyleFile('attachments.css');
const chatCss = readStyleFile('chat.css');
const componentsCss = readStyleFile('components.css');

test('live attachment and modal selectors remain defined', () => {
  const requiredSelectors = [
    '.attachment-preview-list',
    '.attachment-item',
    '.btn-attach-menu',
    '.drag-overlay-global',
    '.message-attachments',
    '.message-attachment',
    '.message-attachment-image',
    '.message-attachment-file',
    '.message-attachment-file-icon',
    '.message-attachment-file-name',
    '.image-modal-overlay',
    '.image-modal-close',
  ];

  for (const selector of requiredSelectors) {
    expect(attachmentsCss).toMatch(new RegExp(`${selector.replace('.', '\\.')}(?=[\\s:{.[,#>+~])`));
  }
});

test('unused legacy attachment selectors are removed', () => {
  const retiredPatterns = [
    /attachment-input-area/,
    /\.btn-attach\s*\{/,
    /file-path-input-group/,
    /file-path-input(?!-)/,
    /btn-path-submit/,
    /btn-path-close/,
  ];

  for (const pattern of retiredPatterns) {
    expect(attachmentsCss).not.toMatch(pattern);
  }
});

test('fullscreen chrome is consolidated without changing surface-specific backgrounds', () => {
  expect(componentsCss).toMatch(
    /\.chat-panel-fullscreen,\s*\.userspace-layout\.userspace-fullscreen\s*\{/s,
  );

  expect(chatCss).toMatch(/\.chat-panel-fullscreen\s*\{[^}]*background: var\(--color-surface\);/s);

  expect(componentsCss).toMatch(
    /\.userspace-layout\.userspace-fullscreen\s*\{[^}]*background: var\(--color-bg-primary\);/s,
  );
});
