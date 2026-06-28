import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';

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
    assert.match(
      attachmentsCss,
      new RegExp(`${selector.replace('.', '\\.')}(?=[\\s:{.[#,>+~])`),
      `${selector} should stay defined in attachments.css`,
    );
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
    assert.doesNotMatch(
      attachmentsCss,
      pattern,
      `${pattern} should be removed from attachments.css`,
    );
  }
});

test('fullscreen chrome is consolidated without changing surface-specific backgrounds', () => {
  assert.match(
    componentsCss,
    /\.chat-panel-fullscreen,\s*\.userspace-layout\.userspace-fullscreen\s*\{/s,
    'fullscreen shell chrome should be defined once in components.css',
  );

  assert.match(
    chatCss,
    /\.chat-panel-fullscreen\s*\{[^}]*background: var\(--color-surface\);/s,
    'chat fullscreen background should stay on the chat surface token',
  );

  assert.match(
    componentsCss,
    /\.userspace-layout\.userspace-fullscreen\s*\{[^}]*background: var\(--color-bg-primary\);/s,
    'userspace fullscreen background should stay on the userspace background token',
  );
});
