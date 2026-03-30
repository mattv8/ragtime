-- Add configurable iframe sandbox flags for User Space previews.
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "userspace_preview_sandbox_flags" TEXT[] NOT NULL DEFAULT ARRAY[
  'allow-scripts',
  'allow-same-origin',
  'allow-forms',
  'allow-popups',
  'allow-popups-to-escape-sandbox',
  'allow-modals',
  'allow-downloads'
]::TEXT[];