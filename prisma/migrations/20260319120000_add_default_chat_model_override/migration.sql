-- Add optional manual override for chat default model.
ALTER TABLE app_settings
ADD COLUMN IF NOT EXISTS default_chat_model TEXT;
