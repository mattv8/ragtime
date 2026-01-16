-- Add 'interrupted' status to ChatTaskStatus enum
-- This status is used for tasks that were running when the server restarted

ALTER TYPE "ChatTaskStatus" ADD VALUE IF NOT EXISTS 'interrupted';
