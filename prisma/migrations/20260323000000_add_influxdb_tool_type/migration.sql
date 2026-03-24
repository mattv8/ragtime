-- Add influxdb to ToolType enum
-- InfluxDB 2.x Flux tool support

ALTER TYPE "ToolType" ADD VALUE IF NOT EXISTS 'influxdb';
