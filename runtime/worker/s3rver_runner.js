#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const S3rver = require('s3rver');

function getArg(flag) {
  const index = process.argv.indexOf(flag);
  if (index === -1) {
    return '';
  }
  return process.argv[index + 1] || '';
}

async function main() {
  const directory = getArg('--directory');
  const port = Number.parseInt(getArg('--port') || '0', 10);
  const bucketsRaw = getArg('--buckets-json') || '[]';
  const bucketNames = JSON.parse(bucketsRaw);

  if (!directory || !Number.isInteger(port) || port <= 0) {
    throw new Error('directory and port are required');
  }
  if (!Array.isArray(bucketNames)) {
    throw new Error('buckets-json must be a JSON array');
  }

  fs.mkdirSync(directory, { recursive: true });
  for (const bucketName of bucketNames) {
    if (!bucketName) {
      continue;
    }
    fs.mkdirSync(path.join(directory, bucketName), { recursive: true });
  }

  const server = new S3rver({
    address: '127.0.0.1',
    port,
    directory,
    silent: true,
    allowMismatchedSignatures: false,
    vhostBuckets: false,
    configureBuckets: bucketNames
      .filter((bucketName) => typeof bucketName === 'string' && bucketName.trim())
      .map((bucketName) => ({ name: bucketName })),
  });

  const closeServer = async () => {
    try {
      await server.close();
    } catch {
      // Best effort shutdown.
    }
    process.exit(0);
  };

  process.on('SIGTERM', () => {
    void closeServer();
  });
  process.on('SIGINT', () => {
    void closeServer();
  });

  await server.run();
  process.stdout.write(`${JSON.stringify({ type: 'ready', port })}\n`);
}

main().catch((error) => {
  process.stdout.write(
    `${JSON.stringify({ type: 'error', error: error instanceof Error ? error.message : String(error) })}\n`
  );
  process.exit(1);
});