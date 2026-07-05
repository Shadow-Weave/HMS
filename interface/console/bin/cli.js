#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const args = process.argv.slice(2);

// Parse command line arguments
let port = process.env.PORT || 9999;
let hostname = process.env.HOSTNAME || '0.0.0.0';
let apiUrl = process.env.HMS_CP_DATAPLANE_API_URL;

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--port' || args[i] === '-p') {
    port = args[++i];
  } else if (args[i] === '--hostname' || args[i] === '-H') {
    hostname = args[++i];
  } else if (args[i] === '--api-url' || args[i] === '-a') {
    apiUrl = args[++i];
  } else if (args[i] === '--help' || args[i] === '-h') {
    console.log(`
HMS Control Plane

Usage: hms-control-plane [options]

Options:
  -p, --port <port>       Port to listen on (default: 9999, env: PORT)
  -H, --hostname <host>   Hostname to bind to (default: 0.0.0.0, env: HOSTNAME)
  -a, --api-url <url>     HMS API URL (env: HMS_CP_DATAPLANE_API_URL)
  -h, --help              Show this help message

Environment Variables:
  PORT                              Port to listen on
  HOSTNAME                          Hostname to bind to
  HMS_CP_DATAPLANE_API_URL    URL of the HMS API server
`);
    process.exit(0);
  }
}

// Find the standalone server
const standaloneDir = path.join(__dirname, '..', 'standalone');
const serverPath = path.join(standaloneDir, 'server.js');

if (!fs.existsSync(serverPath)) {
  console.error('Error: Standalone server not found at', serverPath);
  console.error('This package may not have been built correctly.');
  process.exit(1);
}

// Set up environment
const env = {
  ...process.env,
  PORT: String(port),
  HOSTNAME: hostname,
};

if (apiUrl) {
  env.HMS_CP_DATAPLANE_API_URL = apiUrl;
}

console.log(`Starting HMS Control Plane on http://${hostname}:${port}`);
if (apiUrl) {
  console.log(`API URL: ${apiUrl}`);
}

// Run the standalone server
const server = spawn('node', [serverPath], {
  cwd: standaloneDir,
  env,
  stdio: 'inherit',
});

server.on('error', (err) => {
  console.error('Failed to start server:', err.message);
  process.exit(1);
});

server.on('close', (code) => {
  process.exit(code || 0);
});

// Handle signals
process.on('SIGTERM', () => server.kill('SIGTERM'));
process.on('SIGINT', () => server.kill('SIGINT'));
