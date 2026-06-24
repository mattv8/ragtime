const fs = require('fs');
const path = require('path');

const input = fs.readFileSync(0, 'utf8');
const filePath = process.argv[2] || 'dashboard/main.ts';

let ts;
for (const candidate of [
  path.join('/ragtime/ragtime/frontend/node_modules/typescript'),
  path.join(__dirname, '..', 'frontend', 'node_modules', 'typescript'),
  'typescript',
]) {
  try {
    ts = require(candidate);
    break;
  } catch (err) {
    // Try the next location.
  }
}

if (!ts) {
  console.log(
    JSON.stringify({
      ok: false,
      validator_available: false,
      message: 'TypeScript validator unavailable in runtime',
      errors: [],
    })
  );
  process.exit(0);
}

function getScriptKind(targetPath) {
  const lower = String(targetPath || '').toLowerCase();
  if (lower.endsWith('.tsx')) {
    return ts.ScriptKind.TSX;
  }
  if (lower.endsWith('.jsx')) {
    return ts.ScriptKind.JSX;
  }
  if (lower.endsWith('.js') || lower.endsWith('.mjs') || lower.endsWith('.cjs')) {
    return ts.ScriptKind.JS;
  }
  return ts.ScriptKind.TS;
}

const scriptKind = getScriptKind(filePath);
const sourceFile = ts.createSourceFile(
  filePath,
  input,
  ts.ScriptTarget.ES2020,
  true,
  scriptKind
);

const result = ts.transpileModule(input, {
  fileName: filePath,
  reportDiagnostics: true,
  compilerOptions: {
    module: ts.ModuleKind.ES2020,
    target: ts.ScriptTarget.ES2020,
    isolatedModules: true,
    allowJs: true,
    checkJs: true,
    jsx: ts.JsxEmit.ReactJSX,
  },
});

const diagnostics = (result.diagnostics || []).filter(
  (d) => d.category === ts.DiagnosticCategory.Error
);
const errors = diagnostics.map((d) => {
  const message = ts.flattenDiagnosticMessageText(d.messageText, '\n');
  if (!d.file || d.start === undefined) {
    return message;
  }
  const pos = d.file.getLineAndCharacterOfPosition(d.start);
  return `${d.file.fileName}:${pos.line + 1}:${pos.character + 1} ${message}`;
});

function isExecuteCall(node) {
  return (
    ts.isCallExpression(node) &&
    ts.isPropertyAccessExpression(node.expression) &&
    node.expression.name.text === 'execute'
  );
}

const asyncExecuteVars = new Set();
function collectExecuteAssignments(node) {
  if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.initializer) {
    if (isExecuteCall(node.initializer)) {
      asyncExecuteVars.add(node.name.text);
    }
  }
  if (
    ts.isBinaryExpression(node) &&
    node.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
    ts.isIdentifier(node.left) &&
    isExecuteCall(node.right)
  ) {
    asyncExecuteVars.add(node.left.text);
  }
  ts.forEachChild(node, collectExecuteAssignments);
}

collectExecuteAssignments(sourceFile);

const runtimeErrors = [];
function addRuntimeError(node, message) {
  const pos = sourceFile.getLineAndCharacterOfPosition(node.getStart());
  runtimeErrors.push(`${filePath}:${pos.line + 1}:${pos.character + 1} ${message}`);
}

function inspectRuntimeMisuse(node) {
  if (
    ts.isPropertyAccessExpression(node) &&
    ts.isIdentifier(node.expression) &&
    asyncExecuteVars.has(node.expression.text)
  ) {
    const prop = node.name.text;
    if (!['then', 'catch', 'finally'].includes(prop)) {
      addRuntimeError(
        node,
        `Potential async execute() misuse: ${node.expression.text}.${prop} is accessed synchronously after execute(). Await execute() or use .then(...) before reading properties.`
      );
    }
  }

  if (ts.isPropertyAccessExpression(node) && isExecuteCall(node.expression)) {
    const prop = node.name.text;
    if (!['then', 'catch', 'finally'].includes(prop)) {
      addRuntimeError(
        node,
        `Potential async execute() misuse: execute().${prop} is accessed synchronously. Await execute() or use .then(...) before reading properties.`
      );
    }
  }

  ts.forEachChild(node, inspectRuntimeMisuse);
}

inspectRuntimeMisuse(sourceFile);

const mergedErrors = [...errors];
for (const runtimeError of runtimeErrors) {
  if (!mergedErrors.includes(runtimeError)) {
    mergedErrors.push(runtimeError);
  }
}

console.log(
  JSON.stringify({
    ok: mergedErrors.length === 0,
    validator_available: true,
    error_count: mergedErrors.length,
    errors: mergedErrors,
    runtime_errors: runtimeErrors,
    runtime_error_count: runtimeErrors.length,
  })
);
