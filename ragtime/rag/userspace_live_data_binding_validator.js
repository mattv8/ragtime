const fs = require('fs');
const path = require('path');

const input = fs.readFileSync(0, 'utf8');
const filePath = process.argv[2] || 'dashboard/main.ts';

let ts;
try {
  const tsModulePath = path.join('/ragtime/ragtime/frontend/node_modules/typescript');
  ts = require(tsModulePath);
} catch (err) {
  console.log(
    JSON.stringify({
      ok: false,
      validator_available: false,
      message: 'TypeScript compiler unavailable for AST analysis',
      has_execute_calls: false,
      found_component_ids: [],
      has_local_imports: false,
      has_context_components_access: false,
    })
  );
  process.exit(0);
}

const scriptKind = filePath.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
const sourceFile = ts.createSourceFile(
  filePath,
  input,
  ts.ScriptTarget.ES2020,
  true,
  scriptKind
);

const foundComponentIds = [];
const foundIdSet = new Set();
let hasAnyExecuteCall = false;
let hasLocalImports = false;
let hasContextComponentsAccess = false;

function visit(node) {
  if (
    ts.isImportDeclaration(node) &&
    node.moduleSpecifier &&
    ts.isStringLiteral(node.moduleSpecifier)
  ) {
    const spec = node.moduleSpecifier.text;
    if (spec.startsWith('./') || spec.startsWith('../')) {
      hasLocalImports = true;
    }
  }

  if (
    ts.isCallExpression(node) &&
    node.expression.kind === ts.SyntaxKind.ImportKeyword &&
    node.arguments.length > 0 &&
    ts.isStringLiteral(node.arguments[0])
  ) {
    const spec = node.arguments[0].text;
    if (spec.startsWith('./') || spec.startsWith('../')) {
      hasLocalImports = true;
    }
  }

  if (ts.isPropertyAccessExpression(node) && node.name.text === 'components') {
    const obj = node.expression;
    if (ts.isIdentifier(obj) && obj.text === 'context') {
      hasContextComponentsAccess = true;
    }
  }

  if (ts.isCallExpression(node)) {
    const expr = node.expression;
    if (ts.isPropertyAccessExpression(expr) && expr.name.text === 'execute') {
      const obj = expr.expression;

      if (ts.isElementAccessExpression(obj)) {
        const container = obj.expression;
        if (
          ts.isPropertyAccessExpression(container) &&
          container.name.text === 'components'
        ) {
          const root = container.expression;
          if (ts.isIdentifier(root) && root.text === 'context') {
            hasAnyExecuteCall = true;
            hasContextComponentsAccess = true;
            const arg = obj.argumentExpression;
            if (ts.isStringLiteral(arg) && !foundIdSet.has(arg.text)) {
              foundIdSet.add(arg.text);
              foundComponentIds.push(arg.text);
            }
          }
        }
      }

      if (ts.isPropertyAccessExpression(obj)) {
        const container = obj.expression;
        if (
          ts.isPropertyAccessExpression(container) &&
          container.name.text === 'components'
        ) {
          const root = container.expression;
          if (ts.isIdentifier(root) && root.text === 'context') {
            hasAnyExecuteCall = true;
            hasContextComponentsAccess = true;
            const propName = obj.name.text;
            if (!foundIdSet.has(propName)) {
              foundIdSet.add(propName);
              foundComponentIds.push(propName);
            }
          }
        }
      }
    }
  }

  ts.forEachChild(node, visit);
}

visit(sourceFile);

console.log(
  JSON.stringify({
    ok: true,
    validator_available: true,
    has_execute_calls: hasAnyExecuteCall,
    found_component_ids: foundComponentIds,
    has_local_imports: hasLocalImports,
    has_context_components_access: hasContextComponentsAccess,
  })
);
