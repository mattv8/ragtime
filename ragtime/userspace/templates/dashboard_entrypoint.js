import React from 'react';
import { createRoot } from 'react-dom/client';
import App from '__RAGTIME_DASHBOARD_MODULE_SPECIFIER__';

const rootElement = document.getElementById('root') || (() => {
  const element = document.createElement('div');
  element.id = 'root';
  document.body.appendChild(element);
  return element;
})();

createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
