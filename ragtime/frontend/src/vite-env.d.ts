/// <reference types="vite/client" />

interface ImportMetaEnv {
	readonly VITE_RAGTIME_ENVIRONMENT?: string;
	readonly VITE_RAGTIME_VERSION?: string;
}

interface ImportMeta {
	readonly env: ImportMetaEnv;
}
