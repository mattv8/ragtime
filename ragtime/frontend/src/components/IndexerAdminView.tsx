import { FilesystemIndexPanel } from './FilesystemIndexPanel';
import { IndexesList } from './IndexesList';
import { JobsTable } from './JobsTable';
import type {
  FilesystemIndexJob,
  IndexInfo,
  IndexJob,
  PdmIndexJob,
  SchemaIndexJob,
  UserSpaceCodeIndexJob,
} from '@/types';

interface IndexerAdminViewProps {
  indexes: IndexInfo[];
  jobs: IndexJob[];
  indexesLoading: boolean;
  indexesError: string | null;
  jobsLoading: boolean;
  jobsError: string | null;
  filesystemJobs: FilesystemIndexJob[];
  schemaJobs: SchemaIndexJob[];
  pdmJobs: PdmIndexJob[];
  userspaceCodeJobs: UserSpaceCodeIndexJob[];
  aggregateSearch: boolean;
  embeddingDimensions: number | null;
  onLoadIndexes: () => void;
  onJobCreated: () => void;
  onNavigateToSettings: () => void;
  onToolsChanged: () => Promise<void> | void;
  onFilesystemJobsChanged: () => Promise<void> | void;
  onJobsChanged: () => Promise<void> | void;
  onSchemaJobsChanged: () => Promise<void> | void;
  onPdmJobsChanged: () => Promise<void> | void;
  onUserSpaceCodeJobsChanged: () => Promise<void> | void;
  onCancelFilesystemJob: (toolId: string, jobId: string) => Promise<void>;
  onCancelSchemaJob: (toolId: string, jobId: string) => Promise<void>;
  onCancelPdmJob: (toolId: string, jobId: string) => Promise<void>;
}

export function IndexerAdminView({
  indexes,
  jobs,
  indexesLoading,
  indexesError,
  jobsLoading,
  jobsError,
  filesystemJobs,
  schemaJobs,
  pdmJobs,
  userspaceCodeJobs,
  aggregateSearch,
  embeddingDimensions,
  onLoadIndexes,
  onJobCreated,
  onNavigateToSettings,
  onToolsChanged,
  onFilesystemJobsChanged,
  onJobsChanged,
  onSchemaJobsChanged,
  onPdmJobsChanged,
  onUserSpaceCodeJobsChanged,
  onCancelFilesystemJob,
  onCancelSchemaJob,
  onCancelPdmJob,
}: IndexerAdminViewProps) {
  return (
    <>
      <IndexesList
        indexes={indexes}
        jobs={jobs}
        loading={indexesLoading}
        error={indexesError}
        onDelete={onLoadIndexes}
        onToggle={onLoadIndexes}
        onDescriptionUpdate={onLoadIndexes}
        onJobCreated={onJobCreated}
        aggregateSearch={aggregateSearch}
        embeddingDimensions={embeddingDimensions}
        onNavigateToSettings={onNavigateToSettings}
      />

      <FilesystemIndexPanel
        onToolsChanged={onToolsChanged}
        onJobsChanged={onFilesystemJobsChanged}
        embeddingDimensions={embeddingDimensions}
      />

      <JobsTable
        jobs={jobs}
        filesystemJobs={filesystemJobs}
        schemaJobs={schemaJobs}
        pdmJobs={pdmJobs}
        userspaceCodeJobs={userspaceCodeJobs}
        loading={jobsLoading}
        error={jobsError}
        onJobsChanged={onJobsChanged}
        onFilesystemJobsChanged={onFilesystemJobsChanged}
        onSchemaJobsChanged={onSchemaJobsChanged}
        onPdmJobsChanged={onPdmJobsChanged}
        onUserSpaceCodeJobsChanged={onUserSpaceCodeJobsChanged}
        onCancelFilesystemJob={onCancelFilesystemJob}
        onCancelSchemaJob={onCancelSchemaJob}
        onCancelPdmJob={onCancelPdmJob}
      />
    </>
  );
}
