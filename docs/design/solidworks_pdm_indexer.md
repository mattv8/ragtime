# SolidWorks PDM Indexer Tool Design

## Overview

The `solidworks_pdm` tool type enables semantic search over SolidWorks PDM metadata by:
1. Connecting to the PDM SQL Server database (same credentials as existing MSSQL tool)
2. Extracting document metadata (part numbers, materials, descriptions, BOMs, etc.)
3. Generating structured text documents suitable for embedding
4. Storing embeddings in pgvector for RAG-powered queries

This complements the existing `mssql` tool (direct SQL queries) with a RAG-based approach that doesn't require users to know SQL or understand the PDM schema.

## Use Cases

| Query Type | Example | How PDM Indexer Helps |
|------------|---------|----------------------|
| Part lookup | "What material is used for SW-13392-A?" | Finds part document, returns Material variable |
| Author search | "Show drawings created by John Smith" | Finds all docs where Author = John Smith |
| Material search | "Find parts with Carbon Steel" | Semantic search across all Material fields |
| BOM queries | "What's the BOM for assembly CU-CL9053?" | Returns parent assembly with all child components |
| Configuration lookup | "What configurations does SW-24779 have?" | Lists all configs with their part numbers |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Ragtime Backend                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ PDM Indexer      │    │ PDM Search Tool  │    │ MSSQL Tool     │ │
│  │ (Background Job) │    │ (RAG Search)     │    │ (Direct SQL)   │ │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬───────┘ │
│           │                       │                       │         │
│           ▼                       ▼                       │         │
│  ┌─────────────────────────────────────┐                 │         │
│  │        pgvector (PostgreSQL)        │                 │         │
│  │   - pdm_embeddings table            │                 │         │
│  │   - pdm_document_metadata table     │                 │         │
│  └─────────────────────────────────────┘                 │         │
│                                                           │         │
└───────────────────────────────────────────────────────────┼─────────┘
                                                            │
                                                            ▼
                                              ┌─────────────────────┐
                                              │ PDM SQL Server      │
                                              │ (HAM-PDM database)  │
                                              │  - Documents        │
                                              │  - VariableValue    │
                                              │  - BomSheets, etc.  │
                                              └─────────────────────┘
```

## Database Schema Changes

### New Prisma Models

```prisma
/// PDM document metadata indexing job status
enum PdmIndexStatus {
  pending
  indexing
  completed
  failed
  cancelled
}

/// PDM indexing job
model PdmIndexJob {
  id           String         @id @default(uuid())
  toolConfigId String         @map("tool_config_id")

  status       PdmIndexStatus @default(pending)
  indexName    String         @map("index_name")

  // Progress tracking
  totalDocuments     Int      @default(0) @map("total_documents")
  processedDocuments Int      @default(0) @map("processed_documents")
  skippedDocuments   Int      @default(0) @map("skipped_documents")
  totalChunks        Int      @default(0) @map("total_chunks")
  processedChunks    Int      @default(0) @map("processed_chunks")
  errorMessage       String?  @map("error_message")

  // Timing
  createdAt   DateTime  @default(now()) @map("created_at")
  startedAt   DateTime? @map("started_at")
  completedAt DateTime? @map("completed_at")

  @@map("pdm_index_jobs")
}

/// PDM document metadata for incremental indexing
model PdmDocumentMetadata {
  id           String   @id @default(uuid())
  indexName    String   @map("index_name")
  documentId   Int      @map("document_id")     // PDM DocumentID
  filename     String
  revisionNo   Int      @map("revision_no")
  metadataHash String   @map("metadata_hash")   // Hash of all variables for change detection
  lastIndexed  DateTime @default(now()) @map("last_indexed")

  @@unique([indexName, documentId])
  @@index([indexName])
  @@map("pdm_document_metadata")
}

/// PDM embeddings stored in pgvector
model PdmEmbedding {
  id           String   @id @default(uuid())
  indexName    String   @map("index_name")
  documentId   Int      @map("document_id")
  documentType String   @map("document_type")   // SLDPRT, SLDASM, SLDDRW, etc.
  content      String   // The structured text content
  // embedding field added via raw SQL migration (vector type)

  // Searchable metadata (duplicated for filtering)
  partNumber   String?  @map("part_number")
  filename     String
  folderPath   String?  @map("folder_path")

  metadata     Json     @default("{}")  // Full variable data as JSON

  createdAt    DateTime @default(now()) @map("created_at")

  @@index([indexName])
  @@index([partNumber])
  @@map("pdm_embeddings")
}
```

### Migration SQL

```sql
-- Add vector column with pgvector
ALTER TABLE pdm_embeddings
ADD COLUMN embedding vector;

-- Create HNSW index for fast similarity search
CREATE INDEX pdm_embeddings_embedding_idx
ON pdm_embeddings
USING hnsw (embedding vector_cosine_ops);
```

## Tool Configuration

### New ToolType Enum Value

Add `solidworks_pdm` to the `ToolType` enum in Prisma schema:

```prisma
enum ToolType {
  postgres
  mssql
  odoo_shell
  ssh_shell
  filesystem_indexer
  solidworks_pdm    // NEW
}
```

### Connection Configuration

```python
class SolidworksPdmConnectionConfig(BaseModel):
    """Connection configuration for SolidWorks PDM indexer tool."""

    # MSSQL Connection (same as MssqlConnectionConfig)
    host: str = Field(description="PDM SQL Server hostname or IP")
    port: int = Field(default=1433, ge=1, le=65535)
    user: str = Field(description="SQL Server username (readonly recommended)")
    password: str = Field(description="SQL Server password")
    database: str = Field(description="PDM database name (e.g., 'HAM-PDM')")

    # Index configuration
    index_name: str = Field(description="Name for this PDM index (used in embeddings)")

    # Document filtering
    file_extensions: List[str] = Field(
        default=["SLDPRT", "SLDASM", "SLDDRW", "dwg"],
        description="File extensions to index (without dot)"
    )
    exclude_patterns: List[str] = Field(
        default=[],
        description="Filename patterns to exclude (glob style)"
    )
    include_deleted: bool = Field(
        default=False,
        description="Include deleted documents (not recommended)"
    )

    # Metadata extraction
    variable_names: List[str] = Field(
        default=[
            "Part Number", "Description", "Material", "Author",
            "Stocked Status", "Finish", "Weight", "Cost",
            "Config Part Number", "Config Description"
        ],
        description="PDM variable names to extract and index"
    )
    include_bom: bool = Field(
        default=True,
        description="Include BOM relationships for assemblies"
    )
    include_folder_path: bool = Field(
        default=True,
        description="Include folder path in indexed content"
    )
    include_revision_history: bool = Field(
        default=False,
        description="Include revision history (increases index size)"
    )

    # Indexing schedule
    reindex_interval_hours: int = Field(
        default=24, ge=0, le=8760,
        description="Hours between automatic re-indexing (0 = manual only)"
    )
    last_indexed_at: Optional[datetime] = None
```

## Indexer Service

### File: `ragtime/indexer/pdm_service.py`

```python
"""
SolidWorks PDM Indexer Service - Creates and manages pgvector-based PDM indexes.

This service handles:
- Connecting to PDM SQL Server database
- Extracting document metadata and variables
- Building structured text for each document
- Storing embeddings in PostgreSQL using pgvector
- Progress tracking and job management
- Incremental indexing (skip unchanged documents based on metadata hash)
"""

class PdmIndexerService:
    """Service for creating and managing SolidWorks PDM indexes with pgvector."""

    def __init__(self):
        self._active_jobs: Dict[str, PdmIndexJob] = {}
        self._cancellation_flags: Dict[str, bool] = {}

    async def trigger_index(
        self,
        tool_config_id: str,
        connection_config: dict,
        full_reindex: bool = False,
    ) -> PdmIndexJob:
        """Trigger PDM metadata indexing."""
        ...

    async def _extract_documents(
        self,
        config: SolidworksPdmConnectionConfig,
    ) -> AsyncIterator[PdmDocument]:
        """Extract documents with metadata from PDM database."""
        ...

    def _build_document_text(self, doc: PdmDocument) -> str:
        """Convert PDM document to embeddable text."""
        ...
```

### Document Text Format

Each PDM document is converted to structured text:

```markdown
# PART: SW-24779.SLDPRT

## Identification
- Part Number: 22-012-02015 (Config: 9PC Swag)
- Filename: SW-24779.SLDPRT
- Folder: \Shared Resources\Parts & Assemblies\Canopies\STANDARD STUDIO CANOPIES\

## Properties
- Description: 9PC Swag Aluminum
- Material: Aluminum
- Stocked Status: BUILT
- Finish: POWDER COAT
- Author: Adam Holt

## Configurations
- 9PC Swag: Part Number 22-012-02015, Description "9PC Swag Aluminum"
- 5PC Linear: Part Number 22-012-02016, Description "5PC Linear Aluminum"

## Revision History
- Rev 1: 2025-12-15 by Adam Holt
- Rev 2: 2026-01-06 by Jorge Gutierrez (Current)

## BOM Components (if assembly)
1. Ground Terminal.SLDPRT (Config: 9"-GND Lug) - Qty: 2
2. 4-40 Pan Head Screw.SLDPRT (Config: 32-012-04111) - Qty: 8
3. IP Nuts.SLDPRT (Config: EL248) - Qty: 4
...
```

## Search Tool

### File: `ragtime/tools/solidworks_pdm.py`

```python
"""
SolidWorks PDM Search Tool - Semantic search over indexed PDM metadata.

This tool provides natural language search over PDM document metadata
that has been indexed into PostgreSQL using pgvector embeddings.
"""

class PdmSearchInput(BaseModel):
    """Input schema for PDM search tool."""

    query: str = Field(
        description="Natural language search query to find PDM documents"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Filter by document type: SLDPRT, SLDASM, SLDDRW, or None for all"
    )
    max_results: int = Field(
        default=10, ge=1, le=50,
        description="Maximum number of results to return"
    )


async def search_pdm_index(
    query: str,
    index_name: str,
    document_type: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """
    Search the PDM index using semantic similarity.
    """
    ...


def create_pdm_search_tool(
    name: str,
    index_name: str,
    description: str = "",
) -> StructuredTool:
    """Create a configured PDM search tool for LangChain."""
    ...
```

## Frontend Changes

### Types (`types/api.ts`)

```typescript
// PDM Indexer Types
export type PdmIndexStatus = 'pending' | 'indexing' | 'completed' | 'failed' | 'cancelled';

export interface PdmIndexJob {
  id: string;
  tool_config_id: string;
  status: PdmIndexStatus;
  index_name: string;
  progress_percent: number;
  total_documents: number;
  processed_documents: number;
  skipped_documents: number;
  total_chunks: number;
  processed_chunks: number;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface SolidworksPdmConnectionConfig {
  host: string;
  port?: number;
  user: string;
  password: string;
  database: string;
  index_name: string;
  file_extensions?: string[];
  exclude_patterns?: string[];
  include_deleted?: boolean;
  variable_names?: string[];
  include_bom?: boolean;
  include_folder_path?: boolean;
  include_revision_history?: boolean;
  reindex_interval_hours?: number;
  last_indexed_at?: string | null;
}
```

### ToolType Info

```typescript
solidworks_pdm: {
  name: 'SolidWorks PDM',
  description: 'Index SolidWorks PDM metadata for semantic search over parts, assemblies, drawings, materials, and BOMs',
  icon: 'cube',  // 3D object icon
  recommended: true,
},
```

### JobsTable Integration

Add `pdmJobs` prop to JobsTable and include in unified job rendering:

```typescript
// In JobsTable.tsx
function toUnifiedPdmJob(job: PdmIndexJob): UnifiedJob {
  let phase = '';
  let progress = 0;

  if (job.status === 'completed') {
    phase = 'Complete';
    progress = 100;
  } else if (job.status === 'pending') {
    phase = 'Queued';
    progress = 0;
  } else if (job.status === 'failed' || job.status === 'cancelled') {
    phase = job.status === 'failed' ? 'Failed' : 'Cancelled';
    progress = 0;
  } else if (job.status === 'indexing') {
    if (job.total_documents > 0) {
      const docProgress = (job.processed_documents / job.total_documents) * 50;
      const chunkProgress = job.total_chunks > 0
        ? (job.processed_chunks / job.total_chunks) * 50
        : 0;
      progress = docProgress + chunkProgress;
      phase = `Indexing: ${job.processed_documents}/${job.total_documents} documents`;
    } else {
      phase = 'Scanning PDM...';
      progress = 5;
    }
  }

  return {
    id: job.id,
    name: job.index_name,
    type: 'pdm',
    status: job.status,
    progress,
    totalFiles: job.total_documents,
    processedFiles: job.processed_documents,
    skippedFiles: job.skipped_documents,
    totalChunks: job.total_chunks,
    processedChunks: job.processed_chunks,
    errorMessage: job.error_message,
    createdAt: job.created_at,
    phase,
    toolConfigId: job.tool_config_id,
  };
}
```

## API Routes

### New Endpoints

```
POST   /tools/{tool_id}/pdm/index           # Trigger PDM indexing
GET    /tools/{tool_id}/pdm/status          # Get indexing status
DELETE /tools/{tool_id}/pdm/index           # Clear PDM index
GET    /tools/{tool_id}/pdm/jobs            # List indexing jobs
POST   /tools/{tool_id}/pdm/jobs/{job_id}/cancel  # Cancel job

# Discovery (for wizard)
POST   /tools/pdm/discover-variables        # Get available PDM variables
GET    /tools/pdm/jobs                      # List all PDM jobs (for JobsTable)
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Add `solidworks_pdm` to ToolType enum
2. Create Prisma models and migration
3. Create `SolidworksPdmConnectionConfig` model
4. Add frontend types

### Phase 2: Indexer Service
1. Create `ragtime/indexer/pdm_service.py`
2. Implement PDM document extraction queries
3. Implement document-to-text conversion
4. Implement embedding storage
5. Implement incremental indexing

### Phase 3: Search Tool
1. Create `ragtime/tools/solidworks_pdm.py`
2. Implement pgvector search function
3. Register tool with LangChain agent

### Phase 4: API & UI
1. Add API routes for indexing/status
2. Add ToolWizard panel for solidworks_pdm
3. Integrate with JobsTable
4. Add variable discovery for wizard

### Phase 5: Testing & Polish
1. Test with real PDM data
2. Tune embedding text format
3. Add error handling
4. Performance optimization (batch queries)

## Key SQL Queries for PDM Extraction

### Get Documents with Metadata

```sql
SELECT TOP 1000
    d.DocumentID,
    d.Filename,
    d.LatestRevisionNo,
    d.CurrentStatusID,
    p.Path as FolderPath,
    RIGHT(d.Filename, CHARINDEX('.', REVERSE(d.Filename)) - 1) as Extension
FROM Documents d
LEFT JOIN DocumentsInProjects dip ON d.DocumentID = dip.DocumentID
LEFT JOIN Projects p ON dip.ProjectID = p.ProjectID
WHERE d.Deleted = 0
  AND d.Filename LIKE '%.SLDPRT' OR d.Filename LIKE '%.SLDASM' OR d.Filename LIKE '%.SLDDRW'
ORDER BY d.DocumentID
```

### Get Variables for Document

```sql
SELECT
    v.VariableName,
    COALESCE(vv.ValueText, CAST(vv.ValueInt AS NVARCHAR), CAST(vv.ValueFloat AS NVARCHAR)) as Value,
    dc.ConfigurationName
FROM VariableValue vv
JOIN Variable v ON vv.VariableID = v.VariableID
LEFT JOIN DocumentConfiguration dc ON vv.ConfigurationID = dc.ConfigurationID
WHERE vv.DocumentID = @DocumentID
  AND v.VariableName IN ('Part Number', 'Description', 'Material', 'Author', ...)
  AND (vv.ValueText IS NOT NULL OR vv.ValueInt IS NOT NULL OR vv.ValueFloat IS NOT NULL)
ORDER BY v.VariableName, dc.ConfigurationName
```

### Get BOM for Assembly

```sql
SELECT
    bsr.RowNo,
    child.Filename as ComponentFile,
    dc.ConfigurationName as ComponentConfig,
    bsv.CellText as Quantity  -- BomSheetValue for QTY column
FROM BomSheets bs
JOIN Documents parent ON bs.SourceDocumentID = parent.DocumentID
JOIN BomSheetRow bsr ON bs.BomDocumentID = bsr.BomDocumentID
JOIN Documents child ON bsr.RowDocumentID = child.DocumentID
LEFT JOIN DocumentConfiguration dc ON bsr.RowConfigurationID = dc.ConfigurationID
LEFT JOIN BomSheetValue bsv ON bsr.BomDocumentID = bsv.BomDocumentID
    AND bsr.RowNo = bsv.RowNo AND bsv.ColNo = 1  -- Column 1 is typically QTY
WHERE parent.DocumentID = @DocumentID
ORDER BY bsr.RowNo
```

## Security Considerations

1. **Read-only access**: The PDM SQL user should have SELECT-only permissions
2. **Credential storage**: Connection passwords stored encrypted in tool_configs
3. **Query validation**: All SQL queries are hardcoded, no user input in SQL
4. **Rate limiting**: Batch queries to avoid overwhelming PDM database
5. **Timeout protection**: Query timeouts to prevent hanging

## Performance Considerations

1. **Batch processing**: Query documents in batches of 100-500
2. **Incremental indexing**: Hash metadata to skip unchanged documents
3. **Async operations**: Non-blocking embedding generation
4. **Connection pooling**: Reuse MSSQL connections
5. **Progress updates**: Update job status every N documents, not every document
