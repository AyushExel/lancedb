/* tslint:disable */
/* eslint-disable */

/* auto-generated by NAPI-RS */

export const enum IndexType {
  Scalar = 0,
  IvfPq = 1
}
export const enum MetricType {
  L2 = 0,
  Cosine = 1,
  Dot = 2
}
export interface ConnectionOptions {
  uri: string
  apiKey?: string
  hostOverride?: string
}
/** Write mode for writing a table. */
export const enum WriteMode {
  Create = 'Create',
  Append = 'Append',
  Overwrite = 'Overwrite'
}
/** Write options when creating a Table. */
export interface WriteOptions {
  mode?: WriteMode
}
export function connect(options: ConnectionOptions): Promise<Connection>
export class Connection {
  /** Create a new Connection instance from the given URI. */
  static new(uri: string): Promise<Connection>
  /** List all tables in the dataset. */
  tableNames(): Promise<Array<string>>
  /**
   * Create table from a Apache Arrow IPC (file) buffer.
   *
   * Parameters:
   * - name: The name of the table.
   * - buf: The buffer containing the IPC file.
   *
   */
  createTable(name: string, buf: Buffer): Promise<Table>
  openTable(name: string): Promise<Table>
  /** Drop table with the name. Or raise an error if the table does not exist. */
  dropTable(name: string): Promise<void>
}
export class IndexBuilder {
  replace(v: boolean): void
  column(c: string): void
  name(name: string): void
  ivfPq(metricType?: MetricType | undefined | null, numPartitions?: number | undefined | null, numSubVectors?: number | undefined | null, numBits?: number | undefined | null, maxIterations?: number | undefined | null, sampleRate?: number | undefined | null): void
  scalar(): void
  build(): Promise<void>
}
/** Typescript-style Async Iterator over RecordBatches  */
export class RecordBatchIterator {
  next(): Promise<Buffer | null>
}
export class Query {
  column(column: string): void
  filter(filter: string): void
  select(columns: Array<string>): void
  limit(limit: number): void
  prefilter(prefilter: boolean): void
  nearestTo(vector: Float32Array): void
  refineFactor(refineFactor: number): void
  nprobes(nprobe: number): void
  executeStream(): Promise<RecordBatchIterator>
}
export class Table {
  /** Return Schema as empty Arrow IPC file. */
  schema(): Buffer
  add(buf: Buffer): Promise<void>
  countRows(): Promise<bigint>
  delete(predicate: string): Promise<void>
  createIndex(): IndexBuilder
  query(): Query
}
