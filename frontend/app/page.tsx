"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { CSSProperties } from "react";

// Metrics computation
interface Metrics {
  rep3gram: number;
  uniqueRatio: number;
  numTokens: number;
  numSentences: number;
  avgTokenRepeat: number;
}

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/\b\w+\b/g) || []);
}

function splitSentences(text: string): string[] {
  return text.split(/[.!?]+/).map(s => s.trim()).filter(Boolean);
}

function repetition3gram(tokens: string[]): number {
  if (tokens.length < 3) return 0;
  const counts = new Map<string, number>();
  for (let i = 0; i < tokens.length - 2; i++) {
    const tri = `${tokens[i]} ${tokens[i + 1]} ${tokens[i + 2]}`;
    counts.set(tri, (counts.get(tri) || 0) + 1);
  }
  let repeated = 0;
  counts.forEach(c => { if (c > 1) repeated += c; });
  return repeated / (tokens.length - 2);
}

function uniqueRatio(tokens: string[]): number {
  if (!tokens.length) return 0;
  return new Set(tokens).size / tokens.length;
}

function avgTokenRepeat(tokens: string[]): number {
  if (!tokens.length) return 0;
  const counts = new Map<string, number>();
  tokens.forEach(t => counts.set(t, (counts.get(t) || 0) + 1));
  let totalRepeat = 0;
  counts.forEach(c => { if (c > 1) totalRepeat += (c - 1); });
  return totalRepeat / tokens.length;
}

function computeMetrics(text: string): Metrics {
  const tokens = tokenize(text);
  const sentences = splitSentences(text);
  return {
    rep3gram: repetition3gram(tokens),
    uniqueRatio: uniqueRatio(tokens),
    numTokens: tokens.length,
    numSentences: sentences.length,
    avgTokenRepeat: avgTokenRepeat(tokens)
  };
}

// Get color for metric value based on thresholds
// Returns: green (good), yellow (ok), red (bad)
function getMetricColor(metric: string, value: number): string {
  switch (metric) {
    case 'rep3gram':
      // Lower is better: <0.15 good, <0.30 ok, >=0.30 bad
      if (value < 0.15) return 'green';
      if (value < 0.30) return '#d4a000'; // gold/yellow
      return '#c33'; // red
    case 'uniqueRatio':
      // Higher is better (within reason): >0.5 good, >0.35 ok, <=0.35 bad
      if (value > 0.5) return 'green';
      if (value > 0.35) return '#d4a000';
      return '#c33';
    case 'avgTokenRepeat':
      // Lower is better: <0.2 good, <0.5 ok, >=0.5 bad
      if (value < 0.2) return 'green';
      if (value < 0.5) return '#d4a000';
      return '#c33';
    default:
      return '#666';
  }
}

// Highlight current experiment row in results.tsv by commit ID
function highlightCurrentRow(resultsTsv: string, expDir: string): React.ReactNode {
  // Extract commit hash from exp_dir (format: YYYYMMDD_HHMMSS-commit_hash)
  const match = expDir.match(/-(.+)$/);
  const commitHash = match ? match[1].trim() : null;

  const lines = resultsTsv.split('\n');
  if (lines.length === 0) return resultsTsv;

  const [header, ...dataLines] = lines;
  const headerCells = header.split('\t');
  const commitIdx = headerCells.findIndex(h => h.toLowerCase() === 'commit');

  if (commitIdx === -1 || !commitHash) {
    return resultsTsv; // No commit column or no hash to match
  }

  // Reverse data lines to show newest first
  const reversedDataLines = [...dataLines].reverse();

  const renderedLines: React.ReactNode[] = [];

  for (let i = 0; i < reversedDataLines.length; i++) {
    const line = reversedDataLines[i];
    if (!line.trim()) continue;
    const cells = line.split('\t');
    const cellValue = cells[commitIdx] || '';

    if (commitHash && cellValue.includes(commitHash)) {
      // Highlight this line with background color
      renderedLines.push(
        <div key={`highlight-${i}`} style={{ backgroundColor: '#fff3cd', display: 'block', padding: '2px 0' }}>
          {line}
        </div>
      );
    } else {
      renderedLines.push(
        <div key={`line-${i}`} style={{ padding: '2px 0' }}>
          {line}
        </div>
      );
    }
  }

  return <>{renderedLines}</>;
}

// Stop at EOS/reserved token - detect <|reserved|> or <|reserved_0|> and truncate
function stripAfterReserved(text: string): string {
  const eos1 = "<|reserved|>";
  const eos2 = "<|reserved_0|>";
  const idx1 = text.indexOf(eos1);
  const idx2 = text.indexOf(eos2);
  let minIdx = -1;
  if (idx1 !== -1 && idx2 !== -1) {
    minIdx = Math.min(idx1, idx2);
  } else if (idx1 !== -1) {
    minIdx = idx1;
  } else if (idx2 !== -1) {
    minIdx = idx2;
  }
  return minIdx === -1 ? text : text.slice(0, minIdx);
}

interface GenerateResponse {
  generated_text: string;
  generated_tokens: number[];
  num_tokens: number;
  prompt_used: string | null;
}

interface SampleOutput {
  sample_content: string | null;
  model_info: string | null;
}

interface GitLogCommit {
  commit_sha: string;
  commit_short: string;
  author: string;
  date: string;
  message: string;
}

interface ExperimentInfo {
  exp_dir: string;
  checkpoint_path: string | null;
  has_checkpoint: boolean;
  has_program: boolean;
  has_results: boolean;
  has_git_log: boolean;
  has_global_results: boolean;
  has_metadata: boolean;
  git_log: GitLogCommit[];
  results_data: {
    header: string[];
    rows: string[][];
  } | null;
  metadata: Record<string, any> | null;
}

interface ExperimentDir {
  exp_dir: string;
  checkpoint_path: string | null;
  has_checkpoint: boolean;
  sample_output: string | null;
  model_info: string | null;
  results_data?: {
    header: string[];
    rows: string[][];
  } | null;
  prompt?: string;
}

interface ListResponse {
  experiments: ExperimentDir[];
  count: number;
}

// Info icon component with tooltip
function InfoIcon({ text }: { text: string }) {
  return (
    <span
      style={{
        position: "relative",
        display: "inline-block",
        width: "14px",
        height: "14px",
        borderRadius: "50%",
        background: "#666",
        color: "#fff",
        fontSize: "10px",
        lineHeight: "14px",
        textAlign: "center",
        cursor: "default",
        fontFamily: "sans-serif",
        fontStyle: "normal",
        marginLeft: "4px",
        verticalAlign: "middle",
      }}
      title={text}
    >
      i
    </span>
  );
}

const styles: Record<string, CSSProperties> = {
  main: {
    minHeight: "100vh",
    backgroundColor: "#f5f5f5",
    padding: "10px 20px",
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  header: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    display: "flex",
    justifyContent: "center",
    padding: "6px 0",
    backgroundColor: "#f5f5f5",
    zIndex: 1,
  },
  headerContent: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  headerText: {
    fontSize: "11px",
    color: "#666",
  },
  headerLink: {
    color: "#666",
    textDecoration: "none",
    display: "flex",
    alignItems: "center",
  },
  githubIcon: {
    display: "block",
  },
  headerDivider: {
    fontSize: "10px",
    color: "#ccc",
  },
  authorLink: {
    fontSize: "11px",
    color: "#666",
    textDecoration: "none",
  },
  navLink: {
    fontSize: "11px",
    color: "#666",
    textDecoration: "none",
  },
  cardsContainer: {
    maxWidth: "800px",
    width: "100%",
    margin: "0 auto",
    paddingTop: "45px",
  },
  container: {
    maxWidth: "800px",
    width: "100%",
    backgroundColor: "white",
    padding: "12px",
    borderRadius: "6px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
    marginBottom: "10px",
  },
  title: {
    fontSize: "18px",
    fontWeight: "bold",
    marginBottom: "4px",
  },
  subtitle: {
    fontSize: "12px",
    color: "#666",
    marginBottom: "10px",
  },
  box: {
    padding: "10px",
    backgroundColor: "#f9f9f9",
    borderRadius: "4px",
  },
  error: {
    backgroundColor: "#fee",
    color: "#c33",
  },
  result: {
    padding: "10px",
    backgroundColor: "#f9f9f9",
    borderRadius: "4px",
  },
  row: {
    marginBottom: "6px",
    lineHeight: "1.5",
  },
  metadataSection: {
    marginTop: "10px",
    borderTop: "1px solid #eee",
    paddingTop: "10px",
  },
  metadataLabel: {
    fontSize: "11px",
    fontWeight: "bold",
    color: "#333",
    marginBottom: "4px",
  },
  metadataValue: {
    fontSize: "11px",
    color: "#666",
    marginBottom: "8px",
  },
  expBadge: {
    display: "inline-block",
    padding: "2px 6px",
    backgroundColor: "#e3f2fd",
    color: "#1976d2",
    borderRadius: "3px",
    fontSize: "10px",
    fontWeight: "bold",
  },
  dropdownContainer: {
    marginTop: "8px",
  },
  dropdownButton: {
    padding: "6px 12px",
    backgroundColor: "#fff",
    border: "1px solid #ddd",
    borderRadius: "4px",
    fontSize: "11px",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "8px",
  },
  dropdownContent: {
    padding: "10px",
    backgroundColor: "#f9f9f9",
    borderRadius: "4px",
    marginTop: "6px",
  },
  commitItem: {
    padding: "6px",
    backgroundColor: "#fff",
    border: "1px solid #eee",
    borderRadius: "3px",
    marginBottom: "6px",
    fontSize: "10px",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "10px",
  },
  th: {
    textAlign: "left",
    padding: "4px 8px",
    backgroundColor: "#f0f0f0",
    fontWeight: "bold",
  },
  td: {
    padding: "4px 8px",
    border: "1px solid #eee",
  },
  arrow: {
    fontSize: "8px",
  },
  loading: {
    padding: "20px",
    textAlign: "center",
    color: "#666",
  },
};

function Header() {
  return (
    <header style={styles.header}>
      <div style={styles.headerContent}>
        <div style={styles.headerText}>Watch autoresearch make progress on an actual text prompt.</div>
        <span style={styles.headerDivider}>|</span>
        <a
          href="https://github.com/TerkaSlan/autoresearch-win-rtx"
          target="_blank"
          rel="noopener noreferrer"
          style={styles.headerLink}
        >
          <svg
            viewBox="0 0 16 16"
            version="1.1"
            width="14"
            height="14"
            style={styles.githubIcon}
            aria-hidden="true"
          >
            <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path>
          </svg>
        </a>
        <span style={styles.headerDivider}>|</span>
        <Link href="/about" style={styles.navLink}>About</Link>
        <span style={styles.headerDivider}>|</span>
        <a
          href="https://terkaslan.github.io/"
          target="_blank"
          rel="noopener noreferrer"
          style={styles.authorLink}
          className="author-link"
        >
          Created by Terézia Slanináková
        </a>
      </div>
    </header>
  );
}

interface ExperimentCardProps {
  experiment: ExperimentDir;
  prevValBpb?: number;
  sampleData?: SampleOutput;
}

function ExperimentCard({ experiment, prevValBpb, sampleData }: ExperimentCardProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expInfo, setExpInfo] = useState<ExperimentInfo | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [resultsTsv, setResultsTsv] = useState<string | null>(null);

  const API_BASE_URL = "/api/proxy";

  useEffect(() => {
    const fetchData = async () => {
      try {
        const infoResponse = await fetch(`${API_BASE_URL}/info/${experiment.exp_dir}`);
        if (!infoResponse.ok) {
          throw new Error(`API error: ${infoResponse.status}`);
        }
        const infoData: ExperimentInfo = await infoResponse.json();
        setExpInfo(infoData);

        // Fetch results.tsv.global
        const fileResponse = await fetch(`${API_BASE_URL}/file/${experiment.exp_dir}?filename=results.tsv.global`);
        if (fileResponse.ok) {
          const fileData = await fileResponse.json();
          if (fileData.content) {
            setResultsTsv(fileData.content);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [experiment]);

  // Compute metrics from sample output
  const rawSample = sampleData?.sample_content || experiment.sample_output || "";
  const cleanSampleLines = rawSample.split('\n')
    .filter(line => !line.startsWith('#'))
    .map(line => line.replace(/^\u0000/, '').trim())
    .filter(line => line)
    .join('\n')
    .trim();
  const cleanGeneratedText = stripAfterReserved(cleanSampleLines);

  // Compute metrics when text is available
  useEffect(() => {
    if (cleanGeneratedText) {
      setMetrics(computeMetrics(cleanGeneratedText));
    }
  }, [cleanGeneratedText]);

  
  // Parse timestamp from exp_dir (format: YYYYMMDD_HHMMSS-<commit_short>)
  const parseTimestamp = (expDir: string | null | undefined): { full: string; date: string; time: string } => {
    if (!expDir) {
      return { full: "", date: "", time: "" };
    }
    const match = expDir.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
    if (!match) {
      return { full: expDir, date: expDir, time: "" };
    }
    const [, year, month, day, hour, minute, second] = match;
    const dateStr = `${year}-${month}-${day}`;
    const timeStr = `${hour}:${minute}:${second}`;
    return { full: expDir, date: dateStr, time: timeStr };
  };

  // Extract val_bpb from metadata or results.tsv
  const valBpbFromModelInfo = sampleData?.model_info && sampleData.model_info.includes('val_bpb:')
    ? parseFloat(sampleData.model_info.split('val_bpb:')[1].split('\n')[0].trim())
    : null;

  // Also try to extract from results.tsv
  const valBpbFromResults = resultsTsv ? (() => {
    const lines = resultsTsv.split('\n').filter(l => l.trim());
    const header = lines[0].split('\t');
    const valBpbIdx = header.findIndex(h => h.toLowerCase() === 'val_bpb');
    if (valBpbIdx !== -1 && lines.length > 1) {
      const lastRow = lines[lines.length - 1].split('\t');
      return parseFloat(lastRow[valBpbIdx]);
    }
    return null;
  })() : null;

  const valBpb = valBpbFromModelInfo ?? valBpbFromResults;

  // Use experiment.exp_dir for timestamp if expInfo not loaded yet
  const timestamp = expInfo ? parseTimestamp(expInfo.exp_dir) : parseTimestamp(experiment.exp_dir);
  const latestCommit = expInfo?.git_log?.[0];

  return (
    <div style={styles.container}>
      <div style={{display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px"}}>
        <div style={{display: "flex", alignItems: "center", gap: "8px"}}>
          {timestamp.date || timestamp.full ? (
            <h1 style={{...styles.title, margin: 0}}>
              {timestamp.date ? `${timestamp.date} ${timestamp.time || ''}` : timestamp.full}
            </h1>
          ) : (
            <h1 style={{...styles.title, margin: 0}}>Experiment: {experiment.exp_dir}</h1>
          )}
        </div>
        {(valBpb !== null || latestCommit) && (
          <div style={{fontSize: "12px", color: "#666", textAlign: "right", display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "2px"}}>
            {valBpb !== null && (
              <div>
                <span>val_bpb</span><InfoIcon text="Validation bits per byte - lower is better" />: {valBpb.toFixed(3)}
              </div>
            )}
            {latestCommit && (
              <div style={{position: "relative", display: "inline-block"}}>
                <span style={{color: "#1976d2", cursor: "help"}}>{latestCommit.commit_short}</span>
                <div style={{
                  position: "absolute",
                  left: 0,
                  bottom: "100%",
                  marginBottom: "8px",
                  backgroundColor: "#333",
                  color: "#fff",
                  padding: "10px",
                  borderRadius: "6px",
                  fontSize: "11px",
                  lineHeight: "1.4",
                  zIndex: 100,
                  display: "none",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                  maxWidth: "300px",
                  wordBreak: "break-word"
                }} onMouseEnter={(e) => (e.currentTarget.style.display = "block")} onMouseLeave={(e) => (e.currentTarget.style.display = "none")}>
                  <div style={{fontWeight: "bold", marginBottom: "6px"}}>{latestCommit.commit_short} — {latestCommit.commit_sha}</div>
                  <div style={{color: "#aaa", fontSize: "10px"}}>{latestCommit.author || "Unknown"} | {latestCommit.date || "Unknown"}</div>
                  <div style={{marginTop: "6px", color: "#fff"}}>{latestCommit.message.trim()}</div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {loading && (
        <div style={styles.box}>
          <p>Loading...</p>
        </div>
      )}

      {!loading && error && (
        <div style={{...styles.box, ...styles.error}}>
          <p><strong>Error:</strong> {error}</p>
          <button onClick={() => { setError(null); setLoading(true); }} style={{fontSize: "11px", marginTop: "4px"}}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && (
        <>
          {/* Sample Output */}
          {cleanGeneratedText && (
            <div style={styles.result}>
              <div style={styles.row}>
                <strong>Prompt:</strong> Once upon a time
              </div>
              <div style={styles.row}>
                <strong>Generated Output:</strong>
              </div>
              <div style={{...styles.row, whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '11px'}}>
                {cleanGeneratedText}
              </div>
            </div>
          )}

          {/* Metrics */}
          {metrics && cleanGeneratedText && (
            <div style={{marginTop: "8px"}}>
              <button
                style={styles.dropdownButton}
                onClick={() => setShowMetrics(!showMetrics)}
              >
                <span><strong>Metrics</strong></span>
                <span style={styles.arrow}>{showMetrics ? "▼" : "▶"}</span>
              </button>
              {showMetrics && (
                <div style={{...styles.dropdownContent, backgroundColor: "transparent", padding: 0}}>
                  <div style={styles.row}>
                    <strong>rep3gram:</strong> <span style={{color: getMetricColor('rep3gram', metrics.rep3gram)}}>{metrics.rep3gram.toFixed(4)}</span> <small style={{color: '#888'}}>(&lt;0.15 good, &lt;0.30 ok)</small>
                  </div>
                  <div style={styles.row}>
                    <strong>uniqueRatio:</strong> <span style={{color: getMetricColor('uniqueRatio', metrics.uniqueRatio)}}>{metrics.uniqueRatio.toFixed(4)}</span> <small style={{color: '#888'}}>(&gt;0.5 good, &gt;0.35 ok)</small>
                  </div>
                  <div style={styles.row}>
                    <strong>avgTokenRepeat:</strong> <span style={{color: getMetricColor('avgTokenRepeat', metrics.avgTokenRepeat)}}>{metrics.avgTokenRepeat.toFixed(4)}</span> <small style={{color: '#888'}}>(&lt;0.2 good, &lt;0.5 ok)</small>
                  </div>
                  <div style={styles.row}>
                    <strong>numTokens:</strong> {metrics.numTokens}
                  </div>
                  <div style={styles.row}>
                    <strong>numSentences:</strong> {metrics.numSentences}
                  </div>
                </div>
              )}
            </div>
          )}

          {expInfo && resultsTsv && (
            <div style={styles.metadataSection}>
              <button
                style={styles.dropdownButton}
                onClick={() => setShowResults(!showResults)}
              >
                <span><strong>results.tsv</strong></span>
                <span style={styles.arrow}>{showResults ? "▼" : "▶"}</span>
              </button>
              {showResults && (
                <div style={{...styles.dropdownContent, whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '10px', maxHeight: '300px', overflow: 'auto'}}>
                  {highlightCurrentRow(resultsTsv, experiment.exp_dir)}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function Home() {
  const [experiments, setExperiments] = useState<ExperimentDir[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sampleData, setSampleData] = useState<{[key: string]: SampleOutput}>({});
  const [valBpbData, setValBpbData] = useState<{[key: string]: number}>({});

  const API_BASE_URL = "/api/proxy";

  useEffect(() => {
    const fetchExperiments = async () => {
      try {
        const response = await fetch("/api/proxy/list");
        if (!response.ok) {
          throw new Error(`Failed to fetch experiments: ${response.status}`);
        }
        const data: ListResponse = await response.json();
        const filteredExperiments = data.experiments?.filter(exp => exp.has_checkpoint) || [];

        // Display experiments immediately with their pre-computed samples
        setExperiments(filteredExperiments);

        // Preload sample data and val_bpb for visible experiments
        const promises = filteredExperiments.map(async (exp) => {
          try {
            let valBpb = null;
            const sample: SampleOutput = {
              sample_content: exp.sample_output || null,
              model_info: exp.model_info || null
            };

            // Try to get val_bpb from results if available
            if (exp.results_data?.header && exp.results_data?.rows) {
              const valBpbIdx = exp.results_data.header.indexOf('val_bpb');
              if (valBpbIdx !== -1 && exp.results_data.rows.length > 0) {
                const row = exp.results_data.rows[exp.results_data.rows.length - 1];
                const val = row?.[valBpbIdx];
                if (val) valBpb = parseFloat(val);
              }
            }

            setSampleData(prev => ({ ...prev, [exp.exp_dir]: sample }));
            if (valBpb !== null) {
              setValBpbData(prev => ({ ...prev, [exp.exp_dir]: valBpb }));
            }
          } catch (err) {
            console.error(`Failed to load data for ${exp.exp_dir}:`, err);
          }
        });

        await Promise.all(promises);
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
        setLoading(false);
      }
    };

    fetchExperiments();
  }, []);

  return (
    <main style={styles.main}>
      <Header />
      <div style={styles.cardsContainer}>
        {loading && (
          <div style={styles.loading}>Loading experiments...</div>
        )}
        {error && (
          <div style={{...styles.box, ...styles.error}}>
            <p><strong>Error:</strong> {error}</p>
          </div>
        )}
        {!loading && !error && experiments.length === 0 && (
          <div style={styles.loading}>No experiments found</div>
        )}
        {!loading && !error && experiments.length > 0 && (
          <>
            {experiments.map((exp, idx) => (
              <ExperimentCard
                key={exp.exp_dir}
                experiment={exp}
                prevValBpb={idx < experiments.length - 1 ? valBpbData[experiments[idx + 1].exp_dir] : undefined}
                sampleData={sampleData[exp.exp_dir]}
              />
            ))}
          </>
        )}
      </div>
    </main>
  );
}