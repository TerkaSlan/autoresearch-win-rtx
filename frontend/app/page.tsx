"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { CSSProperties } from "react";

// Stop at EOS/reserved token - detect <|reserved|> or <|reserved_0|> and truncate
function stripAfterReserved(text: string): string {
  // Check for both token variants
  const eos1 = "<|reserved|>";
  const eos2 = "<|reserved_0|>";

  const idx1 = text.indexOf(eos1);
  const idx2 = text.indexOf(eos2);

  // Find the earliest occurrence
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
  name: string;
  checkpoint_path: string | null;
  has_checkpoint: boolean;
  sample_output: string | null;
  model_info: string | null;
  results_data?: {
    header: string[];
    rows: string[][];
  } | null;
}

interface ListResponse {
  experiments: ExperimentDir[];
  count: number;
}

// Info icon component with tooltip
function InfoIcon({ text }: { text: string }) {
  return (
    <>
      <span className="info-icon" data-info={text}>i</span>
      <style jsx>{`
        .info-icon {
          position: relative;
          display: inline-block;
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: #666;
          color: #fff;
          font-size: 10px;
          line-height: 14px;
          text-align: center;
          cursor: default;
          font-family: sans-serif;
          font-style: normal;
          margin-left: 4px;
          vertical-align: middle;
        }
        .info-icon::after {
          content: attr(data-info);
          position: absolute;
          bottom: 125%;
          left: 50%;
          transform: translateX(-50%);
          background: #333;
          color: #fff;
          padding: 6px 8px;
          border-radius: 6px;
          font-size: 11px;
          white-space: nowrap;
          opacity: 0;
          pointer-events: none;
          transition: opacity 0.2s ease;
          z-index: 10;
        }
        .info-icon:hover::after {
          opacity: 1;
        }
      `}</style>
    </>
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
  const [showCommits, setShowCommits] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(false);

  const API_BASE_URL = "/api/proxy";

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Get experiment info
        const infoResponse = await fetch(`${API_BASE_URL}/info/${experiment.name}`);
        if (!infoResponse.ok) {
          throw new Error(`API error: ${infoResponse.status}`);
        }
        const infoData: ExperimentInfo = await infoResponse.json();
        setExpInfo(infoData);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [experiment]);

  // Don't hide card on error - show error state within the card instead

  // Display pre-computed sample text - strip metadata headers and reserved tokens
  const rawSample = sampleData?.sample_content || experiment.sample_output || "";
  const cleanSampleLines = rawSample.split('\n')
    .filter(line => !line.startsWith('#') && !line.startsWith('\u0000'))
    .join('\n')
    .trim();
  const cleanGeneratedText = stripAfterReserved(cleanSampleLines);

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

  // Fix malformed TSV data where header and data are on the same line
  const fixResultsTsv = (header: string[], rows: string[][]): { header: string[], rows: string[][] } => {
    if (rows.length > 0) {
      return { header, rows };
    }

    const knownHeaders = ['commit', 'val_bpb', 'timestamp', 'step', 'depth', 'vocab_size', 'model_dim', 'n_heads'];
    const firstHeader = header[0];

    if (firstHeader) {
      const firstValueIsTimestamp = /^\d{8}_\d{6}/.test(firstHeader);
      const firstValueIsKnownHeader = knownHeaders.includes(firstHeader);

      if (firstValueIsTimestamp || !firstValueIsKnownHeader) {
        const headerBoundary = header.findIndex(h => knownHeaders.includes(h));

        if (headerBoundary >= 0) {
          const realHeader = header.slice(headerBoundary);
          const dataRow = header.slice(0, headerBoundary);

          const descIdx = realHeader.indexOf('description');
          if (descIdx >= 0) {
            realHeader.splice(descIdx, 1);
            if (descIdx < dataRow.length) {
              dataRow.splice(descIdx, 1);
            }
          }

          return { header: realHeader, rows: dataRow.length > 0 ? [dataRow] : [] };
        }
      }
    }

    return { header, rows };
  };

  const fixedResults = expInfo?.results_data ? fixResultsTsv(expInfo.results_data.header, expInfo.results_data.rows) : { header: [], rows: [] };

  // Reverse rows so most recent appears first
  const reversedResults = { header: fixedResults.header, rows: [...fixedResults.rows].reverse() };

  // Extract val_bpb from results (first row after reverse = most recent)
  const getValBpb = (): number | null => {
    if (!reversedResults.header.length || !reversedResults.rows.length) return null;
    const valBpbIdx = reversedResults.header.indexOf('val_bpb');
    if (valBpbIdx === -1) return null;
    const firstRow = reversedResults.rows[0];
    const val = firstRow?.[valBpbIdx];
    return val ? parseFloat(val) : null;
  };
  const valBpb = getValBpb();

  // Use experiment.name for timestamp if expInfo not loaded yet
  const timestamp = expInfo ? parseTimestamp(expInfo.exp_dir) : parseTimestamp(experiment.name);
  const latestCommit = expInfo?.git_log?.[0];

  return (
    <div style={styles.container}>
      <div style={{display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px"}}>
        <div style={{display: "flex", alignItems: "center", gap: "8px"}}>
          {timestamp.date || timestamp.full ? (
            <h1 style={{...styles.title, margin: 0}}>
              {timestamp.date ? `${timestamp.date} ${timestamp.time ? `@ ${timestamp.time}` : ''}` : timestamp.full}
            </h1>
          ) : (
            <h1 style={{...styles.title, margin: 0}}>Experiment: {experiment.name}</h1>
          )}
        </div>
        {valBpb !== null && (
          <div style={{fontSize: "12px", color: "#666", textAlign: "right"}}>
            <span>val_bpb</span><InfoIcon text="Validation bits per byte - lower is better" />: {valBpb.toFixed(3)}
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
          {cleanGeneratedText && (
            <div style={styles.result}>
              <div style={styles.row}>
                <strong>Sample Output:</strong>
              </div>
              <div style={{...styles.row, whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '11px'}}>
                {cleanGeneratedText}
              </div>
            </div>
          )}

          {expInfo && (
            <div style={styles.metadataSection}>
              {/* Model Info from sample_output */}
              {sampleData?.model_info && (
                <div style={{marginBottom: "8px"}}>
                  <button
                    style={styles.dropdownButton}
                    onClick={() => setShowModelInfo(!showModelInfo)}
                  >
                    <span><strong>Model Info</strong></span>
                    <span style={styles.arrow}>{showModelInfo ? "▼" : "▶"}</span>
                  </button>
                  {showModelInfo && (
                    <div style={{...styles.dropdownContent, whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '11px'}}>
                      {sampleData.model_info}
                    </div>
                  )}
                </div>
              )}

              {latestCommit && (
                <div>
                  <button
                    style={styles.dropdownButton}
                    onClick={() => setShowCommits(!showCommits)}
                  >
                    <span>
                      <strong>Latest Commit:</strong> {latestCommit.commit_short} — {latestCommit.message.trim()}
                    </span>
                    <span style={styles.arrow}>{showCommits ? "▼" : "▶"}</span>
                  </button>
                  {showCommits && expInfo.git_log && (
                    <div style={styles.dropdownContent}>
                      {expInfo.git_log.map((commit, idx) => (
                        <div key={idx} style={styles.commitItem}>
                          <div><strong>{commit.commit_short}</strong> - {commit.commit_sha}</div>
                          <div style={{color: "#666"}}>
                            {commit.author} — {commit.date}
                          </div>
                          <div style={{marginTop: "2px"}}>{commit.message.trim()}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {expInfo.has_results && expInfo.results_data && (
                <div style={{marginTop: "8px"}}>
                  <button
                    style={styles.dropdownButton}
                    onClick={() => setShowResults(!showResults)}
                  >
                    <span><strong>Results.tsv</strong></span>
                    <span style={styles.arrow}>{showResults ? "▼" : "▶"}</span>
                  </button>
                  {showResults && (
                    <div style={styles.dropdownContent}>
                      <table style={styles.table}>
                        <thead>
                          <tr>
                            {reversedResults.header.map((h, idx) => (
                              <th key={idx} style={styles.th}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {reversedResults.rows.map((row, rowIdx) => (
                            <tr key={rowIdx}>
                              {row.map((cell, cellIdx) => (
                                <td key={cellIdx} style={styles.td}>{cell}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* Metrics removed - sample output is pre-computed */}
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

            setSampleData(prev => ({ ...prev, [exp.name]: sample }));
            if (valBpb !== null) {
              setValBpbData(prev => ({ ...prev, [exp.name]: valBpb }));
            }
          } catch (err) {
            console.error(`Failed to load data for ${exp.name}:`, err);
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
                key={exp.name}
                experiment={exp}
                prevValBpb={idx < experiments.length - 1 ? valBpbData[experiments[idx + 1].name] : undefined}
                sampleData={sampleData[exp.name]}
              />
            ))}
          </>
        )}
      </div>
    </main>
  );
}