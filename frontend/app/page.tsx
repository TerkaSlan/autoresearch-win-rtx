"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { CSSProperties } from "react";

// Metrics computation
interface Metrics {
  rep3gram: number;
  uniqueRatio: number;
  sheStartRatio: number;
  constraintScore: number;
  hasEnding: number;
  objectTracking: number;
  numTokens: number;
  numSentences: number;
  constraintDetails: boolean[];
}

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/\b\w+\b/g) || []);
}

function splitSentences(text: string): string[] {
  return text
    .split(/[.!?]+/)
    .map(s => s.trim())
    .filter(Boolean);
}

function repetition3gram(tokens: string[]): number {
  if (tokens.length < 3) return 0;

  const counts = new Map<string, number>();
  for (let i = 0; i < tokens.length - 2; i++) {
    const tri = `${tokens[i]} ${tokens[i + 1]} ${tokens[i + 2]}`;
    counts.set(tri, (counts.get(tri) || 0) + 1);
  }

  let repeated = 0;
  counts.forEach(c => {
    if (c > 1) repeated += c;
  });

  const total = tokens.length - 2;
  return repeated / total;
}

function uniqueRatio(tokens: string[]): number {
  if (!tokens.length) return 0;
  return new Set(tokens).size / tokens.length;
}

function sheStartRatio(sentences: string[]): number {
  if (!sentences.length) return 0;
  const count = sentences.filter(s =>
    s.toLowerCase().startsWith("she")
  ).length;
  return count / sentences.length;
}

function constraintScore(text: string): [number, boolean[]] {
  const t = text.toLowerCase();
  const checks: boolean[] = [
    t.includes("red key"),
    t.includes("blue door"),
    (t.match(/\b(tried|first|second|third)\b/g) || []).length >= 3,
    ["learned", "realized"].some(w => t.includes(w))
  ];
  const score = checks.filter(Boolean).length / checks.length;
  return [score, checks];
}

function hasEnding(sentences: string[]): number {
  if (!sentences.length) return 0;
  const last = sentences[sentences.length - 1].toLowerCase();
  const keywords = ["learned", "finally", "in the end", "was happy", "realized"];
  return keywords.some(k => last.includes(k)) ? 1 : 0;
}

function objectTracking(tokens: string[]): number {
  const countKey = tokens.filter(t => t === "key").length;
  const countDoor = tokens.filter(t => t === "door").length;
  if (countKey === 0 && countDoor === 0) return 0;
  return Math.min(countKey, countDoor) / Math.max(countKey, countDoor);
}

function evaluate(text: string): Metrics {
  const tokens = tokenize(text);
  const sentences = splitSentences(text);
  return {
    rep3gram: repetition3gram(tokens),
    uniqueRatio: uniqueRatio(tokens),
    sheStartRatio: sheStartRatio(sentences),
    constraintScore: constraintScore(text)[0],
    hasEnding: hasEnding(sentences),
    objectTracking: objectTracking(tokens),
    numTokens: tokens.length,
    numSentences: sentences.length,
    constraintDetails: constraintScore(text)[1]
  };
}

// Compute comparison between current and previous metrics
function computeComparison(current: Metrics, prev: Metrics): Record<string, { value: number; diff: number; diffPercent: number; isImprovement: boolean }> {
  const fields: (keyof Metrics)[] = ['rep3gram', 'uniqueRatio', 'sheStartRatio', 'constraintScore', 'hasEnding', 'objectTracking', 'numTokens', 'numSentences'];

  const result: Record<string, any> = {};

  fields.forEach(field => {
    const curr = current[field];
    const p = prev[field];

    // For numTokens and numSentences, higher is usually better (more content)
    // For rep3gram, lower is better (less repetition)
    // For uniqueRatio, sheStartRatio, constraintScore, objectTracking, higher is better
    // For hasEnding (0/1), higher is better

    let isBetter: boolean;
    if (field === 'rep3gram') {
      // Lower is better for repetition
      isBetter = curr < p;
    } else {
      // Higher is better for everything else
      isBetter = curr > p;
    }

    const diff = curr instanceof Number || typeof curr === 'number' ? Number(curr) - Number(p) : 0;
    const prevNum = Number(p);
    const diffPercent = prevNum !== 0 ? (diff / Math.abs(prevNum)) * 100 : (Number(curr) > 0 ? 100 : 0);

    result[String(field)] = {
      value: curr,
      diff: diff,
      diffPercent: diffPercent,
      isImprovement: isBetter
    };
  });

  return result;
}

// Format comparison display with color and direction indicator
// isLowerBetter: if true, lower values are better (e.g., rep3gram, sheStartRatio)
// isHigherBetter: if true, higher values are better (e.g., uniqueRatio, constraintScore)
function formatComparison(current: number, previous: number, isLowerBetter: boolean): React.ReactNode {
  const prevNum = Number(previous);
  const currNum = Number(current);
  const diff = currNum - prevNum;
  const diffPercent = prevNum !== 0 ? (diff / Math.abs(prevNum)) * 100 : (currNum > 0 ? 100 : 0);

  // Determine if this is improvement based on whether lower or higher is better
  const isImprovement = isLowerBetter ? diff < 0 : diff > 0;
  const isNeutral = diff === 0;

  const sign = diffPercent > 0 ? "+" : "";
  const color = isNeutral ? "gray" : (isImprovement ? "green" : "red");
  const statusText = isNeutral ? "no change" : (isImprovement ? "improvement" : "deterioration");
  const direction = isNeutral ? "●" : (isImprovement ? "▲" : "▼");

  // Special handling for hasEnding (binary 0/1)
  if ((previous === 0 || previous === 1) && (current === 0 || current === 1)) {
    if (previous === 0 && current === 1) {
      return <span style={{ fontSize: "10px", color: "green" }}>(+100% improvement ▲)</span>;
    }
    if (previous === 1 && current === 0) {
      return <span style={{ fontSize: "10px", color: "red" }}>(-100% deterioration ▼)</span>;
    }
    // Same value - no change
    return <span style={{ fontSize: "10px", color: "gray" }}>(0% no change ●)</span>;
  }

  return (
    <span style={{ fontSize: "10px", color }}>
      ({sign}{diffPercent.toFixed(1)}% {statusText} {direction})
    </span>
  );
}

interface GenerateResponse {
  generated_text: string;
  generated_tokens: number[];
  num_tokens: number;
  prompt_used: string | null;
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
}

interface ListResponse {
  experiments: ExperimentDir[];
  count: number;
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
  prevMetrics?: Metrics;
  prevValBpb?: number;
  currentData?: { metrics: Metrics; result: GenerateResponse };
  collapsed?: boolean;
}

function ExperimentCard({ experiment, prevMetrics, prevValBpb, currentData, collapsed = false }: ExperimentCardProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expInfo, setExpInfo] = useState<ExperimentInfo | null>(null);
  const [showCommits, setShowCommits] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);
  const [isExpanded, setIsExpanded] = useState(!collapsed);

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

  // Return null if there's an error to hide the card
  if (error) {
    return null;
  }

  const cleanGeneratedText = currentData?.result?.generated_text?.replace("<|reserved_0|>", "") ?? "";

  // Parse timestamp from exp_dir (format: YYYYMMDD_HHMMSS-<commit_short>)
  const parseTimestamp = (expDir: string): { full: string; date: string; time: string } => {
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

  // Extract val_bpb from results
  const getValBpb = (): number | null => {
    if (!fixedResults.header.length || !fixedResults.rows.length) return null;
    const valBpbIdx = fixedResults.header.indexOf('val_bpb');
    if (valBpbIdx === -1) return null;
    const lastRow = fixedResults.rows[fixedResults.rows.length - 1];
    const val = lastRow?.[valBpbIdx];
    return val ? parseFloat(val) : null;
  };
  const valBpb = getValBpb();

  const timestamp = expInfo ? parseTimestamp(expInfo.exp_dir) : { full: "", date: "", time: "" };
  const latestCommit = expInfo?.git_log?.[0];

  // Collapsed view - just header with expand button
  if (collapsed && !isExpanded) {
    return (
      <div style={{...styles.container, cursor: "pointer"}} onClick={() => setIsExpanded(true)}>
        <div style={{display: "flex", justifyContent: "space-between", alignItems: "center"}}>
          <div style={{display: "flex", alignItems: "center", gap: "8px"}}>
            <span style={{color: "#888", fontSize: "12px"}}>▶</span>
            {timestamp.date || timestamp.full ? (
              <span style={{fontSize: "14px", fontWeight: 500}}>
                {timestamp.date ? `${timestamp.date} ${timestamp.time ? `@ ${timestamp.time}` : ''}` : timestamp.full}
              </span>
            ) : (
              <span style={{fontSize: "14px", fontWeight: 500}}>Experiment: {experiment.name}</span>
            )}
          </div>
          {valBpb !== null && (
            <div style={{fontSize: "12px", color: "#666"}}>
              val_bpb: {valBpb.toFixed(3)}
              {prevValBpb !== undefined && prevValBpb !== null && formatComparison(valBpb, prevValBpb, true)}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={{display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px"}}>
        <div style={{display: "flex", alignItems: "center", gap: "8px"}}>
          {collapsed && (
            <span
              style={{color: "#888", fontSize: "12px", cursor: "pointer"}}
              onClick={() => setIsExpanded(false)}
              title="Collapse"
            >
              ▼
            </span>
          )}
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
            <span title="Validation bits per byte - lower is better" style={{cursor: "help"}}>val_bpb</span>: {valBpb.toFixed(3)}
            {prevValBpb !== undefined && prevValBpb !== null && formatComparison(valBpb, prevValBpb, true)}
          </div>
        )}
      </div>

      {loading && (
        <div style={styles.box}>
          <p>Loading...</p>
        </div>
      )}

      {!loading && currentData && (
        <>
          <div style={styles.result}>
            <div style={styles.row}>
              <strong>Prompt:</strong> {currentData.result.prompt_used}
            </div>
            <div style={styles.row}>
              <strong>Generated:</strong> {cleanGeneratedText}
            </div>
          </div>

          {expInfo && (
            <div style={styles.metadataSection}>
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
                            {fixedResults.header.map((h, idx) => (
                              <th key={idx} style={styles.th}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {fixedResults.rows.map((row, rowIdx) => (
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

              {currentData && (
                <div style={{marginTop: "8px"}}>
                  <button
                    style={styles.dropdownButton}
                    onClick={() => setShowMetrics(!showMetrics)}
                  >
                    <span><strong>Metrics</strong></span>
                    <span style={styles.arrow}>{showMetrics ? "▼" : "▶"}</span>
                  </button>
                  {showMetrics && (
                    <div style={styles.dropdownContent}>
                      {prevMetrics ? (
                        <>
                          <div style={styles.row}><strong>rep3gram</strong> <span title="Measures text repetition by counting repeated 3-word sequences. Lower values indicate less repetitive, more natural text." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.rep3gram.toFixed(4)} {formatComparison(currentData.metrics.rep3gram, prevMetrics.rep3gram, true)}</div>
                          <div style={styles.row}><strong>uniqueRatio</strong> <span title="Ratio of unique words to total words. Values between 0.4-0.7 indicate healthy vocabulary diversity without being too repetitive or too random." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.uniqueRatio.toFixed(4)} {formatComparison(currentData.metrics.uniqueRatio, prevMetrics.uniqueRatio, false)}</div>
                          <div style={styles.row}><strong>sheStartRatio</strong> <span title="Fraction of sentences starting with 'She'. Lower values indicate more varied sentence structure and better writing quality." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.sheStartRatio.toFixed(4)} {formatComparison(currentData.metrics.sheStartRatio, prevMetrics.sheStartRatio, true)}</div>
                          <div style={styles.row}><strong>constraintScore</strong> <span title="How many prompt constraints were satisfied: red key, blue door, tried/first/second/third, learned/realized. 1.0 means all 4 constraints met." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.constraintScore.toFixed(4)} {formatComparison(currentData.metrics.constraintScore, prevMetrics.constraintScore, false)}</div>
                          <div style={styles.row}><strong>objectTracking</strong> <span title="Measures balance between 'key' and 'door' mentions. Values closer to 1 indicate the model properly tracks and references both objects equally." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.objectTracking.toFixed(4)} {formatComparison(currentData.metrics.objectTracking, prevMetrics.objectTracking, false)}</div>
                          <div style={styles.row}><strong>numSentences</strong> <span title="Total number of sentences generated. Helps track output length consistency across experiments." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.numSentences} {formatComparison(currentData.metrics.numSentences, prevMetrics.numSentences, false)}</div>
                        </>
                      ) : (
                        <>
                          <div style={styles.row}><strong>rep3gram</strong> <span title="Measures text repetition by counting repeated 3-word sequences. Lower values indicate less repetitive, more natural text." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.rep3gram.toFixed(4)}</div>
                          <div style={styles.row}><strong>uniqueRatio</strong> <span title="Ratio of unique words to total words. Values between 0.4-0.7 indicate healthy vocabulary diversity without being too repetitive or too random." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.uniqueRatio.toFixed(4)}</div>
                          <div style={styles.row}><strong>sheStartRatio</strong> <span title="Fraction of sentences starting with 'She'. Lower values indicate more varied sentence structure and better writing quality." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.sheStartRatio.toFixed(4)}</div>
                          <div style={styles.row}><strong>constraintScore</strong> <span title="How many prompt constraints were satisfied: red key, blue door, tried/first/second/third, learned/realized. 1.0 means all 4 constraints met." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.constraintScore.toFixed(4)}</div>
                          <div style={styles.row}><strong>objectTracking</strong> <span title="Measures balance between 'key' and 'door' mentions. Values closer to 1 indicate the model properly tracks and references both objects equally." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.objectTracking.toFixed(4)}</div>
                          <div style={styles.row}><strong>numSentences</strong> <span title="Total number of sentences generated. Helps track output length consistency across experiments." style={{cursor: "help", color: "#888", fontSize: "11px", marginLeft: "2px"}}>ⓘ</span>: {currentData.metrics.numSentences}</div>
                        </>
                      )}
                      <div style={{marginTop: "6px", paddingTop: "6px", borderTop: "1px solid #eee"}}>
                        <strong>constraintDetails:</strong>
                        <div style={{fontSize: "10px", color: "#666", marginTop: "4px"}}>
                          <span style={{color: currentData.metrics.constraintDetails[0] ? "green" : "gray"}}>red key: {currentData.metrics.constraintDetails[0] ? "✓" : "✗"}</span> |
                          <span style={{color: currentData.metrics.constraintDetails[1] ? "green" : "gray"}}>blue door: {currentData.metrics.constraintDetails[1] ? "✓" : "✗"}</span> |
                          <span style={{color: currentData.metrics.constraintDetails[2] ? "green" : "gray"}}>tried/first/second/third: {currentData.metrics.constraintDetails[2] ? "✓" : "✗"}</span> |
                          <span style={{color: currentData.metrics.constraintDetails[3] ? "green" : "gray"}}>learned/realized: {currentData.metrics.constraintDetails[3] ? "✓" : "✗"}</span>
                        </div>
                      </div>
                    </div>
                  )}
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
  const [experimentData, setExperimentData] = useState<{[key: string]: { metrics: Metrics; result: GenerateResponse }}>({});
  const [valBpbData, setValBpbData] = useState<{[key: string]: number}>({});

  useEffect(() => {
    const fetchExperiments = async () => {
      try {
        const response = await fetch("/api/proxy/list");
        if (!response.ok) {
          throw new Error(`Failed to fetch experiments: ${response.status}`);
        }
        const data: ListResponse = await response.json();
        const filteredExperiments = data.experiments?.filter(exp => exp.has_checkpoint) || [];

        // Display experiments immediately
        setExperiments(filteredExperiments);
        setLoading(false);

        // Then fetch metrics data in the background
        const API_BASE_URL = "/api/proxy";
        const promises = filteredExperiments.map(async (exp) => {
          try {
            // Fetch both generate and info in parallel
            const [genResponse, infoResponse] = await Promise.all([
              fetch(`${API_BASE_URL}/generate`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  prompt: "Once upon a time, there was a girl named Lily. She found a red key and a locked blue door.\nShe tried to open it. First she pushed it. Second she pulled it. Third she used the key.\nIn the end, she learned patience.",
                  max_tokens: 100,
                  temperature: 0.8,
                  checkpoint: exp.checkpoint_path,  // Use specific checkpoint for each experiment
                }),
              }),
              fetch(`${API_BASE_URL}/info/${exp.name}`)
            ]);

            let genData = null;
            let valBpb = null;

            if (genResponse.ok) {
              genData = await genResponse.json();
            }

            if (infoResponse.ok) {
              const infoData = await infoResponse.json();
              if (infoData.results_data?.header && infoData.results_data?.rows) {
                const valBpbIdx = infoData.results_data.header.indexOf('val_bpb');
                if (valBpbIdx !== -1 && infoData.results_data.rows.length > 0) {
                  const lastRow = infoData.results_data.rows[infoData.results_data.rows.length - 1];
                  const val = lastRow?.[valBpbIdx];
                  if (val) valBpb = parseFloat(val);
                }
              }
            }

            if (genData) {
              const genText = genData.generated_text?.replace("<|reserved_0|>", "") ?? "";
              const metrics = evaluate(genText);
              return { exp, metrics, result: genData, valBpb };
            }
          } catch (err) {
            console.error(`Failed to fetch data for ${exp.name}:`, err);
          }
          return null;
        });

        const results = await Promise.all(promises);
        const newData: {[key: string]: { metrics: Metrics; result: GenerateResponse }} = {};
        const newvalBpbData: {[key: string]: number} = {};

        for (const result of results) {
          if (result) {
            newData[result.exp.name] = { metrics: result.metrics, result: result.result };
            if (result.valBpb !== null && result.valBpb !== undefined) {
              newvalBpbData[result.exp.name] = result.valBpb;
            }
          }
        }

        setExperimentData(newData);
        setValBpbData(newvalBpbData);
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
                // prevMetrics should be the next experiment (older) since newest is at index 0
                prevMetrics={idx < experiments.length - 1 ? experimentData[experiments[idx + 1].name]?.metrics : undefined}
                prevValBpb={idx < experiments.length - 1 ? valBpbData[experiments[idx + 1].name] : undefined}
                currentData={experimentData[exp.name]}
                collapsed={idx >= 5}
              />
            ))}
          </>
        )}
      </div>
    </main>
  );
}