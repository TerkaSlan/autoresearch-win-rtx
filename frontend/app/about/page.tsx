import Link from "next/link";
import type { CSSProperties } from "react";

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
  navLink: {
    fontSize: "11px",
    color: "#666",
    textDecoration: "none",
  },
  authorLink: {
    fontSize: "11px",
    color: "#666",
    textDecoration: "none",
  },
  content: {
    maxWidth: "800px",
    width: "100%",
    margin: "0 auto",
    paddingTop: "45px",
  },
  title: {
    fontSize: "18px",
    fontWeight: "bold",
    marginBottom: "16px",
  },
  text: {
    fontSize: "12px",
    lineHeight: "1.6",
    color: "#666",
    marginBottom: "12px",
  },
  link: {
    color: "#666",
    textDecoration: "underline",
  },
  code: {
    fontFamily: "'Courier New', monospace",
    backgroundColor: "#e8e8e8",
    padding: "2px 4px",
    borderRadius: "3px",
    fontSize: "11px",
  },
  details: {
    marginTop: "20px",
  },
  detail: {
    fontSize: "12px",
    lineHeight: "1.6",
    color: "#666",
    marginBottom: "8px",
  },
};

export default function About() {
  return (
    <main style={styles.main}>
      <Header />
      <div style={styles.content}>
        <h1 style={styles.title}>About</h1>
        <p style={styles.text}>
          This app documents the training progress of fully autonomous agentic training of a language model (<a href="https://github.com/karpathy/nanochat" target="_blank" rel="noopener noreferrer" style={styles.link}>nanochat</a>) from scratch.
        </p>
        <p style={styles.text}>
          The training is done on <a href="https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean" target="_blank" rel="noopener noreferrer" style={styles.link}>TinyStories</a> and the objective is text generation.
        </p>
        <p style={styles.text}>
          See <a href="https://github.com/karpathy/autoresearch" target="_blank" rel="noopener noreferrer" style={styles.link}>github.com/karpathy/autoresearch</a> for more details.
        </p>
        <div style={styles.details}>
          <p style={styles.detail}><strong>Dataset:</strong> TinyStories</p>
          <p style={styles.detail}><strong>GPU:</strong> NVIDIA RTX PRO 6000 Blackwell Server Edition (95GB VRAM)</p>
        </div>
      </div>
    </main>
  );
}

function Header() {
  return (
    <header style={styles.header}>
      <div style={styles.headerContent}>
        <div style={styles.headerText}>Monitoring autoresearch progress</div>
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
            <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55-.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path>
          </svg>
        </a>
        <span style={styles.headerDivider}>|</span>
        <Link href="/" style={styles.navLink}>Home</Link>
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