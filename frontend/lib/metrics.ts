export type Metrics = {
  rep3gram: number;
  uniqueRatio: number;
  sheStartRatio: number;
  constraintScore: number;
  hasEnding: number;
  objectTracking: number;
  numTokens: number;
  numSentences: number;
  constraintDetails: boolean[];
};

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

export function evaluate(text: string): Metrics {
  const tokens = tokenize(text);
  const sentences = splitSentences(text);

  const rep3gram = repetition3gram(tokens);
  const unique = uniqueRatio(tokens);
  const sheRatio = sheStartRatio(sentences);
  const [constraint, details] = constraintScore(text);
  const ending = hasEnding(sentences);
  const objTrack = objectTracking(tokens);

  return {
    rep3gram,
    uniqueRatio: unique,
    sheStartRatio: sheRatio,
    constraintScore: constraint,
    hasEnding: ending,
    objectTracking: objTrack,
    numTokens: tokens.length,
    numSentences: sentences.length,
    constraintDetails: details
  };
}
