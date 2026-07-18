export function markdownPieChart(title: string, labels: string[], values: number[]) {
  const rows = labels
    .map((label, index) => ({ label: cleanLabel(label), value: finite(values[index]) }))
    .filter((row) => row.value > 0)
    .slice(0, 12);
  if (!rows.length) return "_Chart unavailable: no values were generated._";
  return [
    "```mermaid",
    "pie showData",
    `    title ${cleanTitle(title)}`,
    ...rows.map((row) => `    \"${row.label}\" : ${round(row.value)}`),
    "```",
  ].join("\n");
}

export function markdownBarChart(title: string, labels: string[], values: number[], axisLabel = "Value") {
  const rows = labels
    .map((label, index) => ({ label: cleanLabel(label), value: finite(values[index]) }))
    .slice(0, 12);
  if (!rows.length) return "_Chart unavailable: no values were generated._";
  const max = Math.max(...rows.map((row) => row.value), 1);
  return [
    "```mermaid",
    "xychart-beta",
    `    title \"${cleanTitle(title)}\"`,
    `    x-axis [${rows.map((row) => `\"${row.label}\"`).join(", ")}]`,
    `    y-axis \"${cleanTitle(axisLabel)}\" 0 --> ${round(max * 1.1)}`,
    `    bar [${rows.map((row) => round(row.value)).join(", ")}]`,
    "```",
  ].join("\n");
}

function cleanLabel(value: string) {
  return String(value).replace(/[\"\n\r]/g, "'").slice(0, 24);
}

function cleanTitle(value: string) {
  return String(value).replace(/[\"\n\r]/g, "'").slice(0, 72);
}

function finite(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, number) : 0;
}

function round(value: number) {
  return Math.round(value * 1000) / 1000;
}
